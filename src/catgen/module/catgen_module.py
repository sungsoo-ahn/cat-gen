import os
import json
import time
import tempfile
from typing import Any, Optional

import torch
import numpy as np
import wandb
from ase.io import write as ase_write
from einops import rearrange
from lightning import Callback, LightningModule
from lightning.pytorch.utilities import rank_zero_info
from torch import Tensor
from torchmetrics import MeanMetric
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm

from src.catgen.data.prior import CatPriorSampler
from src.catgen.data.conversions import virtual_coords_to_lattice_and_supercell
from src.catgen.module.parallel_utils import (
    handle_cuda_oom,
    expand_tensor_for_multiplicity,
    run_parallel_tasks,
)
from src.catgen.module.lr_schedulers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from src.catgen.models.loss.validation import (
    find_best_match_rmsd_prim,
    find_best_match_rmsd_slab,
    find_train_match_prim,
    find_train_match_slab,
    compute_structural_validity_single,
    compute_prim_structural_validity_single,
    compute_comprehensive_validity_single,
    compute_adsorption_and_validity_single,
    get_uma_calculator,
)
from src.catgen.models.loss.utils import stratify_loss_by_time
from src.catgen.models.utils import ExponentialMovingAverage
from src.catgen.module.flow import AtomFlowMatching
from src.catgen.scripts.assemble import assemble


class CombinedMuonAdamW(torch.optim.Optimizer):
    """Combined optimizer that wraps Muon and AdamW for Lightning compatibility.

    Lightning requires a single optimizer for automatic optimization.
    This wrapper combines Muon (for 2D weight matrices) and AdamW (for other params)
    into a single optimizer interface.
    """

    def __init__(self, muon_optimizer, adamw_optimizer):
        """Initialize combined optimizer.

        Parameters
        ----------
        muon_optimizer : Muon
            Muon optimizer for 2D weight matrices.
        adamw_optimizer : AdamW
            AdamW optimizer for biases, norms, embeddings.
        """
        self.muon = muon_optimizer
        self.adamw = adamw_optimizer

        # Combine param_groups from both optimizers
        self.param_groups = self.muon.param_groups + self.adamw.param_groups

        # Track defaults (use AdamW defaults as base)
        self.defaults = self.adamw.defaults.copy()

        # State dict combines both
        self._state = {}

    @property
    def state(self):
        """Combined state from both optimizers."""
        combined = {}
        combined.update(self.muon.state)
        combined.update(self.adamw.state)
        return combined

    @state.setter
    def state(self, value):
        """Set state (for checkpoint loading)."""
        self._state = value

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for both optimizers."""
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """Step both optimizers."""
        loss = None
        if closure is not None:
            loss = closure()

        self.muon.step()
        self.adamw.step()

        return loss

    def state_dict(self):
        """Return combined state dict."""
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load combined state dict."""
        if "muon" in state_dict:
            self.muon.load_state_dict(state_dict["muon"])
        if "adamw" in state_dict:
            self.adamw.load_state_dict(state_dict["adamw"])


class CatGen(LightningModule):
    def __init__(
        self,
        atom_s: int,
        token_s: int,
        flow_model_args: dict[str, Any],
        training_args: dict[str, Any],
        validation_args: dict[str, Any],
        flow_process_args: dict[str, Any],
        prior_sampler_args: dict[str, Any],
        predict_args: Optional[dict[str, Any]] = None,
        use_kernels: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Args
        self.training_args = training_args
        self.validation_args = validation_args
        self.predict_args = predict_args

        # Kernels
        self.use_kernels = use_kernels

        # Validation metrics
        self.matcher_kwargs = StructureMatcher(
            **self.validation_args["structure_matcher_args"]
        ).as_dict()
        self.val_prim_match_rate = MeanMetric(sync_on_compute=False)
        self.val_prim_rmsd = MeanMetric(sync_on_compute=False)
        self.val_slab_match_rate = MeanMetric(sync_on_compute=False)
        self.val_slab_rmsd = MeanMetric(sync_on_compute=False)
        self.val_adsorption_energy = MeanMetric(sync_on_compute=False)
        self.val_structural_validity = MeanMetric(sync_on_compute=False)
        self.val_prim_structural_validity = MeanMetric(sync_on_compute=False)
        self.val_smact_validity = MeanMetric(sync_on_compute=False)
        self.val_crystal_validity = MeanMetric(sync_on_compute=False)
        self.validation_step_outputs = []

        # Training match rate metrics (compare generated samples against training set)
        # This helps detect if model is memorizing training data
        self.val_train_prim_match_rate = MeanMetric(sync_on_compute=False)
        self.val_train_slab_match_rate = MeanMetric(sync_on_compute=False)
        self._train_sample_cache = []  # Cache training samples for match rate comparison
        self._max_train_cache_size = self.validation_args.get("max_train_cache_size", 500)
        
        # UMA calculator for adsorption energy (lazy initialization)
        self._uma_calculator = None

        # Prior sampler for flow matching
        prior_sampler = CatPriorSampler(**prior_sampler_args)

        # Flow matching module
        # Pass train_multiplicity as fixed_prior_multiplicity for deterministic overfitting tests
        flow_kwargs = dict(flow_process_args)
        if training_args.get("train_multiplicity") and flow_process_args.get("fixed_prior_seed"):
            flow_kwargs["fixed_prior_multiplicity"] = training_args["train_multiplicity"]

        self.structure_module = AtomFlowMatching(
            flow_model_args={
                "atom_s": atom_s,
                "token_s": token_s,
                **flow_model_args,
            },
            prior_sampler=prior_sampler,
            **flow_kwargs,
        )

        # EMA (Exponential Moving Average) for improved generation quality
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self._ema: Optional[ExponentialMovingAverage] = None
        if self.use_ema:
            # EMA will be initialized after model is moved to device (in setup or first training step)
            rank_zero_info(f"| EMA enabled with decay={ema_decay}")

    def setup(self, stage: str) -> None:
        """Set the model for training, validation and inference."""
        if stage == "predict" and not (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(torch.device("cuda")).major >= 8
        ):
            self.use_kernels = False

        # Initialize EMA if enabled (after model is on device)
        if self.use_ema and self._ema is None:
            self._ema = ExponentialMovingAverage(
                self.structure_module.parameters(),
                decay=self.ema_decay,
            )
            # Move shadow params to the same device as the model
            self._ema.to(self.device)
            rank_zero_info(f"| EMA initialized with {len(self._ema.shadow_params)} shadow parameters on {self.device}")

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Called when checkpoint is loaded. Use this to verify optimizer state."""
        if self.global_rank == 0:  # Only log on rank 0
            if "optimizer_states" in checkpoint:
                opt_states = checkpoint["optimizer_states"]
                if len(opt_states) > 0:
                    opt_state = opt_states[0]
                    if "state" in opt_state and len(opt_state["state"]) > 0:
                        rank_zero_info(
                            f"| [CHECKPOINT] Optimizer state loaded: {len(opt_state['state'])} parameters"
                        )
                        # Check first parameter's state
                        first_state = list(opt_state["state"].values())[0]
                        if "exp_avg" in first_state:
                            rank_zero_info(
                                f"| [CHECKPOINT] AdamW momentum buffers found (exp_avg, exp_avg_sq)"
                            )
                        if "param_groups" in opt_state and len(opt_state["param_groups"]) > 0:
                            lr = opt_state["param_groups"][0].get("lr", "N/A")
                            rank_zero_info(f"| [CHECKPOINT] Learning rate from checkpoint: {lr}")
                    else:
                        rank_zero_info("| [WARNING] Optimizer state exists but is empty")
                else:
                    rank_zero_info("| [WARNING] Optimizer states list is empty")
            else:
                rank_zero_info("| [WARNING] No optimizer_states found in checkpoint - starting fresh")

            if "global_step" in checkpoint:
                rank_zero_info(f"| [CHECKPOINT] Resuming from global_step: {checkpoint['global_step']}")
            if "epoch" in checkpoint:
                rank_zero_info(f"| [CHECKPOINT] Resuming from epoch: {checkpoint['epoch']}")

        # Load EMA state if present
        if self.use_ema and "ema_state" in checkpoint:
            if self._ema is None:
                # Initialize EMA first
                self._ema = ExponentialMovingAverage(
                    self.structure_module.parameters(),
                    decay=self.ema_decay,
                )
            self._ema.load_state_dict(checkpoint["ema_state"], device=self.device)
            rank_zero_info("| [CHECKPOINT] EMA state loaded")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save EMA state to checkpoint."""
        if self.use_ema and self._ema is not None:
            checkpoint["ema_state"] = self._ema.state_dict()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """Update EMA after each training step (after optimizer step)."""
        if self.use_ema and self._ema is not None:
            self._ema.update(self.structure_module.parameters())

    def on_validation_epoch_start(self) -> None:
        """Swap to EMA weights before validation."""
        if self.use_ema and self._ema is not None:
            self._ema.store(self.structure_module.parameters())
            self._ema.copy_to(self.structure_module.parameters())

    def forward(
        self,
        feats: dict[str, Tensor],
        # Training arguments
        multiplicity_flow_train: int = 1,
        # Sampling arguments
        multiplicity_flow_sample: int = 1,
        num_sampling_steps: Optional[int] = None,
        center_during_sampling: bool = False,
        refine_final: bool = False,
        return_trajectory: bool = False,
    ) -> dict[str, Any]:

        dict_out = {}

        if self.training:
            # Returns denoised predictions for loss computation
            dict_out.update(
                self.structure_module(feats=feats, multiplicity=multiplicity_flow_train)
            )
        else:
            # Returns sampled structure
            prim_slab_atom_mask = feats["prim_slab_atom_pad_mask"]
            ads_atom_mask = feats["ads_atom_pad_mask"]

            network_condition_kwargs = dict(
                feats=feats,
            )

            dict_out.update(
                self.structure_module.sample(
                    prim_slab_atom_mask=prim_slab_atom_mask,
                    ads_atom_mask=ads_atom_mask,
                    num_sampling_steps=num_sampling_steps,
                    multiplicity=multiplicity_flow_sample,
                    center_coords=False,
                    refine_final=refine_final,
                    return_trajectory=return_trajectory,
                    **network_condition_kwargs,
                )
            )

        return dict_out

    def _compute_and_log_loss(
        self, batch: dict[str, Tensor], prefix: str
    ) -> tuple[Tensor, dict, dict]:
        # Compute the forward pass
        out = self.structure_module(
            feats=batch,
            multiplicity=self.training_args["train_multiplicity"],
        )

        # Store timesteps for monitoring callback
        self._last_timesteps = out.get("times", None)

        # Compute losses
        flow_loss_dict, check_dict = self.structure_module.compute_loss(
            batch,
            out,
            multiplicity=self.training_args["train_multiplicity"],
            loss_type=self.training_args["loss_type"],
            loss_space=self.training_args.get("loss_space", "raw"),
        )

        # Calculate total weighted loss
        total_loss = (
            self.training_args["prim_slab_coord_loss_weight"]
            * flow_loss_dict["prim_slab_coord_loss"].mean()
            + self.training_args["ads_coord_loss_weight"]
            * flow_loss_dict["ads_coord_loss"].mean()
            + self.training_args["prim_virtual_loss_weight"]
            * flow_loss_dict["prim_virtual_loss"].mean()
            + self.training_args["supercell_virtual_loss_weight"]
            * flow_loss_dict["supercell_virtual_loss"].mean()
            + self.training_args["scaling_factor_loss_weight"]
            * flow_loss_dict["scaling_factor_loss"].mean()
        )

        # Loggings
        batch_size = batch["ref_prim_slab_element"].shape[0]
        sync_dist = prefix == "val"

        # Loss per component
        for loss_name, batch_loss in flow_loss_dict.items():
            self.log(
                f"{prefix}/loss/{loss_name}",
                batch_loss.mean(),
                batch_size=batch_size,
                sync_dist=sync_dist,
            )
        
        # Check metrics for supercell matrix (not trained)
        for metric_name, batch_metric in check_dict.items():
            self.log(
                f"{prefix}/check/{metric_name}",
                batch_metric.mean(),
                batch_size=batch_size,
                sync_dist=sync_dist,
            )

        # Total loss
        self.log(
            f"{prefix}/loss/total",
            total_loss,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )

        return total_loss, flow_loss_dict, out

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        step_start_time = time.time()

        total_loss, flow_loss_dict, out = self._compute_and_log_loss(
            batch, prefix="train"
        )

        # NaN detection for debugging
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            nan_losses = []
            for loss_name, batch_loss in flow_loss_dict.items():
                if torch.isnan(batch_loss).any() or torch.isinf(batch_loss).any():
                    nan_losses.append(loss_name)
            print(f"[NaN DETECTED] Epoch {self.current_epoch}, batch {batch_idx}")
            print(f"  NaN/Inf in losses: {nan_losses}")
            print(f"  total_loss: {total_loss.item()}")
            for k, v in flow_loss_dict.items():
                print(f"  {k}: mean={v.mean().item()}, max={v.max().item()}, min={v.min().item()}")
            # Check model outputs for NaN
            for k, v in out.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        print(f"  NaN/Inf in output '{k}': count={torch.isnan(v).sum().item()}")

        # Training-specific loggings
        batch_size = batch["ref_prim_slab_element"].shape[0]
        self.log("train/throughput/batch_size", batch_size)

        # Stratified loss logging
        for loss_name, batch_loss in flow_loss_dict.items():
            # Stratified loss
            stratified_losses = stratify_loss_by_time(
                batch_t=out["times"], batch_loss=batch_loss, loss_name=loss_name
            )
            for k, v in stratified_losses.items():
                self.log(f"train/loss_by_time/{k}", v, batch_size=batch_size)

        # Time
        step_time = time.time() - step_start_time
        self.log("train/throughput/samples_per_second", batch_size / step_time)

        # Cache training samples for train match rate comparison
        if len(self._train_sample_cache) < self._max_train_cache_size:
            # Cache ground truth structures for comparison during validation
            for i in range(min(batch_size, self._max_train_cache_size - len(self._train_sample_cache))):
                # Store on CPU to save GPU memory
                self._train_sample_cache.append({
                    "prim_slab_coords": batch["prim_slab_cart_coords"][i].detach().cpu().numpy(),
                    "ads_coords": batch["ads_cart_coords"][i].detach().cpu().numpy(),
                    "lattice": batch["lattice"][i].detach().cpu().numpy(),
                    "supercell_matrix": batch["supercell_matrix"][i].detach().cpu().numpy(),
                    "scaling_factor": float(batch["scaling_factor"][i].detach().cpu()),
                    "prim_slab_atom_types": batch["ref_prim_slab_element"][i].detach().cpu().numpy(),
                    "ads_atom_types": batch["ref_ads_element"][i].detach().cpu().numpy(),
                    "prim_slab_atom_mask": batch["prim_slab_atom_pad_mask"][i].detach().cpu().numpy().astype(bool),
                    "ads_atom_mask": batch["ads_atom_pad_mask"][i].detach().cpu().numpy().astype(bool),
                })

        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        # Compute and log validation loss
        self._compute_and_log_loss(batch, prefix="val")

        # Sampling and structure matching
        sample_every_n_epochs = self.validation_args.get("sample_every_n_epochs", 1)
        if (
            sample_every_n_epochs <= 0
            or (self.current_epoch + 1) % sample_every_n_epochs != 0
        ):
            return

        n_samples = self.validation_args["flow_samples"]

        # Use context manager for cleaner OOM handling
        out = None
        with handle_cuda_oom("validation sampling"):
            # Set fixed seed for deterministic validation sampling (for overfitting tests)
            val_sampling_seed = self.validation_args.get("sampling_seed", None)
            if val_sampling_seed is not None:
                torch.manual_seed(val_sampling_seed)
            out = self(
                batch,
                num_sampling_steps=self.validation_args["sampling_steps"],
                center_during_sampling=False,
                multiplicity_flow_sample=n_samples,
            )

        if out is None:
            # OOM occurred, skip this batch
            return

        # Convert virtual coords to lattice params and supercell matrix for validation
        sampled_prim_virtual_coords = out["sampled_prim_virtual_coords"]
        sampled_supercell_virtual_coords = out["sampled_supercell_virtual_coords"]
        sampled_lattices, sampled_supercell_matrices = virtual_coords_to_lattice_and_supercell(
            sampled_prim_virtual_coords, sampled_supercell_virtual_coords
        )

        # Aggregate outputs
        return_dict = {
            "sampled_prim_slab_coords": out["sampled_prim_slab_coords"],
            "sampled_ads_coords": out["sampled_ads_coords"],
            "sampled_lattices": sampled_lattices,
            "sampled_supercell_matrices": sampled_supercell_matrices,
            "sampled_scaling_factors": out["sampled_scaling_factor"],
            "true_prim_slab_coords": batch["prim_slab_cart_coords"],
            "true_ads_coords": batch["ads_cart_coords"],
            "true_lattices": batch["lattice"],
            "true_supercells": batch["supercell_matrix"],
            "true_scaling_factors": batch["scaling_factor"],
            "prim_slab_atom_mask": batch["prim_slab_atom_pad_mask"],
            "ads_atom_mask": batch["ads_atom_pad_mask"],
        }
        
        return_dict["prim_slab_atom_types"] = batch["ref_prim_slab_element"]
        return_dict["ads_atom_types"] = batch["ref_ads_element"]
        self.validation_step_outputs.append(return_dict)

    def _log_generated_structures_to_wandb(
        self,
        valid_outputs: list,
        n_samples: int,
        max_structures: int = 5,
    ) -> None:
        """
        Log generated catalyst structures to W&B as 3D molecular visualizations.

        Args:
            valid_outputs: List of validation step outputs containing generated structures
            n_samples: Number of samples per input (flow_samples)
            max_structures: Maximum number of structures to log per epoch
        """
        if not self.logger or not hasattr(self.logger, 'experiment'):
            rank_zero_info("| W&B logger not available, skipping structure logging.")
            return

        structures_logged = 0
        molecules_to_log = []
        temp_files = []  # Track temp files for cleanup after logging

        for batch_idx, batch_output in enumerate(valid_outputs):
            if structures_logged >= max_structures:
                break

            # Get data from batch output
            sampled_prim_slab_coords = (
                rearrange(
                    batch_output["sampled_prim_slab_coords"],
                    "(b m) n c -> b m n c",
                    m=n_samples,
                )
                .cpu()
                .numpy()
            )
            sampled_ads_coords = (
                rearrange(
                    batch_output["sampled_ads_coords"],
                    "(b m) n c -> b m n c",
                    m=n_samples,
                )
                .cpu()
                .numpy()
            )
            sampled_lattices = (
                rearrange(
                    batch_output["sampled_lattices"],
                    "(b m) c -> b m c",
                    m=n_samples,
                )
                .cpu()
                .numpy()
            )
            sampled_supercell_matrices = (
                rearrange(
                    batch_output["sampled_supercell_matrices"],
                    "(b m) ... -> b m ...",
                    m=n_samples,
                )
                .cpu()
                .numpy()
            )
            sampled_scaling_factors = (
                rearrange(
                    batch_output["sampled_scaling_factors"],
                    "(b m) -> b m",
                    m=n_samples,
                )
                .cpu()
                .numpy()
            )
            prim_slab_atom_types = batch_output["prim_slab_atom_types"].cpu().numpy()
            ads_atom_types = batch_output["ads_atom_types"].cpu().numpy()
            prim_slab_atom_mask = batch_output["prim_slab_atom_mask"].cpu().numpy().astype(bool)
            ads_atom_mask = batch_output["ads_atom_mask"].cpu().numpy().astype(bool)

            batch_size = sampled_prim_slab_coords.shape[0]

            for i in range(batch_size):
                if structures_logged >= max_structures:
                    break

                # Only log the first sample (sample_idx=0) for each batch item
                sample_idx = 0

                try:
                    # Assemble the structure
                    atoms, _ = assemble(
                        generated_prim_slab_coords=sampled_prim_slab_coords[i, sample_idx],
                        generated_ads_coords=sampled_ads_coords[i, sample_idx],
                        generated_lattice=sampled_lattices[i, sample_idx],
                        generated_supercell_matrix=sampled_supercell_matrices[i, sample_idx],
                        generated_scaling_factor=sampled_scaling_factors[i, sample_idx],
                        prim_slab_atom_types=prim_slab_atom_types[i],
                        ads_atom_types=ads_atom_types[i],
                        prim_slab_atom_mask=prim_slab_atom_mask[i],
                        ads_atom_mask=ads_atom_mask[i],
                    )

                    # Write to temporary PDB file and log (W&B supports PDB, not XYZ)
                    with tempfile.NamedTemporaryFile(
                        mode='w', suffix='.pdb', delete=False
                    ) as f:
                        temp_path = f.name

                    ase_write(temp_path, atoms, format='proteindatabank')
                    temp_files.append(temp_path)  # Track for cleanup

                    molecules_to_log.append(
                        wandb.Molecule(
                            temp_path,
                            caption=f"Generated catalyst (epoch {self.current_epoch}, sample {structures_logged})"
                        )
                    )
                    structures_logged += 1

                except Exception as e:
                    rank_zero_info(
                        f"| WARNING: Failed to log structure {structures_logged}: {e}"
                    )
                    continue

        # Log all molecules to W&B
        if molecules_to_log:
            try:
                self.logger.experiment.log({
                    "val/structures/generated": molecules_to_log,
                    "epoch": self.current_epoch,
                })
                rank_zero_info(f"| Logged {len(molecules_to_log)} generated structures to W&B.")
            except Exception as e:
                rank_zero_info(f"| WARNING: Failed to log structures to W&B: {e}")

        # Clean up temp files after logging
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _log_high_rmsd_structures_to_wandb(
        self,
        valid_outputs: list,
        prim_rmsd_results_per_item: list,
        n_samples: int,
        max_structures: int = 3,
    ) -> None:
        """
        Log generated structures with highest RMSD to W&B for debugging.

        Helps identify worst-performing samples to understand model failures.

        Args:
            valid_outputs: List of validation step outputs containing generated structures
            prim_rmsd_results_per_item: List of RMSD results per item (each is list of n_samples RMSDs)
            n_samples: Number of samples per input (flow_samples)
            max_structures: Maximum number of high-RMSD structures to log
        """
        if not self.logger or not hasattr(self.logger, 'experiment'):
            rank_zero_info("| W&B logger not available, skipping high-RMSD structure logging.")
            return

        # Build flat list of (rmsd, batch_output_idx, item_idx, sample_idx)
        # Include structures even when RMSD is None (no match) - these are often the worst
        rmsd_with_indices = []
        item_counter = 0

        for batch_output_idx, batch_output in enumerate(valid_outputs):
            batch_size = batch_output["sampled_prim_slab_coords"].shape[0] // n_samples

            # Get supercell matrices to check for singular matrices
            supercell_matrices = (
                rearrange(
                    batch_output["sampled_supercell_matrices"],
                    "(b m) i j -> b m i j",
                    m=n_samples,
                )
                .cpu()
                .numpy()
            )

            for item_idx in range(batch_size):
                if item_counter < len(prim_rmsd_results_per_item):
                    rmsd_list = prim_rmsd_results_per_item[item_counter]
                    for sample_idx, rmsd in enumerate(rmsd_list):
                        # Skip singular supercell matrices (det ≈ 0) - assembly will fail
                        det = np.linalg.det(supercell_matrices[item_idx, sample_idx])
                        if abs(det) < 0.1:
                            continue
                        # Include all structures (matched or not)
                        rmsd_with_indices.append((rmsd, batch_output_idx, item_idx, sample_idx))
                item_counter += 1

        if not rmsd_with_indices:
            rank_zero_info("| No structures to log (all have singular supercell matrices).")
            return

        # Sort by RMSD descending (highest first), None values (no match) treated as worst (infinity)
        rmsd_with_indices.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0), reverse=True)
        worst_structures = rmsd_with_indices[:max_structures]

        generated_molecules = []
        ground_truth_molecules = []
        temp_files = []

        for rank, (rmsd, batch_output_idx, item_idx, sample_idx) in enumerate(worst_structures):
            batch_output = valid_outputs[batch_output_idx]

            # Reshape sampled data from (B*M, ...) to (B, M, ...)
            sampled_prim_slab_coords = (
                rearrange(
                    batch_output["sampled_prim_slab_coords"],
                    "(b m) n c -> b m n c",
                    m=n_samples,
                )
                .cpu()
                .numpy()
            )
            sampled_ads_coords = (
                rearrange(
                    batch_output["sampled_ads_coords"],
                    "(b m) n c -> b m n c",
                    m=n_samples,
                )
                .cpu()
                .numpy()
            )
            sampled_lattices = (
                rearrange(
                    batch_output["sampled_lattices"],
                    "(b m) c -> b m c",
                    m=n_samples,
                )
                .cpu()
                .numpy()
            )
            sampled_supercell_matrices = (
                rearrange(
                    batch_output["sampled_supercell_matrices"],
                    "(b m) ... -> b m ...",
                    m=n_samples,
                )
                .cpu()
                .numpy()
            )
            sampled_scaling_factors = (
                rearrange(
                    batch_output["sampled_scaling_factors"],
                    "(b m) -> b m",
                    m=n_samples,
                )
                .cpu()
                .numpy()
            )

            # Ground truth data (no multiplicity dimension)
            true_prim_slab_coords = batch_output["true_prim_slab_coords"].cpu().numpy()
            true_ads_coords = batch_output["true_ads_coords"].cpu().numpy()
            true_lattices = batch_output["true_lattices"].cpu().numpy()
            true_supercells = batch_output["true_supercells"].cpu().numpy()
            true_scaling_factors = batch_output["true_scaling_factors"].cpu().numpy()

            prim_slab_atom_types = batch_output["prim_slab_atom_types"].cpu().numpy()
            ads_atom_types = batch_output["ads_atom_types"].cpu().numpy()
            prim_slab_atom_mask = batch_output["prim_slab_atom_mask"].cpu().numpy().astype(bool)
            ads_atom_mask = batch_output["ads_atom_mask"].cpu().numpy().astype(bool)

            try:
                # Assemble generated structure
                gen_atoms, _ = assemble(
                    generated_prim_slab_coords=sampled_prim_slab_coords[item_idx, sample_idx],
                    generated_ads_coords=sampled_ads_coords[item_idx, sample_idx],
                    generated_lattice=sampled_lattices[item_idx, sample_idx],
                    generated_supercell_matrix=sampled_supercell_matrices[item_idx, sample_idx],
                    generated_scaling_factor=sampled_scaling_factors[item_idx, sample_idx],
                    prim_slab_atom_types=prim_slab_atom_types[item_idx],
                    ads_atom_types=ads_atom_types[item_idx],
                    prim_slab_atom_mask=prim_slab_atom_mask[item_idx],
                    ads_atom_mask=ads_atom_mask[item_idx],
                )

                # Write generated structure to temp PDB
                with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
                    gen_temp_path = f.name
                ase_write(gen_temp_path, gen_atoms, format='proteindatabank')
                temp_files.append(gen_temp_path)

                rmsd_str = f"RMSD={rmsd:.3f}A" if rmsd is not None else "No match"
                generated_molecules.append(
                    wandb.Molecule(
                        gen_temp_path,
                        caption=f"Worst #{rank+1}: {rmsd_str} (epoch {self.current_epoch})"
                    )
                )

                # Assemble ground truth structure
                true_atoms, _ = assemble(
                    generated_prim_slab_coords=true_prim_slab_coords[item_idx],
                    generated_ads_coords=true_ads_coords[item_idx],
                    generated_lattice=true_lattices[item_idx],
                    generated_supercell_matrix=true_supercells[item_idx],
                    generated_scaling_factor=true_scaling_factors[item_idx],
                    prim_slab_atom_types=prim_slab_atom_types[item_idx],
                    ads_atom_types=ads_atom_types[item_idx],
                    prim_slab_atom_mask=prim_slab_atom_mask[item_idx],
                    ads_atom_mask=ads_atom_mask[item_idx],
                )

                # Write ground truth to temp PDB
                with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
                    true_temp_path = f.name
                ase_write(true_temp_path, true_atoms, format='proteindatabank')
                temp_files.append(true_temp_path)

                ground_truth_molecules.append(
                    wandb.Molecule(
                        true_temp_path,
                        caption=f"Ground Truth #{rank+1} (epoch {self.current_epoch})"
                    )
                )

            except Exception as e:
                rank_zero_info(f"| WARNING: Failed to log high-RMSD structure #{rank+1}: {e}")
                continue

        # Log to W&B
        if generated_molecules:
            try:
                self.logger.experiment.log({
                    "val/structures/high_rmsd_generated": generated_molecules,
                    "val/structures/high_rmsd_ground_truth": ground_truth_molecules,
                    "epoch": self.current_epoch,
                })
                rank_zero_info(f"| Logged {len(generated_molecules)} high-RMSD structures to W&B.")
            except Exception as e:
                rank_zero_info(f"| WARNING: Failed to log high-RMSD structures to W&B: {e}")

        # Clean up temp files
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def on_validation_epoch_end(self) -> None:
        # NOTE: This hook is called on ALL ranks in DDP, but we only execute code on rank 0.
        # PyTorch Lightning will synchronize all ranks after this hook completes.
        # If rank 0 takes too long (e.g., adsorption energy computation), other ranks will
        # wait and may hit NCCL timeout (default: 30 minutes).
        # Solution: Disable adsorption computation in DDP by setting compute_adsorption=false.
        if self.global_rank == 0:
            # Filter out None
            valid_outputs = [
                out for out in self.validation_step_outputs if out is not None
            ]
            if not valid_outputs:
                return

            n_samples = self.validation_args["flow_samples"]
            prim_rmsd_tasks = []
            slab_rmsd_tasks = []
            prim_validity_tasks = []
            slab_validity_tasks = []
            adsorption_tasks = []

            # Collate results from all batches
            for batch_output in valid_outputs:
                # Reshape and move to cpu (prim_slab coords)
                sampled_prim_slab_coords = (
                    rearrange(
                        batch_output["sampled_prim_slab_coords"],
                        "(b m) n c -> b m n c",
                        m=n_samples,
                    )
                    .cpu()
                    .numpy()
                )
                sampled_ads_coords = (
                    rearrange(
                        batch_output["sampled_ads_coords"],
                        "(b m) n c -> b m n c",
                        m=n_samples,
                    )
                    .cpu()
                    .numpy()
                )
                sampled_lattices = (
                    rearrange(
                        batch_output["sampled_lattices"],
                        "(b m) c -> b m c",
                        m=n_samples,
                    )
                    .cpu()
                    .numpy()
                )
                sampled_supercell_matrices = (
                    rearrange(
                        batch_output["sampled_supercell_matrices"],
                        "(b m) ... -> b m ...",
                        m=n_samples,
                    )
                    .cpu()
                    .numpy()
                )
                sampled_scaling_factors = (
                    rearrange(
                        batch_output["sampled_scaling_factors"],
                        "(b m) -> b m",
                        m=n_samples,
                    )
                    .cpu()
                    .numpy()
                )

                true_prim_slab_coords = batch_output["true_prim_slab_coords"].cpu().numpy()
                true_ads_coords = batch_output["true_ads_coords"].cpu().numpy()
                true_lattices = batch_output["true_lattices"].cpu().numpy()
                true_supercells = batch_output["true_supercells"].cpu().numpy()
                true_scaling_factors = batch_output["true_scaling_factors"].cpu().numpy()
                prim_slab_atom_types = batch_output["prim_slab_atom_types"].cpu().numpy()
                ads_atom_types = batch_output["ads_atom_types"].cpu().numpy()
                prim_slab_atom_mask = batch_output["prim_slab_atom_mask"].cpu().numpy().astype(bool)
                ads_atom_mask = batch_output["ads_atom_mask"].cpu().numpy().astype(bool)

                # Prepare tasks for each item in the batch
                batch_size = sampled_prim_slab_coords.shape[0]
                for i in range(batch_size):
                    # RMSD task (primitive slab only)
                    prim_rmsd_tasks.append(
                        (
                            sampled_prim_slab_coords[i],
                            sampled_lattices[i],
                            true_prim_slab_coords[i],
                            true_lattices[i],
                            prim_slab_atom_types[i],
                            prim_slab_atom_mask[i],
                            self.matcher_kwargs,
                        )
                    )

                    # Slab RMSD task (full system after assemble)
                    slab_rmsd_tasks.append(
                        (
                            sampled_prim_slab_coords[i],
                            sampled_ads_coords[i],
                            sampled_lattices[i],
                            sampled_supercell_matrices[i],
                            sampled_scaling_factors[i],
                            true_prim_slab_coords[i],
                            true_ads_coords[i],
                            true_lattices[i],
                            true_supercells[i],
                            float(true_scaling_factors[i]),
                            prim_slab_atom_types[i],
                            ads_atom_types[i],
                            prim_slab_atom_mask[i],
                            ads_atom_mask[i],
                            self.matcher_kwargs,
                        )
                    )

                    # Prim structural validity task (primitive slab only)
                    prim_validity_tasks.append(
                        (
                            sampled_prim_slab_coords[i],
                            sampled_lattices[i],
                            prim_slab_atom_types[i],
                            prim_slab_atom_mask[i],
                            self.matcher_kwargs,
                        )
                    )
                    
                    # Slab structural validity task (full system after assemble)
                    slab_validity_tasks.append(
                        (
                            sampled_prim_slab_coords[i],
                            sampled_ads_coords[i],
                            sampled_lattices[i],
                            sampled_supercell_matrices[i],
                            sampled_scaling_factors[i],
                            prim_slab_atom_types[i],
                            ads_atom_types[i],
                            prim_slab_atom_mask[i],
                            ads_atom_mask[i],
                        )
                    )
                    
                    # Adsorption energy task (using assemble logic)
                    adsorption_tasks.append(
                        (
                            sampled_prim_slab_coords[i],
                            sampled_ads_coords[i],
                            sampled_lattices[i],
                            sampled_supercell_matrices[i],
                            sampled_scaling_factors[i],
                            prim_slab_atom_types[i],
                            ads_atom_types[i],
                            prim_slab_atom_mask[i],
                            ads_atom_mask[i],
                            self.validation_args.get("adsorption_device", "cuda"),
                            self.validation_args.get("adsorption_model", "uma-m-1p1"),
                        )
                    )

            # Settings for parallel processing
            timeout_seconds = self.validation_args["timeout"]
            num_workers = self.validation_args["num_workers"]

            # Execute prim RMSD tasks in parallel
            prim_rmsd_results_per_item = run_parallel_tasks(
                tasks=prim_rmsd_tasks,
                task_fn=find_best_match_rmsd_prim,
                num_workers=num_workers,
                timeout_seconds=timeout_seconds,
                task_name="Prim RMSD matching",
                default_result=[None] * n_samples,
            )

            # Compute prim match rate and RMSD (primitive slab)
            for result_list in prim_rmsd_results_per_item:
                valid_rmsds = [r for r in result_list if r is not None]
                if valid_rmsds:
                    self.val_prim_match_rate.update(1.0)
                    self.val_prim_rmsd.update(min(valid_rmsds))
                else:
                    self.val_prim_match_rate.update(0.0)

            # Execute slab RMSD tasks in parallel
            slab_rmsd_results_per_item = run_parallel_tasks(
                tasks=slab_rmsd_tasks,
                task_fn=find_best_match_rmsd_slab,
                num_workers=num_workers,
                timeout_seconds=timeout_seconds,
                task_name="Slab RMSD matching",
                default_result=[None] * n_samples,
            )

            # Compute slab match rate and RMSD
            for result_list in slab_rmsd_results_per_item:
                valid_rmsds = [r for r in result_list if r is not None]
                if valid_rmsds:
                    self.val_slab_match_rate.update(1.0)
                    self.val_slab_rmsd.update(min(valid_rmsds))
                else:
                    self.val_slab_match_rate.update(0.0)

                # Compute training match rate (compare generated samples against training set)
                # This helps detect if the model is memorizing training data
                compute_train_match = self.validation_args.get("compute_train_match_rate", True)
                if compute_train_match and len(self._train_sample_cache) > 0:
                    rank_zero_info(f"| Computing train match rate against {len(self._train_sample_cache)} cached training samples...")

                    # Prepare tasks for training match rate (prim only for efficiency)
                    train_match_prim_tasks = []
                    for batch_output in valid_outputs:
                        sampled_prim_slab_coords = (
                            rearrange(
                                batch_output["sampled_prim_slab_coords"],
                                "(b m) n c -> b m n c",
                                m=n_samples,
                            )
                            .cpu()
                            .numpy()
                        )
                        sampled_lattices = (
                            rearrange(
                                batch_output["sampled_lattices"],
                                "(b m) c -> b m c",
                                m=n_samples,
                            )
                            .cpu()
                            .numpy()
                        )
                        prim_slab_atom_types = batch_output["prim_slab_atom_types"].cpu().numpy()
                        prim_slab_atom_mask = batch_output["prim_slab_atom_mask"].cpu().numpy().astype(bool)

                        batch_size = sampled_prim_slab_coords.shape[0]
                        for i in range(batch_size):
                            # For each sample within the generated batch
                            for sample_idx in range(n_samples):
                                train_match_prim_tasks.append(
                                    (
                                        sampled_prim_slab_coords[i, sample_idx],
                                        sampled_lattices[i, sample_idx],
                                        prim_slab_atom_types[i],
                                        prim_slab_atom_mask[i],
                                        self._train_sample_cache,
                                        self.matcher_kwargs,
                                    )
                                )

                    # Execute training match tasks in parallel
                    train_match_prim_results = run_parallel_tasks(
                        tasks=train_match_prim_tasks,
                        task_fn=find_train_match_prim,
                        num_workers=num_workers,
                        timeout_seconds=timeout_seconds * 2,  # Allow more time for many comparisons
                        task_name="Train match rate (prim)",
                        default_result=False,
                    )

                    # Compute train match rate per input (at least one sample matches training)
                    # Results are flattened: [batch1_sample1, batch1_sample2, ..., batch2_sample1, ...]
                    total_items = len(train_match_prim_results) // n_samples
                    for item_idx in range(total_items):
                        item_results = train_match_prim_results[item_idx * n_samples : (item_idx + 1) * n_samples]
                        if any(item_results):
                            self.val_train_prim_match_rate.update(1.0)
                        else:
                            self.val_train_prim_match_rate.update(0.0)

                    rank_zero_info(f"| Train prim match rate: {self.val_train_prim_match_rate.compute():.4f}")

                # Log high-RMSD structures for debugging (top 3 worst by prim RMSD)
                self._log_high_rmsd_structures_to_wandb(
                    valid_outputs=valid_outputs,
                    prim_rmsd_results_per_item=prim_rmsd_results_per_item,
                    n_samples=n_samples,
                    max_structures=3,
                )

            # Execute prim structural validity tasks in parallel (always computed)
            prim_validity_results_per_item = run_parallel_tasks(
                tasks=prim_validity_tasks,
                task_fn=compute_prim_structural_validity_single,
                num_workers=num_workers,
                timeout_seconds=timeout_seconds,
                task_name="Prim structural validity",
                default_result=[False] * n_samples,
            )

            # Compute prim structural validity rate (always computed)
            # Count as valid if at least one sample is valid (consistent with prim/slab RMSD)
            for result_list in prim_validity_results_per_item:
                if any(result_list):
                    self.val_prim_structural_validity.update(1.0)
                else:
                    self.val_prim_structural_validity.update(0.0)

            # Execute comprehensive validity tasks in parallel (structural + SMACT + crystal)
            comprehensive_validity_results_per_item = run_parallel_tasks(
                tasks=slab_validity_tasks,
                task_fn=compute_comprehensive_validity_single,
                num_workers=num_workers,
                timeout_seconds=timeout_seconds,
                task_name="Comprehensive validity",
                default_result=[
                    {"basic_valid": False, "structural_valid": False, "smact_valid": False, "crystal_valid": False}
                    for _ in range(n_samples)
                ],
            )

            # Process comprehensive validity results
            # Count as valid if at least one sample is valid (consistent with prim/slab RMSD)
            for result_list in comprehensive_validity_results_per_item:
                # Structural validity (basic + width/height)
                structural_valid_list = [r.get("basic_valid", False) and r.get("structural_valid", False) for r in result_list]
                if any(structural_valid_list):
                    self.val_structural_validity.update(1.0)
                else:
                    self.val_structural_validity.update(0.0)

                # SMACT validity
                smact_valid_list = [r.get("smact_valid", False) for r in result_list]
                if any(smact_valid_list):
                    self.val_smact_validity.update(1.0)
                else:
                    self.val_smact_validity.update(0.0)

                # Crystal validity
                crystal_valid_list = [r.get("crystal_valid", False) for r in result_list]
                if any(crystal_valid_list):
                    self.val_crystal_validity.update(1.0)
                else:
                    self.val_crystal_validity.update(0.0)

            # Execute adsorption energy tasks in main process (reuse calculator)
            # NOTE: Adsorption computation is disabled by default in DDP due to long runtime
            compute_adsorption = self.validation_args.get("compute_adsorption", False)
            adsorption_results_per_item = []
            
            if not compute_adsorption:
                # Adsorption computation disabled - skip entirely
                rank_zero_info("| Adsorption energy computation disabled (compute_adsorption=false).")
            elif self._uma_calculator is None:
                try:
                    adsorption_device = self.validation_args.get("adsorption_device", "cpu")
                    adsorption_model = self.validation_args.get("adsorption_model", "uma-m-1p1")
                    rank_zero_info(f"| Initializing UMA calculator ({adsorption_model}) on {adsorption_device}...")
                    self._uma_calculator = get_uma_calculator(
                        model_name=adsorption_model, 
                        device=adsorption_device
                    )
                    rank_zero_info("| UMA calculator initialized successfully.")
                except Exception as e:
                    rank_zero_info(f"| WARNING: Failed to initialize UMA calculator: {e}")
                    self._uma_calculator = None
            
            # Compute adsorption energy for each task (in main process)
            if self._uma_calculator is not None:
                rank_zero_info(f"| Computing adsorption energy for {len(adsorption_tasks)} structures...")
                
                for task_idx, task in enumerate(tqdm(
                    adsorption_tasks, 
                    desc="Adsorption energy", 
                    unit="struct",
                    leave=False,
                )):
                    (
                        sampled_prim_slab_coords_i,
                        sampled_ads_coords_i,
                        sampled_lattices_i,
                        sampled_supercell_matrices_i,
                        sampled_scaling_factors_i,
                        prim_slab_atom_types_i,
                        ads_atom_types_i,
                        prim_slab_atom_mask_i,
                        ads_atom_mask_i,
                        _device,  # ignored, use self._uma_calculator
                        _model,   # ignored, use self._uma_calculator
                    ) = task
                    
                    try:
                        result = compute_adsorption_and_validity_single(
                            sampled_prim_slab_coords=sampled_prim_slab_coords_i,
                            sampled_ads_coords=sampled_ads_coords_i,
                            sampled_lattices=sampled_lattices_i,
                            sampled_supercell_matrices=sampled_supercell_matrices_i,
                            sampled_scaling_factors=sampled_scaling_factors_i,
                            prim_slab_atom_types=prim_slab_atom_types_i,
                            ads_atom_types=ads_atom_types_i,
                            prim_slab_atom_mask=prim_slab_atom_mask_i,
                            ads_atom_mask=ads_atom_mask_i,
                            calc=self._uma_calculator,
                        )
                        adsorption_results_per_item.append(result)
                    except Exception as e:
                        rank_zero_info(
                            f"| WARNING: Adsorption task {task_idx+1}/{len(adsorption_tasks)} failed: {e}"
                        )
                        adsorption_results_per_item.append(
                            [{"E_adsorption": float('nan'), "struct_valid": False}] * n_samples
                        )
                
                rank_zero_info(f"| Adsorption energy computation completed.")
            else:
                # Calculator not available, skip all adsorption tasks
                rank_zero_info("| WARNING: UMA calculator not available. Skipping adsorption energy computation.")
                for _ in adsorption_tasks:
                    adsorption_results_per_item.append(
                        [{"E_adsorption": float('nan'), "struct_valid": False}] * n_samples
                    )

            # Compute adsorption energy (only if enabled)
            if compute_adsorption and adsorption_results_per_item:
                for result_list in adsorption_results_per_item:
                    for res in result_list:
                        # Adsorption energy (only for valid energies)
                        e_ads = res["E_adsorption"]
                        if not (e_ads != e_ads):  # Check for NaN
                            self.val_adsorption_energy.update(e_ads)

            # Log aggregated metrics
            # Primitive slab RMSD
            self.log(
                "val/match_rate/primitive_slab", self.val_prim_match_rate.compute(), rank_zero_only=True
            )
            if self.val_prim_rmsd.update_count > 0:
                self.log("val/rmsd/primitive_slab", self.val_prim_rmsd.compute(), rank_zero_only=True)
            else:
                self.log("val/rmsd/primitive_slab", float("nan"), rank_zero_only=True)

            # Slab RMSD (whole slab after assemble)
            self.log(
                "val/match_rate/slab", self.val_slab_match_rate.compute(), rank_zero_only=True
            )
            if self.val_slab_rmsd.update_count > 0:
                self.log("val/rmsd/slab", self.val_slab_rmsd.compute(), rank_zero_only=True)
            else:
                self.log("val/rmsd/slab", float("nan"), rank_zero_only=True)

            # Training match rate (compare generated samples against training set)
            if self.val_train_prim_match_rate.update_count > 0:
                self.log(
                    "val/match_rate/train_primitive",
                    self.val_train_prim_match_rate.compute(),
                    rank_zero_only=True,
                )
            else:
                self.log("val/match_rate/train_primitive", float("nan"), rank_zero_only=True)

            # Log prim structural validity (always computed, regardless of compute_adsorption)
            self.log(
                "val/validity/structural_primitive",
                self.val_prim_structural_validity.compute() if self.val_prim_structural_validity.update_count > 0 else float("nan"),
                rank_zero_only=True,
            )

            # Log slab structural validity (always computed, regardless of compute_adsorption)
            self.log(
                "val/validity/structural",
                self.val_structural_validity.compute() if self.val_structural_validity.update_count > 0 else float("nan"),
                rank_zero_only=True,
            )

            # Log SMACT validity
            self.log(
                "val/validity/smact",
                self.val_smact_validity.compute() if self.val_smact_validity.update_count > 0 else float("nan"),
                rank_zero_only=True,
            )

            # Log crystal validity
            self.log(
                "val/validity/crystal",
                self.val_crystal_validity.compute() if self.val_crystal_validity.update_count > 0 else float("nan"),
                rank_zero_only=True,
            )

            # Log adsorption energy (only if computed)
            if compute_adsorption:
                if self.val_adsorption_energy.update_count > 0:
                    self.log(
                        "val/energy/adsorption",
                        self.val_adsorption_energy.compute(),
                        rank_zero_only=True,
                    )
                else:
                    self.log("val/energy/adsorption", float("nan"), rank_zero_only=True)

            # Log generated structures to W&B as 3D visualizations
            log_structures = self.validation_args.get("log_structures_to_wandb", False)
            if log_structures:
                max_structures = self.validation_args.get("max_structures_to_log", 5)
                self._log_generated_structures_to_wandb(
                    valid_outputs=valid_outputs,
                    n_samples=n_samples,
                    max_structures=max_structures,
                )

            # Reset metrics for next epoch
            self.val_prim_match_rate.reset()
            self.val_prim_rmsd.reset()
            self.val_slab_match_rate.reset()
            self.val_slab_rmsd.reset()
            self.val_train_prim_match_rate.reset()
            self.val_train_slab_match_rate.reset()
            self.val_adsorption_energy.reset()
            self.val_structural_validity.reset()
            self.val_prim_structural_validity.reset()
            self.val_smact_validity.reset()
            self.val_crystal_validity.reset()
            self.validation_step_outputs.clear()  # free memory

        # Restore original weights after validation (EMA was swapped in on_validation_epoch_start)
        if self.use_ema and self._ema is not None:
            self._ema.restore(self.structure_module.parameters())

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step - same as validation step."""
        # Test step is handled the same as validation step
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        """Test epoch end - same as validation epoch end."""
        # Test epoch end is handled the same as validation epoch end
        self.on_validation_epoch_end()

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        num_samples = self.predict_args.get("num_samples", 1)

        # Use context manager for cleaner OOM handling
        out = None
        with handle_cuda_oom(f"prediction on batch {batch_idx}"):
            out = self(
                batch,
                num_sampling_steps=self.predict_args["sampling_steps"],
                center_during_sampling=False,
                refine_final=self.predict_args["refine_final"],
                multiplicity_flow_sample=num_samples,
            )

        if out is None:
            # OOM occurred
            return {"exception": True}

        prediction_output = {
            "exception": False,
            "generated_prim_slab_coords": out["sampled_prim_slab_coords"],
            "generated_ads_coords": out["sampled_ads_coords"],
            "generated_prim_virtual_coords": out["sampled_prim_virtual_coords"],
            "generated_supercell_virtual_coords": out["sampled_supercell_virtual_coords"],
            "generated_scaling_factors": out["sampled_scaling_factor"],
            "prim_slab_atom_types": batch["ref_prim_slab_element"],
            "ads_atom_types": batch["ref_ads_element"],
            "prim_slab_atom_mask": batch["prim_slab_atom_pad_mask"],
            "ads_atom_mask": batch["ads_atom_pad_mask"],
            "tags": batch.get("tags", None),  # TODO: May change feature name
        }

        if "prim_slab_cart_coords" in batch:
            prediction_output["true_prim_slab_coords"] = batch["prim_slab_cart_coords"]

        if "ads_cart_coords" in batch:
            prediction_output["true_ads_coords"] = batch["ads_cart_coords"]

        if "lattice" in batch:
            prediction_output["true_lattices"] = batch["lattice"]

        if "supercell_matrix" in batch:
            prediction_output["true_supercells"] = batch["supercell_matrix"]

        if "scaling_factor" in batch:
            prediction_output["true_scaling_factors"] = batch["scaling_factor"]

        return prediction_output

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.

        Supports:
        - AdamW optimizer with configurable betas (default)
        - Muon optimizer for 2D weight matrices + AdamW for others (combined wrapper)
        - Cosine warmup scheduler (linear warmup + cosine annealing)
        - Linear warmup scheduler (linear warmup + linear decay)
        - No scheduler (constant learning rate)
        """
        optimizer_type = self.training_args.get("optimizer", "adamw").lower()

        if optimizer_type == "muon":
            # Returns CombinedMuonAdamW wrapper (single optimizer for Lightning)
            optimizer = self._configure_muon_optimizer()
        else:
            # Default to AdamW
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.training_args["lr"],
                weight_decay=self.training_args.get("weight_decay", 0.0),
                betas=(
                    self.training_args.get("adam_beta1", 0.9),
                    self.training_args.get("adam_beta2", 0.999),
                ),
            )

        # Check if scheduler is configured
        scheduler_config = self.training_args.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "none") if scheduler_config else "none"

        if scheduler_type == "none" or not scheduler_config:
            rank_zero_info("| Using constant learning rate (no scheduler)")
            return {"optimizer": optimizer}

        # Calculate total training steps
        # Note: estimated_stepping_batches accounts for accumulate_grad_batches and devices
        total_steps = self.trainer.estimated_stepping_batches
        rank_zero_info(f"| Total training steps: {total_steps}")

        # Calculate warmup steps
        warmup_epochs = scheduler_config.get("warmup_epochs")
        if warmup_epochs is not None and warmup_epochs > 0:
            # Calculate steps from epochs
            steps_per_epoch = total_steps // self.trainer.max_epochs
            warmup_steps = int(warmup_epochs * steps_per_epoch)
            rank_zero_info(f"| Warmup: {warmup_epochs} epochs = {warmup_steps} steps")
        else:
            warmup_steps = scheduler_config.get("warmup_steps", 1000)
            rank_zero_info(f"| Warmup: {warmup_steps} steps")

        # Ensure warmup doesn't exceed total steps
        warmup_steps = min(warmup_steps, total_steps - 1)

        # Create scheduler
        if scheduler_type == "cosine_warmup":
            min_lr_ratio = scheduler_config.get("min_lr_ratio", 0.01)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
            )
            rank_zero_info(
                f"| Using cosine warmup scheduler: "
                f"warmup={warmup_steps}, total={total_steps}, min_lr_ratio={min_lr_ratio}"
            )
        elif scheduler_type == "linear_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            rank_zero_info(
                f"| Using linear warmup scheduler: "
                f"warmup={warmup_steps}, total={total_steps}"
            )
        else:
            rank_zero_info(f"| Unknown scheduler type: {scheduler_type}, using constant LR")
            return {"optimizer": optimizer}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update LR every step, not every epoch
                "frequency": 1,
            },
        }

    def _configure_muon_optimizer(self):
        """Configure Muon optimizer for 2D weight matrices with Adam for others.

        Muon is designed for 2D weight matrices in transformers.
        Other parameters (biases, LayerNorm, embeddings, 1D params) use Adam.

        Uses SingleDeviceMuonWithAuxAdam for single-GPU training.

        Returns
        -------
        optimizer
            Combined optimizer with Muon for 2D weights and Adam for others.
        """
        from muon import SingleDeviceMuonWithAuxAdam

        # Separate parameters into 2D weight matrices and others
        muon_params = []
        adam_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Check if this is a 2D weight matrix (not bias, not LayerNorm, not embedding)
            is_2d_weight = (
                param.ndim == 2
                and "bias" not in name.lower()
                and "norm" not in name.lower()
                and "embed" not in name.lower()
                and "layernorm" not in name.lower()
            )

            if is_2d_weight:
                muon_params.append(param)
            else:
                adam_params.append(param)

        # Log parameter counts
        muon_param_count = sum(p.numel() for p in muon_params)
        adam_param_count = sum(p.numel() for p in adam_params)
        total_params = muon_param_count + adam_param_count

        rank_zero_info("| Muon optimizer configuration:")
        rank_zero_info(
            f"|   2D weight params (Muon): {muon_param_count:,} "
            f"({100*muon_param_count/total_params:.1f}%)"
        )
        rank_zero_info(
            f"|   Other params (Adam): {adam_param_count:,} "
            f"({100*adam_param_count/total_params:.1f}%)"
        )

        # Get Muon-specific hyperparameters
        muon_lr = self.training_args["lr"]
        muon_momentum = self.training_args.get("muon_momentum", 0.95)

        # Adam hyperparameters (use 10x lower LR by default for non-Muon params)
        adam_lr = self.training_args.get("adamw_lr", muon_lr * 0.1)
        weight_decay = self.training_args.get("weight_decay", 0.0)

        rank_zero_info(f"|   Muon LR: {muon_lr}, momentum: {muon_momentum}")
        rank_zero_info(f"|   Adam LR: {adam_lr}, weight_decay: {weight_decay}")

        # Create param groups for SingleDeviceMuonWithAuxAdam
        # Muon group: 2D weight matrices
        muon_group = {
            "params": muon_params,
            "use_muon": True,
            "lr": muon_lr,
            "momentum": muon_momentum,
            "weight_decay": weight_decay,
        }

        # Adam group: biases, norms, embeddings
        adam_group = {
            "params": adam_params,
            "use_muon": False,
            "lr": adam_lr,
            "betas": (
                self.training_args.get("adam_beta1", 0.9),
                self.training_args.get("adam_beta2", 0.95),  # Muon package uses 0.95 default
            ),
            "eps": 1e-10,
            "weight_decay": weight_decay,
        }

        # Create combined optimizer for single-GPU training
        return SingleDeviceMuonWithAuxAdam([muon_group, adam_group])

    def configure_callbacks(self) -> list[Callback]:
        """Configure model callbacks.

        Returns
        -------
        List[Callback]
            List of callbacks to be used in the model.

        """
        return []
