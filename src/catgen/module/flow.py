from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from src.catgen.data.prior import PriorSampler, CatPriorSampler
from src.catgen.models.layers import (
    InputEmbedder,
    TransformerBackbone,
    OutputProjection,
)
from src.catgen.models.utils import LinearNoBias, default
from src.catgen.scripts.refine_sc_mat import refine_sc_mat
from src.catgen.constants import (
    FLOW_EPSILON,
    EPS_NUMERICAL,
)


# =============================================================================
# Helper Functions
# =============================================================================

def compute_flow_vector(
    predicted: Tensor,
    current: Tensor,
    t: Union[float, Tensor],
    eps: float = FLOW_EPSILON,
) -> Tensor:
    """Compute flow vector field for ODE integration.

    The flow is defined as: flow = (x_1_pred - x_t) / (1 - t)
    where x_1_pred is the model's prediction of the final state.

    Args:
        predicted: Model's prediction of final state x_1
        current: Current state x_t
        t: Current timestep (scalar or tensor)
        eps: Small epsilon for numerical stability

    Returns:
        Flow vector of same shape as predicted/current
    """
    # Ensure t is broadcastable
    if isinstance(t, float):
        divisor = 1.0 - t + eps
    else:
        # Handle tensor t - reshape for proper broadcasting
        divisor = 1.0 - t + eps
        # Add dimensions as needed based on predicted shape
        while divisor.dim() < predicted.dim():
            divisor = divisor.unsqueeze(-1)

    return (predicted - current) / divisor


class FlowModule(nn.Module):
    """
    Flow model that jointly processes primitive slab and adsorbate atoms.

    Architecture:
        1. InputEmbedder: embeds all input features into hidden representations
        2. TransformerBackbone: single DiT transformer for joint attention
        3. OutputProjection: projects to coordinate predictions
    """

    def __init__(
        self,
        hidden_dim,
        transformer_depth,
        transformer_heads,
        positional_encoding=False,
        attention_impl="pytorch",
        activation_checkpointing=False,
        **kwargs,
    ):
        super().__init__()

        # Input embedding
        self.input_embedder = InputEmbedder(
            hidden_dim=hidden_dim,
            positional_encoding=positional_encoding,
        )

        # Transformer backbone
        self.transformer = TransformerBackbone(
            hidden_dim=hidden_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing,
        )

        # Output projection
        self.output_projection = OutputProjection(hidden_dim=hidden_dim)

    def forward(self, prim_slab_r_noisy, ads_r_noisy, prim_virtual_noisy, supercell_virtual_noisy, sf_noisy, times, feats, multiplicity=1):
        """
        Forward pass through the flow model.

        Args:
            prim_slab_r_noisy: (B*mult, N, 3) noisy prim_slab coordinates
            ads_r_noisy: (B*mult, M, 3) noisy adsorbate coordinates
            prim_virtual_noisy: (B*mult, 3, 3) noisy primitive virtual coordinates (lattice vectors)
            supercell_virtual_noisy: (B*mult, 3, 3) noisy supercell virtual coordinates
            sf_noisy: (B*mult,) noisy scaling factor (normalized)
            times: (B*mult,) timesteps
            feats: feature dictionary
            multiplicity: number of samples per input

        Returns:
            dict with:
                - prim_slab_r_update: (B*mult, N, 3)
                - ads_r_update: (B*mult, M, 3)
                - prim_virtual_update: (B*mult, 3, 3)
                - supercell_virtual_update: (B*mult, 3, 3)
                - sf_update: (B*mult,)
        """
        # Embed input features
        h, mask, n_prim_slab, n_ads = self.input_embedder(
            prim_slab_x_t=prim_slab_r_noisy,
            ads_x_t=ads_r_noisy,
            prim_virtual_t=prim_virtual_noisy,
            supercell_virtual_t=supercell_virtual_noisy,
            sf_t=sf_noisy,
            feats=feats,
            multiplicity=multiplicity,
        )

        # Transformer backbone
        h = self.transformer(h, times, mask)

        # Output projection
        prim_slab_mask = feats["prim_slab_atom_pad_mask"].repeat_interleave(multiplicity, 0)
        return self.output_projection(h, n_prim_slab, n_ads, prim_slab_mask)


class AtomFlowMatching(Module):
    """Atom flow matching module for joint prim_slab, adsorbate, lattice, and supercell matrix prediction."""

    # Lattice length conversion (Angstrom <-> nm)
    NM_TO_ANG_SCALE = 10.0
    ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

    def __init__(
        self,
        flow_model_args: dict,
        prior_sampler: CatPriorSampler,
        num_sampling_steps: int = 16,
        synchronize_timesteps: bool = False,
        compile_model: bool = False,
        timestep_distribution: str = "uniform",
        fixed_timestep: float = None,
        **kwargs: dict,
    ):
        super().__init__()

        # Model
        self.flow_model = FlowModule(**flow_model_args)
        if compile_model:
            self.flow_model = torch.compile(
                self.flow_model, dynamic=False, fullgraph=False
            )

        # Prior
        self.prior_sampler = prior_sampler

        # Parameters
        self.num_sampling_steps = num_sampling_steps
        self.synchronize_timesteps = synchronize_timesteps
        self.timestep_distribution = timestep_distribution
        self.use_time_reweighting = kwargs.get("use_time_reweighting", False)
        self.fixed_timestep = fixed_timestep  # For overfitting tests
        self.fixed_prior_seed = kwargs.get("fixed_prior_seed", None)  # For deterministic prior in overfitting tests
        self.fixed_prior_multiplicity = kwargs.get("fixed_prior_multiplicity", None)  # Training multiplicity for matching prior sequence

    @property
    def device(self):
        return next(self.flow_model.parameters()).device

    def _sample_timesteps(self, n: int) -> torch.Tensor:
        """Sample timesteps from the configured distribution.

        Args:
            n: Number of timesteps to sample

        Returns:
            Tensor of shape (n,) with timesteps in [0, 1]
        """
        # Fixed timestep for overfitting tests (deterministic training)
        if self.fixed_timestep is not None:
            return torch.full((n,), self.fixed_timestep, device=self.device)

        if self.timestep_distribution == "exponential":
            # Exponential distribution: more samples near t=0
            # Transform: t = 1 - exp(-3 * u) where u ~ Uniform(0, 1)
            # This gives higher density near t=0 for better denoising
            return 1 - torch.exp(-3 * torch.rand(n, device=self.device))
        else:
            # Default: uniform distribution
            return torch.rand(n, device=self.device)

    def _compute_time_weights(self, times: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
        """Compute time-based loss weights using SNR-inspired weighting.

        Gives higher weight to timesteps near t=0 and t=1 where denoising is critical.
        weight(t) = 1 / (t * (1 - t) + eps)

        Args:
            times: Tensor of shape (B,) with timesteps in [0, 1]
            eps: Small constant for numerical stability

        Returns:
            Tensor of shape (B,) with normalized weights
        """
        weights = 1.0 / (times * (1 - times) + eps)
        # Normalize weights to have mean 1.0 (so total loss magnitude stays similar)
        weights = weights / weights.mean()
        return weights

    def preconditioned_network_forward(
        self,
        noised_prim_slab_coords,
        noised_ads_coords,
        noised_prim_virtual_coords,
        noised_supercell_virtual_coords,
        noised_scaling_factor,
        time,
        network_condition_kwargs: dict,
        training: bool = True,
    ):
        """
        Forward pass with preconditioning (standardization).

        Args:
            noised_prim_slab_coords: (B, N, 3) noised prim_slab coordinates (raw space, Angstrom)
            noised_ads_coords: (B, M, 3) noised adsorbate coordinates (raw space, Angstrom)
            noised_prim_virtual_coords: (B, 3, 3) noised primitive virtual coordinates (raw space)
            noised_supercell_virtual_coords: (B, 3, 3) noised supercell virtual coordinates (raw space)
            noised_scaling_factor: (B,) noised scaling factor (raw space)
            time: timestep(s)
            network_condition_kwargs: additional kwargs for network
            training: whether in training mode

        Returns:
            Dictionary with denoised predictions for all variables
        """
        batch, device = noised_prim_slab_coords.shape[0], noised_prim_slab_coords.device

        if isinstance(time, float):
            time = torch.full((batch,), time, device=device)

        # Standardize prim_slab and ads coordinates using mean/std
        prim_slab_r_noisy = self.prior_sampler.normalize_prim_slab_coords(noised_prim_slab_coords)
        ads_r_noisy = self.prior_sampler.normalize_ads_coords(noised_ads_coords)

        # Normalize virtual coordinates
        prim_virtual_noisy = self.prior_sampler.normalize_prim_virtual(noised_prim_virtual_coords)
        supercell_virtual_noisy = self.prior_sampler.normalize_supercell_virtual(noised_supercell_virtual_coords)

        # Normalize scaling factor
        sf_noisy = self.prior_sampler.normalize_scaling_factor(noised_scaling_factor)

        # Predict (network operates in standardized/normalized space)
        net_out = self.flow_model(
            prim_slab_r_noisy=prim_slab_r_noisy,
            ads_r_noisy=ads_r_noisy,
            prim_virtual_noisy=prim_virtual_noisy,
            supercell_virtual_noisy=supercell_virtual_noisy,
            sf_noisy=sf_noisy,
            times=time,
            **network_condition_kwargs,
        )

        # Destandardize prim_slab and ads coordinates back to Angstrom
        denoised_prim_slab_coords = self.prior_sampler.denormalize_prim_slab_coords(net_out["prim_slab_r_update"])
        denoised_ads_coords = self.prior_sampler.denormalize_ads_coords(net_out["ads_r_update"])

        # Denormalize virtual coordinates back to raw space
        denoised_prim_virtual_coords = self.prior_sampler.denormalize_prim_virtual(net_out["prim_virtual_update"])
        denoised_supercell_virtual_coords = self.prior_sampler.denormalize_supercell_virtual(net_out["supercell_virtual_update"])

        # Denormalize scaling factor back to raw space
        denoised_scaling_factor = self.prior_sampler.denormalize_scaling_factor(net_out["sf_update"])

        # Build result dict
        result = {
            "denoised_prim_slab_coords": denoised_prim_slab_coords,
            "denoised_ads_coords": denoised_ads_coords,
            "denoised_prim_virtual_coords": denoised_prim_virtual_coords,
            "denoised_supercell_virtual_coords": denoised_supercell_virtual_coords,
            "denoised_scaling_factor": denoised_scaling_factor,
            # Also store normalized predictions for normalized loss computation
            "normalized_prim_slab_coords": net_out["prim_slab_r_update"],
            "normalized_ads_coords": net_out["ads_r_update"],
            "normalized_prim_virtual_coords": net_out["prim_virtual_update"],
            "normalized_supercell_virtual_coords": net_out["supercell_virtual_update"],
            "normalized_scaling_factor": net_out["sf_update"],
        }

        return result

    def forward(self, feats, multiplicity=1, **kwargs):
        """Flow matching training step.

        Args:
            feats: Dictionary containing:
                - prim_slab_cart_coords: (B, N, 3) target prim_slab coordinates
                - ads_cart_coords: (B, M, 3) target adsorbate coordinates
                - primitive_virtual_coords: (B, 3, 3) target primitive virtual coordinates
                - supercell_virtual_coords: (B, 3, 3) target supercell virtual coordinates
                - scaling_factor: (B,) target scaling factor
                - prim_slab_atom_pad_mask: (B, N) mask for valid prim_slab atoms
                - ads_atom_pad_mask: (B, M) mask for valid adsorbate atoms
            multiplicity: number of samples per input

        Returns:
            Dictionary with denoised predictions and ground truth
        """
        batch_size = feats["prim_slab_cart_coords"].shape[0]
        num_prim_slab_atoms = feats["prim_slab_cart_coords"].shape[1]
        num_ads_atoms = feats["ads_cart_coords"].shape[1]
        prim_slab_mask = feats["prim_slab_atom_pad_mask"]
        ads_mask = feats["ads_atom_pad_mask"]

        # Sample timesteps using configured distribution
        if self.synchronize_timesteps:
            times = self._sample_timesteps(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            times = self._sample_timesteps(batch_size * multiplicity)

        # Sample prior from prior_sampler (now includes virtual coords)
        sampler_data = {
            "prim_slab_cart_coords": feats["prim_slab_cart_coords"].repeat_interleave(multiplicity, 0),
            "prim_slab_atom_mask": prim_slab_mask.repeat_interleave(multiplicity, 0),
            "ads_cart_coords": feats["ads_cart_coords"].repeat_interleave(multiplicity, 0),
            "ads_atom_mask": ads_mask.repeat_interleave(multiplicity, 0),
        }
        # Seed random generator for deterministic prior sampling in overfitting tests
        if self.fixed_prior_seed is not None:
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            torch.manual_seed(self.fixed_prior_seed)
        priors = self.prior_sampler.sample(sampler_data)
        if self.fixed_prior_seed is not None:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)

        prim_slab_coords_0 = priors["prim_slab_coords_0"]  # (B * multiplicity, N, 3) raw space
        ads_coords_0 = priors["ads_coords_0"]  # (B * multiplicity, M, 3) raw space
        prim_virtual_coords_0 = priors["prim_virtual_coords_0"]  # (B * multiplicity, 3, 3) raw space
        supercell_virtual_coords_0 = priors["supercell_virtual_coords_0"]  # (B * multiplicity, 3, 3) raw space
        scaling_factor_0 = priors["scaling_factor_0"]  # (B * multiplicity,) raw space

        # Prepare target data
        prim_slab_coords = feats["prim_slab_cart_coords"].repeat_interleave(multiplicity, 0)
        ads_coords = feats["ads_cart_coords"].repeat_interleave(multiplicity, 0)
        prim_virtual_coords = feats["primitive_virtual_coords"].repeat_interleave(multiplicity, 0)  # (B, 3, 3)
        supercell_virtual_coords = feats["supercell_virtual_coords"].repeat_interleave(multiplicity, 0)  # (B, 3, 3)
        scaling_factor = feats["scaling_factor"].repeat_interleave(multiplicity, 0)  # (B,)

        # Add noise - interpolate between prior and target (in raw space for all variables)
        noised_prim_slab_coords = (1 - times[:, None, None]) * prim_slab_coords_0 + times[:, None, None] * prim_slab_coords
        noised_ads_coords = (1 - times[:, None, None]) * ads_coords_0 + times[:, None, None] * ads_coords
        noised_prim_virtual_coords = (1 - times[:, None, None]) * prim_virtual_coords_0 + times[:, None, None] * prim_virtual_coords
        noised_supercell_virtual_coords = (1 - times[:, None, None]) * supercell_virtual_coords_0 + times[:, None, None] * supercell_virtual_coords
        noised_scaling_factor = (1 - times) * scaling_factor_0 + times * scaling_factor

        # Get model prediction
        net_result = self.preconditioned_network_forward(
            noised_prim_slab_coords,
            noised_ads_coords,
            noised_prim_virtual_coords,
            noised_supercell_virtual_coords,
            noised_scaling_factor,
            times,
            training=True,
            network_condition_kwargs=dict(
                feats=feats, multiplicity=multiplicity, **kwargs
            ),
        )

        out_dict = dict(
            denoised_prim_slab_coords=net_result["denoised_prim_slab_coords"],
            denoised_ads_coords=net_result["denoised_ads_coords"],
            denoised_prim_virtual_coords=net_result["denoised_prim_virtual_coords"],
            denoised_supercell_virtual_coords=net_result["denoised_supercell_virtual_coords"],
            denoised_scaling_factor=net_result["denoised_scaling_factor"],
            # Store normalized predictions for normalized loss computation
            normalized_prim_slab_coords=net_result.get("normalized_prim_slab_coords"),
            normalized_ads_coords=net_result.get("normalized_ads_coords"),
            normalized_prim_virtual_coords=net_result.get("normalized_prim_virtual_coords"),
            normalized_supercell_virtual_coords=net_result.get("normalized_supercell_virtual_coords"),
            normalized_scaling_factor=net_result.get("normalized_scaling_factor"),
            times=times,
            aligned_true_prim_slab_coords=prim_slab_coords,
            aligned_true_ads_coords=ads_coords,
            aligned_true_prim_virtual_coords=prim_virtual_coords,
            aligned_true_supercell_virtual_coords=supercell_virtual_coords,
            aligned_true_scaling_factor=scaling_factor,
        )

        return out_dict

    def compute_loss(self, feats, out_dict, multiplicity=1, loss_type: str = "l2", loss_space: str = "raw") -> tuple[dict, dict]:
        """Compute losses for prim_slab coords, adsorbate coords, virtual coords, and scaling factor.

        Args:
            feats: Input features dictionary
            out_dict: Output dictionary from forward pass
            multiplicity: number of samples per input
            loss_type: "l1" or "l2" loss
            loss_space: "raw" for loss in Angstrom space, "normalized" for unit-variance space

        Returns:
            Dictionary with per-sample losses:
                - prim_slab_coord_loss: (B * multiplicity,) prim_slab coordinate loss
                - ads_coord_loss: (B * multiplicity,) adsorbate coordinate loss
                - prim_virtual_loss: (B * multiplicity,) primitive virtual coord loss
                - supercell_virtual_loss: (B * multiplicity,) supercell virtual coord loss
                - scaling_factor_loss: (B * multiplicity,) scaling factor loss
        """
        if loss_type == "l2":
            loss_fn = F.mse_loss
        elif loss_type == "l1":
            loss_fn = F.l1_loss
        else:
            raise ValueError(
                f"Unsupported loss_type: {loss_type}. Must be 'l1' or 'l2'."
            )

        # Get predictions and targets based on loss_space
        if loss_space == "normalized":
            # Use normalized predictions (unit variance space)
            pred_prim_slab_coords = out_dict["normalized_prim_slab_coords"]
            pred_ads_coords = out_dict["normalized_ads_coords"]
            pred_prim_virtual = out_dict["normalized_prim_virtual_coords"]
            pred_supercell_virtual = out_dict["normalized_supercell_virtual_coords"]
            pred_scaling_factor = out_dict["normalized_scaling_factor"]
            # Normalize targets
            true_prim_slab_coords = self.prior_sampler.normalize_prim_slab_coords(out_dict["aligned_true_prim_slab_coords"])
            true_ads_coords = self.prior_sampler.normalize_ads_coords(out_dict["aligned_true_ads_coords"])
            true_prim_virtual = self.prior_sampler.normalize_prim_virtual(out_dict["aligned_true_prim_virtual_coords"])
            true_supercell_virtual = self.prior_sampler.normalize_supercell_virtual(out_dict["aligned_true_supercell_virtual_coords"])
            true_scaling_factor = self.prior_sampler.normalize_scaling_factor(out_dict["aligned_true_scaling_factor"])
        else:
            # Use raw space (Angstrom)
            pred_prim_slab_coords = out_dict["denoised_prim_slab_coords"]
            pred_ads_coords = out_dict["denoised_ads_coords"]
            pred_prim_virtual = out_dict["denoised_prim_virtual_coords"]
            pred_supercell_virtual = out_dict["denoised_supercell_virtual_coords"]
            pred_scaling_factor = out_dict["denoised_scaling_factor"]
            true_prim_slab_coords = out_dict["aligned_true_prim_slab_coords"]
            true_ads_coords = out_dict["aligned_true_ads_coords"]
            true_prim_virtual = out_dict["aligned_true_prim_virtual_coords"]
            true_supercell_virtual = out_dict["aligned_true_supercell_virtual_coords"]
            true_scaling_factor = out_dict["aligned_true_scaling_factor"]

        prim_slab_mask = feats["prim_slab_atom_pad_mask"].repeat_interleave(multiplicity, 0)

        prim_slab_coord_loss = loss_fn(pred_prim_slab_coords, true_prim_slab_coords, reduction="none")
        prim_slab_coord_loss = (prim_slab_coord_loss * prim_slab_mask[..., None]).sum(dim=(1, 2)) / (
            prim_slab_mask.sum(dim=1) + 1e-8
        )

        # Loss for adsorbate coordinates
        ads_mask = feats["ads_atom_pad_mask"].repeat_interleave(multiplicity, 0)
        ads_mask_sum = ads_mask.sum(dim=1, keepdim=True)

        ads_coord_loss = loss_fn(pred_ads_coords, true_ads_coords, reduction="none")
        ads_coord_loss = (ads_coord_loss * ads_mask[..., None]).sum(dim=(1, 2)) / (ads_mask_sum.squeeze(-1) + EPS_NUMERICAL)

        # Handle case where all adsorbate atoms are masked (no adsorbate)
        ads_mask_sum_flat = ads_mask_sum.squeeze(-1)
        ads_coord_loss = torch.where(ads_mask_sum_flat > 0, ads_coord_loss, torch.zeros_like(ads_coord_loss))

        # Loss for primitive virtual coordinates
        prim_virtual_loss = loss_fn(
            pred_prim_virtual, true_prim_virtual, reduction="none"
        ).mean(dim=(1, 2))

        # Loss for supercell virtual coordinates
        supercell_virtual_loss = loss_fn(
            pred_supercell_virtual, true_supercell_virtual, reduction="none"
        ).mean(dim=(1, 2))

        # Loss for scaling factor
        scaling_factor_loss = loss_fn(
            pred_scaling_factor, true_scaling_factor, reduction="none"
        )

        # Apply time-based loss reweighting if enabled
        if self.use_time_reweighting and "times" in out_dict:
            time_weights = self._compute_time_weights(out_dict["times"])
            prim_slab_coord_loss = prim_slab_coord_loss * time_weights
            ads_coord_loss = ads_coord_loss * time_weights
            prim_virtual_loss = prim_virtual_loss * time_weights
            supercell_virtual_loss = supercell_virtual_loss * time_weights
            scaling_factor_loss = scaling_factor_loss * time_weights

        loss_dict = {
            "prim_slab_coord_loss": prim_slab_coord_loss,
            "ads_coord_loss": ads_coord_loss,
            "prim_virtual_loss": prim_virtual_loss,
            "supercell_virtual_loss": supercell_virtual_loss,
            "scaling_factor_loss": scaling_factor_loss,
        }

        check_dict = {}

        return loss_dict, check_dict

    @torch.no_grad()
    def sample(
        self,
        prim_slab_atom_mask,
        ads_atom_mask,
        num_sampling_steps: int | None = None,
        multiplicity: int = 1,
        center_coords: bool = True,
        refine_final: bool = False,
        return_trajectory: bool = False,
        # The following arguments are passed directly to the model
        **network_condition_kwargs,
    ):
        """
        Performs sampling from the learned flow ODE for prim_slab coords, adsorbate coords, lattice, supercell matrix, and scaling factor.

        Parameters
        ----------
        prim_slab_atom_mask : torch.Tensor
            A boolean tensor of shape (batch, num_prim_slab_atoms) indicating valid atoms.
        ads_atom_mask : torch.Tensor
            A boolean tensor of shape (batch, num_ads_atoms) indicating valid adsorbate atoms.
        num_sampling_steps : int, optional
            The number of discrete steps for the ODE solver.
        multiplicity : int, optional
            The number of samples to generate per input.
        center_coords : bool, optional
            If True, coordinates will be centered after each sampling step.
        refine_final : bool, optional
            If True, a final refinement step will be performed.
        return_trajectory : bool, optional
            If True, full trajectories will be returned.
        **network_condition_kwargs :
            Additional keyword arguments passed to the flow model.

        Returns
        -------
        dict
            Dictionary containing sampled prim_slab coords, adsorbate coords, lattice, supercell matrix, and scaling factor.
        """
        ### Setup and initialize
        num_steps = default(num_sampling_steps, self.num_sampling_steps)
        # Expand masks by multiplicity
        prim_slab_atom_mask = prim_slab_atom_mask.repeat_interleave(multiplicity, 0)
        ads_atom_mask = ads_atom_mask.repeat_interleave(multiplicity, 0)
        batch_size = prim_slab_atom_mask.shape[0]
        num_prim_slab_atoms = prim_slab_atom_mask.shape[1]
        num_ads_atoms = ads_atom_mask.shape[1]
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device)

        # Update network_condition_kwargs with multiplicity
        network_condition_kwargs = {**network_condition_kwargs, "multiplicity": multiplicity}

        # Dummy data dict for correct sampling shape
        sampler_data = {
            "prim_slab_cart_coords": torch.zeros((batch_size, num_prim_slab_atoms, 3), device=self.device),
            "prim_slab_atom_mask": prim_slab_atom_mask,
            "ads_cart_coords": torch.zeros((batch_size, num_ads_atoms, 3), device=self.device),
            "ads_atom_mask": ads_atom_mask,
        }

        # Initialize from the prior sampler at t=0 (all priors are already in raw space)
        # Use fixed_prior_seed if set (for overfitting tests - must match training prior)
        if self.fixed_prior_seed is not None:
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            torch.manual_seed(self.fixed_prior_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.fixed_prior_seed)

            # When fixed_prior_multiplicity is set, sample with that multiplicity to match
            # training's random sequence, then extract the samples we need
            if self.fixed_prior_multiplicity is not None and self.fixed_prior_multiplicity > batch_size:
                # Expand sampler_data to match training multiplicity
                expanded_sampler_data = {
                    "prim_slab_cart_coords": sampler_data["prim_slab_cart_coords"].repeat(self.fixed_prior_multiplicity, 1, 1)[:self.fixed_prior_multiplicity],
                    "prim_slab_atom_mask": sampler_data["prim_slab_atom_mask"].repeat(self.fixed_prior_multiplicity, 1)[:self.fixed_prior_multiplicity],
                    "ads_cart_coords": sampler_data["ads_cart_coords"].repeat(self.fixed_prior_multiplicity, 1, 1)[:self.fixed_prior_multiplicity],
                    "ads_atom_mask": sampler_data["ads_atom_mask"].repeat(self.fixed_prior_multiplicity, 1)[:self.fixed_prior_multiplicity],
                }
                expanded_priors = self.prior_sampler.sample(expanded_sampler_data)
                # Extract only the samples we need
                priors = {
                    "prim_slab_coords_0": expanded_priors["prim_slab_coords_0"][:batch_size],
                    "ads_coords_0": expanded_priors["ads_coords_0"][:batch_size],
                    "prim_virtual_coords_0": expanded_priors["prim_virtual_coords_0"][:batch_size],
                    "supercell_virtual_coords_0": expanded_priors["supercell_virtual_coords_0"][:batch_size],
                    "scaling_factor_0": expanded_priors["scaling_factor_0"][:batch_size],
                }
            else:
                priors = self.prior_sampler.sample(sampler_data)
        else:
            priors = self.prior_sampler.sample(sampler_data)

        if self.fixed_prior_seed is not None:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)
        prim_slab_coords_t = priors["prim_slab_coords_0"] * prim_slab_atom_mask.unsqueeze(-1)
        ads_coords_t = priors["ads_coords_0"] * ads_atom_mask.unsqueeze(-1)
        prim_virtual_coords_t = priors["prim_virtual_coords_0"]
        supercell_virtual_coords_t = priors["supercell_virtual_coords_0"]
        scaling_factor_t = priors["scaling_factor_0"]

        # Initialize lists to store trajectories if requested
        prim_slab_coord_trajectory = [prim_slab_coords_t] if return_trajectory else None
        ads_coord_trajectory = [ads_coords_t] if return_trajectory else None
        prim_virtual_coord_trajectory = [prim_virtual_coords_t] if return_trajectory else None
        supercell_virtual_coord_trajectory = [supercell_virtual_coords_t] if return_trajectory else None
        scaling_factor_trajectory = [scaling_factor_t] if return_trajectory else None

        ### ODE Solver Loop (Euler's Method)
        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Get the model's prediction for the final state (x_1)
            net_result = self.preconditioned_network_forward(
                noised_prim_slab_coords=prim_slab_coords_t,
                noised_ads_coords=ads_coords_t,
                noised_prim_virtual_coords=prim_virtual_coords_t,
                noised_supercell_virtual_coords=supercell_virtual_coords_t,
                noised_scaling_factor=scaling_factor_t,
                time=t.item(),
                training=False,
                network_condition_kwargs=network_condition_kwargs,
            )
            pred_prim_slab_coords_1 = net_result["denoised_prim_slab_coords"]
            pred_ads_coords_1 = net_result["denoised_ads_coords"]
            pred_prim_virtual_1 = net_result["denoised_prim_virtual_coords"]
            pred_supercell_virtual_1 = net_result["denoised_supercell_virtual_coords"]
            pred_scaling_factor_1 = net_result["denoised_scaling_factor"]

            # Calculate the flow (vector field) using helper for numerical stability
            flow_prim_slab_coords = compute_flow_vector(pred_prim_slab_coords_1, prim_slab_coords_t, t.item())
            flow_ads_coords = compute_flow_vector(pred_ads_coords_1, ads_coords_t, t.item())
            flow_prim_virtual = compute_flow_vector(pred_prim_virtual_1, prim_virtual_coords_t, t.item())
            flow_supercell_virtual = compute_flow_vector(pred_supercell_virtual_1, supercell_virtual_coords_t, t.item())
            flow_scaling_factor = compute_flow_vector(pred_scaling_factor_1, scaling_factor_t, t.item())

            # Perform one step of Euler's method
            dt = t_next - t
            prim_slab_coords_t = prim_slab_coords_t + flow_prim_slab_coords * dt
            ads_coords_t = ads_coords_t + flow_ads_coords * dt
            prim_virtual_coords_t = prim_virtual_coords_t + flow_prim_virtual * dt
            supercell_virtual_coords_t = supercell_virtual_coords_t + flow_supercell_virtual * dt
            scaling_factor_t = scaling_factor_t + flow_scaling_factor * dt

            # Ensure padding atoms remain at zero
            prim_slab_coords_t = prim_slab_coords_t * prim_slab_atom_mask.unsqueeze(-1)
            ads_coords_t = ads_coords_t * ads_atom_mask.unsqueeze(-1)

            # # Center using prim + ads together, then split back
            # combined_coords = torch.cat([prim_slab_coords_t, ads_coords_t], dim=1)
            # combined_mask = torch.cat([prim_slab_atom_mask, ads_atom_mask], dim=1)
            # combined_coords = center_random_augmentation(
            #     combined_coords, combined_mask, augmentation=False, centering=center_coords
            # )
            # prim_slab_coords_t, ads_coords_t = torch.split(
            #     combined_coords, [num_prim_slab_atoms, num_ads_atoms], dim=1
            # )
            # prim_slab_coords_t = prim_slab_coords_t * prim_slab_atom_mask.unsqueeze(-1)
            # ads_coords_t = ads_coords_t * ads_atom_mask.unsqueeze(-1)

            # Store the current state if collecting trajectories
            if return_trajectory:
                prim_slab_coord_trajectory.append(prim_slab_coords_t)
                ads_coord_trajectory.append(ads_coords_t)
                prim_virtual_coord_trajectory.append(prim_virtual_coords_t)
                supercell_virtual_coord_trajectory.append(supercell_virtual_coords_t)
                scaling_factor_trajectory.append(scaling_factor_t)

        ### Final Refinement Step
        if refine_final:
            final_prim_slab_coords, final_ads_coords, final_prim_virtual_coords, final_supercell_virtual_coords, final_scaling_factor = self._refine_step(
                prim_slab_coords_t=prim_slab_coords_t,
                ads_coords_t=ads_coords_t,
                prim_virtual_coords_t=prim_virtual_coords_t,
                supercell_virtual_coords_t=supercell_virtual_coords_t,
                scaling_factor_t=scaling_factor_t,
                prim_slab_atom_mask=prim_slab_atom_mask,
                ads_atom_mask=ads_atom_mask,
                multiplicity=multiplicity,
                network_condition_kwargs=network_condition_kwargs,
            )
        else:
            final_prim_slab_coords = prim_slab_coords_t
            final_ads_coords = ads_coords_t
            final_prim_virtual_coords = prim_virtual_coords_t
            final_supercell_virtual_coords = supercell_virtual_coords_t
            final_scaling_factor = scaling_factor_t

        # # Center prim + ads together at the end, then split and mask
        # final_combined_coords = torch.cat([final_prim_slab_coords, final_ads_coords], dim=1)
        # final_combined_mask = torch.cat([prim_slab_atom_mask, ads_atom_mask], dim=1)
        # final_combined_coords = center_random_augmentation(
        #     final_combined_coords, final_combined_mask, augmentation=False, centering=center_coords
        # )
        # final_prim_slab_coords, final_ads_coords = torch.split(
        #     final_combined_coords, [num_prim_slab_atoms, num_ads_atoms], dim=1
        # )
        # final_prim_slab_coords = final_prim_slab_coords * prim_slab_atom_mask.unsqueeze(-1)
        # final_ads_coords = final_ads_coords * ads_atom_mask.unsqueeze(-1)

        # # Apply refine_sc_mat to final_supercell_matrix (batch processing)
        # final_supercell_matrix = refine_sc_mat(final_supercell_matrix).to(
        #     device=final_supercell_matrix.device, 
        #     dtype=final_supercell_matrix.dtype
        # )

        # All outputs are already in raw space after preconditioned_network_forward
        # Prepare the output dictionary (all in raw space)
        output = {
            "sampled_prim_slab_coords": final_prim_slab_coords,  # (batch_size * multiplicity, N, 3) raw space
            "sampled_ads_coords": final_ads_coords,  # (batch_size * multiplicity, M, 3) raw space
            "sampled_prim_virtual_coords": final_prim_virtual_coords,  # (batch_size * multiplicity, 3, 3) raw space
            "sampled_supercell_virtual_coords": final_supercell_virtual_coords,  # (batch_size * multiplicity, 3, 3) raw space
            "sampled_scaling_factor": final_scaling_factor,  # (batch_size * multiplicity,) raw value
        }

        if return_trajectory:
            # Only append final refined values if refine_final=True
            # (if refine_final=False, the last step is already in trajectory from the loop)
            if refine_final:
                prim_slab_coord_trajectory.append(final_prim_slab_coords)
                ads_coord_trajectory.append(final_ads_coords)
                prim_virtual_coord_trajectory.append(final_prim_virtual_coords)
                supercell_virtual_coord_trajectory.append(final_supercell_virtual_coords)
                scaling_factor_trajectory.append(final_scaling_factor)

            output["prim_slab_coord_trajectory"] = torch.stack(prim_slab_coord_trajectory, dim=0)
            output["ads_coord_trajectory"] = torch.stack(ads_coord_trajectory, dim=0)
            output["prim_virtual_coord_trajectory"] = torch.stack(prim_virtual_coord_trajectory, dim=0)
            output["supercell_virtual_coord_trajectory"] = torch.stack(supercell_virtual_coord_trajectory, dim=0)
            output["scaling_factor_trajectory"] = torch.stack(scaling_factor_trajectory, dim=0)

        return output

    @torch.no_grad()
    def _refine_step(
        self,
        prim_slab_coords_t,
        ads_coords_t,
        prim_virtual_coords_t,
        supercell_virtual_coords_t,
        scaling_factor_t,
        prim_slab_atom_mask,
        ads_atom_mask,
        multiplicity,
        network_condition_kwargs,
    ):
        """
        Performs a final refinement step by feeding the t=1 state through the model once more.
        """
        net_result = self.preconditioned_network_forward(
            noised_prim_slab_coords=prim_slab_coords_t,
            noised_ads_coords=ads_coords_t,
            noised_prim_virtual_coords=prim_virtual_coords_t,
            noised_supercell_virtual_coords=supercell_virtual_coords_t,
            noised_scaling_factor=scaling_factor_t,
            time=1.0,
            training=False,
            network_condition_kwargs=network_condition_kwargs,
        )

        final_prim_slab_coords = net_result["denoised_prim_slab_coords"]
        final_ads_coords = net_result["denoised_ads_coords"]
        final_prim_virtual_coords = net_result["denoised_prim_virtual_coords"]
        final_supercell_virtual_coords = net_result["denoised_supercell_virtual_coords"]
        final_scaling_factor = net_result["denoised_scaling_factor"]

        # Ensure padding atoms remain at zero
        final_prim_slab_coords = final_prim_slab_coords * prim_slab_atom_mask.unsqueeze(-1)
        final_ads_coords = final_ads_coords * ads_atom_mask.unsqueeze(-1)

        return final_prim_slab_coords, final_ads_coords, final_prim_virtual_coords, final_supercell_virtual_coords, final_scaling_factor
