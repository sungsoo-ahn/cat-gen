#!/usr/bin/env python3
"""Evaluate overfitting experiment by comparing generated vs ground truth structures.

Usage:
    uv run python src/scripts/evaluate_overfit.py \
        configs/default/overfit.yaml \
        --checkpoint data/catgen/overfit_test/checkpoints/last.ckpt \
        --wandb-project CatGen \
        --wandb-run-name overfit_evaluation
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
import wandb
from ase.io import write as ase_write

from src.catgen.module.catgen_module import CatGen
from src.catgen.data.lmdb_dataset import LMDBCachedDataset, collate_fn_with_dynamic_padding
from src.catgen.scripts.assemble import assemble
from src.catgen.data.conversions import virtual_coords_to_lattice_and_supercell


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, device: str = "cuda") -> CatGen:
    """Load model from checkpoint."""
    model = CatGen.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    model.eval()
    model.to(device)
    return model


def load_single_sample(config: dict, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Load single sample from the overfitting dataset."""
    dataset = LMDBCachedDataset(
        lmdb_path=config["data"]["train_lmdb_path"],
        preload_to_ram=True,
    )

    # Get first sample
    raw_sample = dataset[0]

    # Collate into batch format
    batch = collate_fn_with_dynamic_padding([raw_sample])

    # Move to device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    return batch


def compute_coordinate_rmsd(
    pred_coords: np.ndarray,
    true_coords: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute RMSD between predicted and true coordinates."""
    pred_masked = pred_coords[mask.astype(bool)]
    true_masked = true_coords[mask.astype(bool)]

    if len(pred_masked) == 0:
        return 0.0

    diff = pred_masked - true_masked
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=-1)))
    return rmsd


def generate_with_trajectory(
    model: CatGen,
    batch: dict[str, torch.Tensor],
    seed: int = 42,
) -> dict[str, Any]:
    """Generate structure and return full trajectory."""
    # Set seed for reproducibility (affects prior sampling)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Get masks
    prim_slab_atom_mask = batch["prim_slab_atom_pad_mask"]
    ads_atom_mask = batch["ads_atom_pad_mask"]

    # Generate with trajectory
    with torch.no_grad():
        output = model.structure_module.sample(
            prim_slab_atom_mask=prim_slab_atom_mask,
            ads_atom_mask=ads_atom_mask,
            num_sampling_steps=model.validation_args.get("sampling_steps", 50),
            multiplicity=1,
            center_coords=False,
            refine_final=False,
            return_trajectory=True,
            feats=batch,
        )

    return output


def compare_structures(
    generated: dict[str, Any],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compare generated structures with ground truth."""
    metrics = {}

    # Extract numpy arrays
    gen_prim_coords = generated["sampled_prim_slab_coords"].cpu().numpy()[0]
    gen_ads_coords = generated["sampled_ads_coords"].cpu().numpy()[0]
    gen_scaling = generated["sampled_scaling_factor"].cpu().numpy()[0]

    # Convert virtual coords to lattice params and supercell matrix
    gen_prim_virtual = generated["sampled_prim_virtual_coords"]
    gen_supercell_virtual = generated["sampled_supercell_virtual_coords"]
    gen_lattice_tensor, gen_supercell_tensor = virtual_coords_to_lattice_and_supercell(
        gen_prim_virtual, gen_supercell_virtual
    )
    gen_lattice = gen_lattice_tensor.cpu().numpy()[0]
    gen_supercell = gen_supercell_tensor.cpu().numpy()[0]

    true_prim_coords = batch["prim_slab_cart_coords"].cpu().numpy()[0]
    true_ads_coords = batch["ads_cart_coords"].cpu().numpy()[0]
    true_lattice = batch["lattice"].cpu().numpy()[0]
    true_supercell = batch["supercell_matrix"].cpu().numpy()[0]
    true_scaling = batch["scaling_factor"].cpu().numpy()[0]

    prim_mask = batch["prim_slab_atom_pad_mask"].cpu().numpy()[0]
    ads_mask = batch["ads_atom_pad_mask"].cpu().numpy()[0]

    # Coordinate RMSD
    metrics["rmsd/coord/primitive_slab"] = compute_coordinate_rmsd(
        gen_prim_coords, true_prim_coords, prim_mask
    )

    if ads_mask.sum() > 0:
        metrics["rmsd/coord/adsorbate"] = compute_coordinate_rmsd(
            gen_ads_coords, true_ads_coords, ads_mask
        )
    else:
        metrics["rmsd/coord/adsorbate"] = 0.0

    # Lattice parameter differences
    metrics["mae/lattice/length"] = float(
        np.abs(gen_lattice[:3] - true_lattice[:3]).mean()
    )
    metrics["mae/lattice/angle"] = float(
        np.abs(gen_lattice[3:] - true_lattice[3:]).mean()
    )

    # Supercell matrix difference
    metrics["mae/supercell/mean"] = float(np.abs(gen_supercell - true_supercell).mean())
    metrics["mae/supercell/max"] = float(np.abs(gen_supercell - true_supercell).max())

    # Scaling factor difference
    metrics["mae/scaling_factor"] = float(np.abs(gen_scaling - true_scaling))

    return metrics


def log_to_wandb(
    generated: dict[str, Any],
    batch: dict[str, torch.Tensor],
    metrics: dict[str, float],
):
    """Log structures and metrics to W&B."""
    # Log metrics
    wandb.log(metrics)

    # Extract arrays
    gen_prim_coords = generated["sampled_prim_slab_coords"].cpu().numpy()[0]
    gen_ads_coords = generated["sampled_ads_coords"].cpu().numpy()[0]
    gen_scaling = generated["sampled_scaling_factor"].cpu().numpy()[0]

    # Convert virtual coords to lattice params and supercell matrix
    gen_prim_virtual = generated["sampled_prim_virtual_coords"]
    gen_supercell_virtual = generated["sampled_supercell_virtual_coords"]
    gen_lattice_tensor, gen_supercell_tensor = virtual_coords_to_lattice_and_supercell(
        gen_prim_virtual, gen_supercell_virtual
    )
    gen_lattice = gen_lattice_tensor.cpu().numpy()[0]
    gen_supercell = gen_supercell_tensor.cpu().numpy()[0]

    true_prim_coords = batch["prim_slab_cart_coords"].cpu().numpy()[0]
    true_ads_coords = batch["ads_cart_coords"].cpu().numpy()[0]
    true_lattice = batch["lattice"].cpu().numpy()[0]
    true_supercell = batch["supercell_matrix"].cpu().numpy()[0]
    true_scaling = batch["scaling_factor"].cpu().numpy()[0]

    prim_types = batch["ref_prim_slab_element"].cpu().numpy()[0]
    ads_types = batch["ref_ads_element"].cpu().numpy()[0]
    prim_mask = batch["prim_slab_atom_pad_mask"].cpu().numpy()[0]
    ads_mask = batch["ads_atom_pad_mask"].cpu().numpy()[0]

    temp_files = []

    try:
        # Assemble and log generated structure
        gen_atoms, _ = assemble(
            generated_prim_slab_coords=gen_prim_coords,
            generated_ads_coords=gen_ads_coords,
            generated_lattice=gen_lattice,
            generated_supercell_matrix=gen_supercell,
            generated_scaling_factor=gen_scaling,
            prim_slab_atom_types=prim_types,
            ads_atom_types=ads_types,
            prim_slab_atom_mask=prim_mask,
            ads_atom_mask=ads_mask,
        )

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            gen_path = f.name
        ase_write(gen_path, gen_atoms, format="proteindatabank")
        temp_files.append(gen_path)

        wandb.log(
            {
                "structures/generated": wandb.Molecule(
                    gen_path, caption="Generated Structure"
                ),
            }
        )

        # Assemble and log ground truth structure
        true_atoms, _ = assemble(
            generated_prim_slab_coords=true_prim_coords,
            generated_ads_coords=true_ads_coords,
            generated_lattice=true_lattice,
            generated_supercell_matrix=true_supercell,
            generated_scaling_factor=true_scaling,
            prim_slab_atom_types=prim_types,
            ads_atom_types=ads_types,
            prim_slab_atom_mask=prim_mask,
            ads_atom_mask=ads_mask,
        )

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            true_path = f.name
        ase_write(true_path, true_atoms, format="proteindatabank")
        temp_files.append(true_path)

        wandb.log(
            {
                "structures/ground_truth": wandb.Molecule(
                    true_path, caption="Ground Truth Structure"
                ),
            }
        )

        # Log trajectory visualizations (sample every few steps)
        if "prim_slab_coord_trajectory" in generated:
            trajectory = generated["prim_slab_coord_trajectory"].cpu().numpy()
            num_steps = trajectory.shape[0]
            step_interval = max(1, num_steps // 5)

            for step_idx in range(0, num_steps, step_interval):
                step_coords = trajectory[step_idx, 0]

                try:
                    # Convert virtual coord trajectory to lattice/supercell
                    step_prim_virtual = generated["prim_virtual_coord_trajectory"][step_idx:step_idx+1]
                    step_supercell_virtual = generated["supercell_virtual_coord_trajectory"][step_idx:step_idx+1]
                    step_lattice_tensor, step_supercell_tensor = virtual_coords_to_lattice_and_supercell(
                        step_prim_virtual, step_supercell_virtual
                    )
                    step_lattice = step_lattice_tensor.cpu().numpy()[0]
                    step_supercell = step_supercell_tensor.cpu().numpy()[0]

                    step_atoms, _ = assemble(
                        generated_prim_slab_coords=step_coords,
                        generated_ads_coords=generated["ads_coord_trajectory"]
                        .cpu()
                        .numpy()[step_idx, 0],
                        generated_lattice=step_lattice,
                        generated_supercell_matrix=step_supercell,
                        generated_scaling_factor=generated["scaling_factor_trajectory"]
                        .cpu()
                        .numpy()[step_idx, 0],
                        prim_slab_atom_types=prim_types,
                        ads_atom_types=ads_types,
                        prim_slab_atom_mask=prim_mask,
                        ads_atom_mask=ads_mask,
                    )

                    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
                        step_path = f.name
                    ase_write(step_path, step_atoms, format="proteindatabank")
                    temp_files.append(step_path)

                    t = step_idx / (num_steps - 1) if num_steps > 1 else 0.0
                    wandb.log(
                        {
                            f"trajectory/step_{step_idx:03d}_t_{t:.2f}": wandb.Molecule(
                                step_path, caption=f"t={t:.2f} (step {step_idx})"
                            ),
                        }
                    )
                except Exception as e:
                    print(f"Failed to log trajectory step {step_idx}: {e}")

    except Exception as e:
        print(f"Failed to assemble structure: {e}")

    finally:
        # Cleanup temp files
        for path in temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass


def main(
    config_path: str,
    checkpoint_path: str,
    wandb_project: str = "CatGen",
    wandb_run_name: Optional[str] = None,
    seed: int = 42,
):
    """Main evaluation function."""
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Initialize W&B
    wandb.init(
        project=wandb_project,
        name=wandb_run_name or "overfit_evaluation",
        config=config,
        tags=["overfit", "evaluation"],
    )

    print(f"Loading checkpoint from: {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint_path, device)

    print("Loading sample...")
    batch = load_single_sample(config, device)

    print("Generating structure with trajectory...")
    generated = generate_with_trajectory(model, batch, seed)

    print("Computing metrics...")
    metrics = compare_structures(generated, batch)

    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")

    print("\nLogging to W&B...")
    log_to_wandb(generated, batch, metrics)

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate overfitting experiment")
    parser.add_argument("config_path", type=str, help="Path to config file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="CatGen", help="W&B project"
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="W&B run name"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    main(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
    )
