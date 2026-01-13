"""De novo generation script for MinCatFlow models.

Generates new catalyst structures using trained flow matching models.

Usage:
    python src/scripts/generate.py configs/default/test.yaml --checkpoint path/to/checkpoint.ckpt
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.catgen.data.conversions import virtual_coords_to_lattice_and_supercell


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_datamodule(config: dict):
    """Create data module."""
    from omegaconf import DictConfig
    from src.catgen.data.datamodule import LMDBDataModule

    # Convert dicts to DictConfig for attribute access
    batch_size = DictConfig(config["data"]["batch_size"])
    num_workers = DictConfig(config["data"]["num_workers"])

    return LMDBDataModule(
        train_lmdb_path=config["data"]["train_lmdb_path"],
        val_lmdb_path=config["data"]["val_lmdb_path"],
        test_lmdb_path=config["data"].get("test_lmdb_path"),
        batch_size=batch_size,
        num_workers=num_workers,
        preload_to_ram=config["data"].get("preload_to_ram", False),
    )


def load_model(checkpoint_path: str):
    """Load model from checkpoint."""
    from src.catgen.module.effcat_module import EffCatModule

    # Load model from checkpoint
    model = EffCatModule.load_from_checkpoint(
        checkpoint_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    return model


def run_generation(
    model,
    datamodule,
    num_samples: int = 5,
    sampling_steps: int = 10,
    seed: int = 42,
):
    """Run de novo generation on validation samples."""
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Setup datamodule and get validation loader
    datamodule.setup("fit")  # Use "fit" stage to get both train and val datasets
    val_loader = datamodule.val_dataloader()

    # Get a batch of samples
    batch = next(iter(val_loader))

    # Move batch to device
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Limit to num_samples
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.shape[0] > num_samples:
            batch[k] = v[:num_samples]

    print(f"\nGenerating {num_samples} samples with {sampling_steps} sampling steps...")
    print(f"Device: {device}")

    # Prepare for sampling
    feats = batch
    prim_slab_atom_mask = feats["prim_slab_atom_pad_mask"]
    ads_atom_mask = feats["ads_atom_pad_mask"]
    network_condition_kwargs = dict(feats=feats)

    # Time the generation
    start_time = time.time()

    with torch.no_grad():
        samples = model.structure_module.sample(
            prim_slab_atom_mask=prim_slab_atom_mask,
            ads_atom_mask=ads_atom_mask,
            num_sampling_steps=sampling_steps,
            multiplicity=1,
            center_coords=False,
            refine_final=False,
            return_trajectory=False,
            **network_condition_kwargs,
        )

    generation_time = time.time() - start_time

    return samples, batch, generation_time


def print_results(samples: dict, batch: dict, generation_time: float):
    """Print generation results."""
    print(f"\n{'='*60}")
    print("Generation Results")
    print(f"{'='*60}")
    print(f"Generation time: {generation_time:.3f}s")
    print(f"Samples generated: {samples['sampled_prim_slab_coords'].shape[0]}")

    # Print sample statistics
    print(f"\nGenerated coordinates statistics:")
    prim_coords = samples["sampled_prim_slab_coords"]
    ads_coords = samples["sampled_ads_coords"]

    # Convert virtual coords to lattice params and supercell matrix
    prim_virtual = samples["sampled_prim_virtual_coords"]
    supercell_virtual = samples["sampled_supercell_virtual_coords"]
    lattice, supercell = virtual_coords_to_lattice_and_supercell(prim_virtual, supercell_virtual)

    print(f"  Primitive slab coords shape: {prim_coords.shape}")
    print(f"    Mean: {prim_coords.mean().item():.4f}, Std: {prim_coords.std().item():.4f}")
    print(f"    Min: {prim_coords.min().item():.4f}, Max: {prim_coords.max().item():.4f}")

    print(f"  Adsorbate coords shape: {ads_coords.shape}")
    print(f"    Mean: {ads_coords.mean().item():.4f}, Std: {ads_coords.std().item():.4f}")
    print(f"    Min: {ads_coords.min().item():.4f}, Max: {ads_coords.max().item():.4f}")

    print(f"  Lattice shape: {lattice.shape}")
    print(f"    Lengths (a,b,c): {lattice[0, :3].tolist()}")
    print(f"    Angles (α,β,γ): {lattice[0, 3:].tolist()}")

    print(f"  Supercell matrix shape: {supercell.shape}")
    print(f"    First sample:\n{supercell[0].cpu().numpy()}")

    # Compare with ground truth if available
    if "prim_slab_cart_coords" in batch:
        gt_coords = batch["prim_slab_cart_coords"]
        diff = (prim_coords - gt_coords).abs().mean()
        print(f"\n  Mean abs diff from GT (prim slab): {diff.item():.4f}")

    if "ads_cart_coords" in batch:
        gt_ads = batch["ads_cart_coords"]
        diff = (ads_coords - gt_ads).abs().mean()
        print(f"  Mean abs diff from GT (adsorbate): {diff.item():.4f}")

    return {
        "prim_coords_mean": prim_coords.mean().item(),
        "prim_coords_std": prim_coords.std().item(),
        "ads_coords_mean": ads_coords.mean().item(),
        "ads_coords_std": ads_coords.std().item(),
        "lattice_first": lattice[0].tolist(),
        "supercell_first": supercell[0].tolist(),
        "generation_time": generation_time,
    }


def main(config_path: str, checkpoint_path: str, num_samples: int, sampling_steps: int, seed: int):
    """Main generation function."""
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    print(f"\nLoading checkpoint from: {checkpoint_path}")
    model = load_model(checkpoint_path)

    if torch.cuda.is_available():
        model = model.cuda()

    print("\nCreating data module...")
    datamodule = create_datamodule(config)

    samples, batch, generation_time = run_generation(
        model=model,
        datamodule=datamodule,
        num_samples=num_samples,
        sampling_steps=sampling_steps,
        seed=seed,
    )

    stats = print_results(samples, batch, generation_time)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="De novo generation with MinCatFlow")
    parser.add_argument("config_path", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--sampling_steps", type=int, default=10, help="Number of sampling steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    main(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        sampling_steps=args.sampling_steps,
        seed=args.seed,
    )
