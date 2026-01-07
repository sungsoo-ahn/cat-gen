"""Compute normalization statistics from training LMDB dataset.

This script computes mean and std for:
- Lattice lengths (in nm after conversion from Angstrom)
- Lattice angles (in radians after conversion from degrees)
- Scaling factor

Usage:
    uv run python src/scripts/compute_norm_stats.py <path_to_train_lmdb>

Example:
    uv run python src/scripts/compute_norm_stats.py dataset/train/dataset.lmdb
"""

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.reimplementation.data.lmdb_dataset import LMDBCachedDataset, cell_to_lattice_params


def compute_statistics(lmdb_path: str) -> dict:
    """Compute normalization statistics from LMDB dataset.

    Args:
        lmdb_path: Path to the training LMDB file

    Returns:
        Dictionary with computed statistics
    """
    print(f"Loading dataset from: {lmdb_path}")
    dataset = LMDBCachedDataset(lmdb_path, preload_to_ram=True)

    # Collect all values
    lattice_lengths_nm = []  # in nm
    lattice_angles_rad = []  # in radians
    scaling_factors = []

    print("Computing statistics...")
    for idx in tqdm(range(len(dataset)), desc="Processing samples"):
        sample = dataset[idx]

        # Get lattice parameters from primitive slab cell
        primitive_slab = sample["primitive_slab"]
        lattice_params = cell_to_lattice_params(primitive_slab.cell[:])

        # Convert to target units
        lengths_ang = lattice_params[:3]  # a, b, c in Angstrom
        angles_deg = lattice_params[3:]   # alpha, beta, gamma in degrees

        lengths_nm = lengths_ang * 0.1  # Angstrom -> nm
        angles_rad = angles_deg * (np.pi / 180.0)  # degrees -> radians

        lattice_lengths_nm.append(lengths_nm)
        lattice_angles_rad.append(angles_rad)

        # Compute scaling factor
        n_slab = sample["n_slab"]
        n_vac = sample["n_vac"]
        sf = (n_vac + n_slab) / n_slab
        scaling_factors.append(sf)

    # Convert to numpy arrays
    lattice_lengths_nm = np.array(lattice_lengths_nm)  # (N, 3)
    lattice_angles_rad = np.array(lattice_angles_rad)  # (N, 3)
    scaling_factors = np.array(scaling_factors)  # (N,)

    # Compute mean and std
    stats = {
        "lattice_length_mean": lattice_lengths_nm.mean(axis=0).tolist(),
        "lattice_length_std": lattice_lengths_nm.std(axis=0).tolist(),
        "lattice_angle_mean": lattice_angles_rad.mean(axis=0).tolist(),
        "lattice_angle_std": lattice_angles_rad.std(axis=0).tolist(),
        "scaling_factor_mean": float(scaling_factors.mean()),
        "scaling_factor_std": float(scaling_factors.std()),
    }

    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python src/scripts/compute_norm_stats.py <path_to_train_lmdb>")
        print("Example: uv run python src/scripts/compute_norm_stats.py dataset/train/dataset.lmdb")
        sys.exit(1)

    lmdb_path = sys.argv[1]

    if not Path(lmdb_path).exists():
        print(f"Error: LMDB file not found: {lmdb_path}")
        sys.exit(1)

    stats = compute_statistics(lmdb_path)

    print("\n" + "=" * 60)
    print("COMPUTED NORMALIZATION STATISTICS")
    print("=" * 60)
    print("\nAdd these to your config files under 'prior_sampler_args':\n")

    print("  # Lattice normalization (after unit conversion)")
    print(f"  lattice_length_mean: {stats['lattice_length_mean']}  # nm")
    print(f"  lattice_length_std: {stats['lattice_length_std']}  # nm")
    print(f"  lattice_angle_mean: {stats['lattice_angle_mean']}  # radians")
    print(f"  lattice_angle_std: {stats['lattice_angle_std']}  # radians")
    print()
    print("  # Scaling factor normalization")
    print(f"  scaling_factor_mean: {stats['scaling_factor_mean']}")
    print(f"  scaling_factor_std: {stats['scaling_factor_std']}")

    print("\n" + "=" * 60)

    # Also print in YAML format for easy copy-paste
    print("\nYAML format (ready to copy):\n")
    print(f"  lattice_length_mean: [{', '.join(f'{v:.6f}' for v in stats['lattice_length_mean'])}]")
    print(f"  lattice_length_std: [{', '.join(f'{v:.6f}' for v in stats['lattice_length_std'])}]")
    print(f"  lattice_angle_mean: [{', '.join(f'{v:.6f}' for v in stats['lattice_angle_mean'])}]")
    print(f"  lattice_angle_std: [{', '.join(f'{v:.6f}' for v in stats['lattice_angle_std'])}]")
    print(f"  scaling_factor_mean: {stats['scaling_factor_mean']:.6f}")
    print(f"  scaling_factor_std: {stats['scaling_factor_std']:.6f}")


if __name__ == "__main__":
    main()
