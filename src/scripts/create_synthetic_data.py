#!/usr/bin/env python3
"""Create synthetic test data for CatGen.

This creates a minimal LMDB dataset with fake catalyst structures
for testing the training pipeline.

Usage:
    uv run python src/scripts/create_synthetic_data.py
"""

import argparse
import lmdb
import pickle
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.build import fcc111


def create_synthetic_sample(idx: int, n_prim_slab_atoms: int = 16, n_ads_atoms: int = 3):
    """Create a single synthetic catalyst sample using ASE Atoms format."""
    # Create a simple FCC(111) slab as primitive slab
    a = np.random.uniform(3.5, 4.0)  # Lattice constant

    # Create a small slab
    primitive_slab = fcc111(
        "Cu",  # Use Cu as base
        size=(2, 2, 3),
        a=a,
        vacuum=10.0,
        periodic=True
    )

    # Randomly change some atoms to other elements
    symbols = primitive_slab.get_chemical_symbols()
    for i in range(len(symbols)):
        if np.random.random() < 0.3:
            symbols[i] = np.random.choice(["Pt", "Pd", "Ni"])
    primitive_slab.set_chemical_symbols(symbols)

    # Limit to requested number of atoms
    if len(primitive_slab) > n_prim_slab_atoms:
        primitive_slab = primitive_slab[:n_prim_slab_atoms]

    # Create adsorbate positions (above the surface)
    cell = primitive_slab.cell[:]
    surface_z = primitive_slab.positions[:, 2].max()

    ads_pos = np.zeros((n_ads_atoms, 3), dtype=np.float32)
    ads_pos[:, 0] = np.random.uniform(0, cell[0, 0], n_ads_atoms)
    ads_pos[:, 1] = np.random.uniform(0, cell[1, 1], n_ads_atoms)
    ads_pos[:, 2] = surface_z + np.random.uniform(1.5, 3.0, n_ads_atoms)

    # Adsorbate elements: CO, OH, H
    ads_elements = np.random.choice([1, 6, 8], size=n_ads_atoms)

    # Reference adsorbate position (same as ads_pos for synthetic data)
    ref_ads_pos = ads_pos.copy()

    # Supercell matrix (2x2x1) - stored as (3,3), will be flattened by collate
    supercell_matrix = np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 1]
    ], dtype=np.float32)  # Keep as (3,3)

    # Number of slab layers and vacuum layers
    n_slab = 3
    n_vac = 1

    sample = {
        "primitive_slab": primitive_slab,  # ASE Atoms object
        "ads_atomic_numbers": ads_elements,
        "ads_pos": ads_pos,
        "ref_ads_pos": ref_ads_pos,
        "supercell_matrix": supercell_matrix,  # Keep as (3,3)
        "n_slab": n_slab,
        "n_vac": n_vac,
        "ref_energy": np.random.uniform(-5.0, 0.0),  # Fake reference energy
    }

    return sample


def create_lmdb_dataset(output_path: Path, n_samples: int = 100):
    """Create an LMDB dataset with synthetic samples."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing LMDB if it exists
    if output_path.exists():
        output_path.unlink()
    lock_path = output_path.parent / (output_path.name + "-lock")
    if lock_path.exists():
        lock_path.unlink()

    # Create LMDB environment (subdir=False creates single file)
    map_size = 1024 * 1024 * 100  # 100 MB
    env = lmdb.open(str(output_path), map_size=map_size, subdir=False)

    with env.begin(write=True) as txn:
        for i in range(n_samples):
            # Vary the number of adsorbate atoms
            n_ads = np.random.randint(2, 5)

            sample = create_synthetic_sample(i, n_ads_atoms=n_ads)

            # Serialize with pickle
            key = f"{i}".encode()
            value = pickle.dumps(sample)
            txn.put(key, value)

        # Store the length
        txn.put(b"length", pickle.dumps(n_samples))

    env.close()
    print(f"Created {n_samples} samples at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create synthetic test data")
    parser.add_argument("--output", type=str, default="dataset", help="Output directory")
    parser.add_argument("--n-train", type=int, default=100, help="Number of training samples")
    parser.add_argument("--n-val", type=int, default=20, help="Number of validation samples")
    args = parser.parse_args()

    output_dir = Path(args.output)

    print("Creating synthetic dataset for testing...")
    print()

    # Create training data
    train_path = output_dir / "train" / "dataset.lmdb"
    print(f"Creating training data: {train_path}")
    create_lmdb_dataset(train_path, n_samples=args.n_train)

    # Create validation data
    val_path = output_dir / "val_id" / "dataset.lmdb"
    print(f"Creating validation data: {val_path}")
    create_lmdb_dataset(val_path, n_samples=args.n_val)

    print()
    print("Synthetic dataset created successfully!")
    print(f"  Training: {args.n_train} samples")
    print(f"  Validation: {args.n_val} samples")


if __name__ == "__main__":
    main()
