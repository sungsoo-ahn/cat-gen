"""Compute dataset statistics for normalization.

Usage:
    PYTHONPATH=. uv run python src/scripts/compute_stats.py dataset/train/dataset.lmdb
"""

import argparse
import pickle
from pathlib import Path

import lmdb
import numpy as np
from tqdm import tqdm


def compute_statistics(lmdb_path: str) -> dict:
    """Compute statistics from LMDB dataset.

    Args:
        lmdb_path: Path to LMDB file

    Returns:
        Dictionary with computed statistics
    """
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=1,
    )

    # Collect values
    scaling_factors = []
    n_slabs = []
    n_vacs = []

    # Get all keys
    with env.begin() as txn:
        keys = []
        cursor = txn.cursor()
        for key, _ in cursor:
            key_str = key.decode("ascii")
            if key_str != "length":
                keys.append(key_str)

    print(f"Found {len(keys)} samples in {lmdb_path}")

    # Iterate through all samples
    with env.begin() as txn:
        for key in tqdm(keys, desc="Computing statistics"):
            value = txn.get(key.encode("ascii"))
            data = pickle.loads(value)

            n_slab = data["n_slab"]
            n_vac = data["n_vac"]
            scaling_factor = (n_vac + n_slab) / n_slab

            scaling_factors.append(scaling_factor)
            n_slabs.append(n_slab)
            n_vacs.append(n_vac)

    env.close()

    # Compute statistics
    scaling_factors = np.array(scaling_factors)
    n_slabs = np.array(n_slabs)
    n_vacs = np.array(n_vacs)

    stats = {
        "scaling_factor": {
            "mean": float(np.mean(scaling_factors)),
            "std": float(np.std(scaling_factors)),
            "min": float(np.min(scaling_factors)),
            "max": float(np.max(scaling_factors)),
            "median": float(np.median(scaling_factors)),
        },
        "n_slab": {
            "mean": float(np.mean(n_slabs)),
            "std": float(np.std(n_slabs)),
            "min": int(np.min(n_slabs)),
            "max": int(np.max(n_slabs)),
        },
        "n_vac": {
            "mean": float(np.mean(n_vacs)),
            "std": float(np.std(n_vacs)),
            "min": int(np.min(n_vacs)),
            "max": int(np.max(n_vacs)),
        },
        "num_samples": len(scaling_factors),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics for normalization")
    parser.add_argument("lmdb_path", type=str, help="Path to LMDB file")
    args = parser.parse_args()

    if not Path(args.lmdb_path).exists():
        print(f"Error: {args.lmdb_path} does not exist")
        return

    stats = compute_statistics(args.lmdb_path)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(f"\nNum samples: {stats['num_samples']}")

    print("\nScaling Factor (n_vac + n_slab) / n_slab:")
    sf = stats["scaling_factor"]
    print(f"  Mean: {sf['mean']:.6f}")
    print(f"  Std:  {sf['std']:.6f}")
    print(f"  Min:  {sf['min']:.6f}")
    print(f"  Max:  {sf['max']:.6f}")
    print(f"  Median: {sf['median']:.6f}")

    print("\nn_slab:")
    ns = stats["n_slab"]
    print(f"  Mean: {ns['mean']:.2f}")
    print(f"  Std:  {ns['std']:.2f}")
    print(f"  Min:  {ns['min']}")
    print(f"  Max:  {ns['max']}")

    print("\nn_vac:")
    nv = stats["n_vac"]
    print(f"  Mean: {nv['mean']:.2f}")
    print(f"  Std:  {nv['std']:.2f}")
    print(f"  Min:  {nv['min']}")
    print(f"  Max:  {nv['max']}")

    print("\n" + "=" * 60)
    print("YAML CONFIG (copy to prior_sampler_args):")
    print("=" * 60)
    print(f"  scaling_factor_mean: {sf['mean']:.6f}")
    print(f"  scaling_factor_std: {sf['std']:.6f}")


if __name__ == "__main__":
    main()
