#!/usr/bin/env python3
"""Create a toy dataset by extracting samples from existing LMDB.

This creates a small LMDB dataset for quick overfitting experiments.
Much faster to load than the full 460K sample dataset.

Usage:
    uv run python src/scripts/create_toy_dataset.py
    uv run python src/scripts/create_toy_dataset.py --n-samples 500 --output dataset/toy
"""

import argparse
import lmdb
import pickle
from pathlib import Path
from tqdm import tqdm


def extract_toy_dataset(
    source_path: str,
    output_path: Path,
    n_samples: int = 300,
    seed: int = 42,
):
    """Extract a subset of samples from source LMDB to create a toy dataset."""
    import numpy as np

    np.random.seed(seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing LMDB if it exists
    if output_path.exists():
        output_path.unlink()
    lock_path = output_path.parent / (output_path.name + "-lock")
    if lock_path.exists():
        lock_path.unlink()

    # Open source LMDB
    src_env = lmdb.open(str(source_path), readonly=True, lock=False, subdir=False)

    with src_env.begin() as txn:
        # Get total length
        length_data = txn.get(b"length")
        if length_data is None:
            raise ValueError("Source LMDB does not have 'length' key")
        total_length = pickle.loads(length_data)
        print(f"Source dataset has {total_length} samples")

        # Randomly select indices
        n_samples = min(n_samples, total_length)
        selected_indices = np.random.choice(total_length, n_samples, replace=False)
        selected_indices = sorted(selected_indices)
        print(f"Extracting {n_samples} samples...")

        # Create output LMDB
        map_size = 1024 * 1024 * 500  # 500 MB should be plenty for toy dataset
        dst_env = lmdb.open(str(output_path), map_size=map_size, subdir=False)

        with dst_env.begin(write=True) as dst_txn:
            for new_idx, src_idx in enumerate(tqdm(selected_indices, desc="Extracting")):
                # Get sample from source
                key = f"{src_idx}".encode()
                value = txn.get(key)
                if value is None:
                    print(f"Warning: sample {src_idx} not found, skipping")
                    continue

                # Store with new contiguous index
                new_key = f"{new_idx}".encode()
                dst_txn.put(new_key, value)

            # Store the new length
            dst_txn.put(b"length", pickle.dumps(n_samples))

        dst_env.close()

    src_env.close()
    print(f"Created toy dataset at {output_path} with {n_samples} samples")


def main():
    parser = argparse.ArgumentParser(description="Create toy dataset from existing LMDB")
    parser.add_argument(
        "--source",
        type=str,
        default="dataset/train/dataset.lmdb",
        help="Source LMDB path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/toy",
        help="Output directory for toy dataset",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=300,
        help="Number of samples to extract",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    print("Creating toy dataset for quick overfitting experiments...")
    print()

    # Create training data (use same samples for train and val in overfitting)
    train_path = output_dir / "dataset.lmdb"
    print(f"Creating toy dataset: {train_path}")
    extract_toy_dataset(
        source_path=args.source,
        output_path=train_path,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    print()
    print("Toy dataset created successfully!")
    print(f"  Path: {train_path}")
    print(f"  Samples: {args.n_samples}")
    print()
    print("To use this dataset, update your config:")
    print(f'  train_lmdb_path: "{train_path}"')
    print(f'  val_lmdb_path: "{train_path}"')


if __name__ == "__main__":
    main()
