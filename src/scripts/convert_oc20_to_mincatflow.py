#!/usr/bin/env python3
"""Convert OC20 IS2RE LMDB data to MinCatFlow format.

MinCatFlow format decomposes catalyst structures into:
- primitive_slab: ASE Atoms (UNRELAXED primitive unit cell of slab)
- supercell_matrix: transformation to reconstruct supercell from primitive
- ads_pos: RELAXED adsorbate positions
- ref_ads_pos: Reference (ground truth) adsorbate positions

OC20 IS2RE format contains:
- pos: Initial (unrelaxed) atomic positions
- pos_relaxed: Relaxed atomic positions
- tags: 0=subsurface, 1=surface, 2=adsorbate
- y_relaxed: Relaxed energy

The conversion extracts:
1. Slab from INITIAL structure (tags 0,1) -> find primitive cell
2. Adsorbate from RELAXED structure (tag 2)

Usage:
    uv run python src/scripts/convert_oc20_to_mincatflow.py \
        --input dataset/oc20_raw/is2res_train_val_test_lmdbs/data/is2re/10k/train/data.lmdb \
        --output dataset/train/dataset.lmdb

References:
    - pymatgen SpacegroupAnalyzer for primitive cell finding
    - ASE for structure manipulation
"""

import argparse
import lmdb
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm

from ase import Atoms
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor


def extract_pyg_data(data) -> Dict[str, Any]:
    """
    Extract data from PyG Data object (handles version compatibility).

    Args:
        data: PyG Data object (may be from older version)

    Returns:
        Dictionary with extracted fields
    """
    # Access internal dict to avoid PyG version issues
    internal = object.__getattribute__(data, '__dict__')

    result = {}
    fields = ['pos', 'pos_relaxed', 'atomic_numbers', 'cell', 'tags',
              'y_init', 'y_relaxed', 'sid', 'natoms', 'fixed']

    for field in fields:
        if field in internal:
            val = internal[field]
            # Convert torch tensors to numpy
            if hasattr(val, 'numpy'):
                val = val.numpy()
            result[field] = val

    return result


def find_primitive_cell(slab_atoms: Atoms, symprec: float = 0.1) -> Tuple[Atoms, np.ndarray]:
    """
    Find the primitive cell of a slab structure.

    Uses pymatgen's SpacegroupAnalyzer to identify symmetry and
    find the primitive cell, then computes the supercell matrix.

    Args:
        slab_atoms: ASE Atoms of the slab (without adsorbate)
        symprec: Symmetry precision for structure matching

    Returns:
        Tuple of (primitive_slab, supercell_matrix)
    """
    adaptor = AseAtomsAdaptor()

    # Convert to pymatgen Structure
    structure = adaptor.get_structure(slab_atoms)

    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
        primitive_struct = analyzer.find_primitive()

        if primitive_struct is None:
            # Fallback: use original as primitive
            primitive_struct = structure
            supercell_matrix = np.eye(3)
        else:
            # Compute supercell matrix: supercell = primitive @ matrix
            prim_lattice = primitive_struct.lattice.matrix
            super_lattice = structure.lattice.matrix
            # Solve: supercell_matrix @ prim_lattice = super_lattice
            supercell_matrix = np.linalg.solve(prim_lattice.T, super_lattice.T).T

    except Exception as e:
        print(f"Warning: Primitive cell finding failed: {e}")
        primitive_struct = structure
        supercell_matrix = np.eye(3)

    # Convert back to ASE Atoms
    primitive_atoms = adaptor.get_atoms(primitive_struct)

    return primitive_atoms, supercell_matrix.astype(np.float32)


def convert_oc20_sample(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a single OC20 IS2RE sample to MinCatFlow format.

    Args:
        data: Dictionary with OC20 fields (pos, pos_relaxed, tags, etc.)

    Returns:
        Dictionary in MinCatFlow format, or None if conversion fails
    """
    pos_initial = data['pos']
    pos_relaxed = data['pos_relaxed']
    atomic_numbers = data['atomic_numbers'].astype(np.int64)
    cell = data['cell']
    tags = data['tags']
    y_relaxed = data.get('y_relaxed', 0.0)

    # Handle cell shape: (1, 3, 3) -> (3, 3)
    if cell.ndim == 3:
        cell = cell[0]

    # Identify adsorbate (tag=2) and slab (tag=0,1) atoms
    ads_mask = tags == 2
    slab_mask = tags != 2

    ads_idx = np.where(ads_mask)[0]
    slab_idx = np.where(slab_mask)[0]

    if len(ads_idx) == 0:
        return None  # No adsorbate atoms

    # Extract adsorbate from RELAXED structure
    ads_atomic_numbers = atomic_numbers[ads_idx]
    ads_pos_relaxed = pos_relaxed[ads_idx]

    # Extract slab from INITIAL structure
    slab_numbers = atomic_numbers[slab_idx]
    slab_pos_initial = pos_initial[slab_idx]

    # Create ASE Atoms for slab
    slab_atoms = Atoms(
        numbers=slab_numbers,
        positions=slab_pos_initial,
        cell=cell,
        pbc=True
    )

    # Find primitive cell
    primitive_slab, supercell_matrix = find_primitive_cell(slab_atoms)

    # Estimate n_slab and n_vac
    cell_z = cell[2, 2]
    slab_z = slab_pos_initial[:, 2].max() - slab_pos_initial[:, 2].min()
    n_slab = 3  # Default assumption
    n_vac = max(1, int((cell_z - slab_z) / slab_z)) if slab_z > 0 else 1

    sample = {
        "primitive_slab": primitive_slab,
        "supercell_matrix": supercell_matrix,
        "ads_atomic_numbers": ads_atomic_numbers,
        "ads_pos": ads_pos_relaxed.astype(np.float32),
        "ref_ads_pos": ads_pos_relaxed.astype(np.float32),
        "n_slab": n_slab,
        "n_vac": n_vac,
        "ref_energy": float(y_relaxed) if y_relaxed is not None else 0.0,
    }

    return sample


def read_oc20_is2re_lmdb(lmdb_path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Read samples from OC20 IS2RE LMDB format.

    Args:
        lmdb_path: Path to LMDB file (data.lmdb)
        max_samples: Maximum number of samples to read (None for all)

    Returns:
        List of data dictionaries
    """
    samples = []

    # OC20 IS2RE uses subdir=False (single file)
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
    )

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in tqdm(cursor, desc="Reading OC20 LMDB"):
            if key == b"length":
                continue

            if max_samples and len(samples) >= max_samples:
                break

            try:
                data = pickle.loads(value)
                extracted = extract_pyg_data(data)
                samples.append(extracted)
            except Exception as e:
                print(f"Warning: Failed to parse sample {key}: {e}")
                continue

    env.close()
    return samples


def write_mincatflow_lmdb(samples: list, output_path: Path):
    """
    Write samples in MinCatFlow LMDB format.
    
    Args:
        samples: List of sample dictionaries
        output_path: Output LMDB file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing file
    if output_path.exists():
        output_path.unlink()
    lock_path = output_path.parent / (output_path.name + "-lock")
    if lock_path.exists():
        lock_path.unlink()
    
    # Create LMDB (subdir=False for single file format)
    map_size = 1024 * 1024 * 1024 * 10  # 10 GB
    env = lmdb.open(str(output_path), map_size=map_size, subdir=False)
    
    with env.begin(write=True) as txn:
        for idx, sample in enumerate(tqdm(samples, desc="Writing LMDB")):
            key = f"{idx}".encode()
            value = pickle.dumps(sample)
            txn.put(key, value)
        
        # Store length
        txn.put(b"length", pickle.dumps(len(samples)))
    
    env.close()
    print(f"Wrote {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert OC20 IS2RE LMDB to MinCatFlow format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to OC20 LMDB file (e.g., data.lmdb)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output MinCatFlow LMDB file path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to convert (for testing)",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=0.1,
        help="Symmetry precision for primitive cell finding (default: 0.1)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 60)
    print("OC20 IS2RE to MinCatFlow Conversion")
    print("=" * 60)
    print()
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    # Check if input exists
    if not input_path.exists():
        print(f"ERROR: Input path does not exist: {input_path}")
        print()
        print("To download OC20 data, run:")
        print("  bash scripts/data/download_data.sh oc20")
        return

    # Read OC20 samples
    print("Step 1: Reading OC20 IS2RE samples...")
    oc20_samples = read_oc20_is2re_lmdb(input_path, max_samples=args.max_samples)
    print(f"Loaded {len(oc20_samples)} samples")

    # Convert samples
    print()
    print("Step 2: Converting to MinCatFlow format...")
    print("  - Slab: from INITIAL structure (unrelaxed)")
    print("  - Adsorbate: from RELAXED structure")
    print("  - Finding primitive cells...")
    print()

    mincatflow_samples = []
    failed = 0

    for data in tqdm(oc20_samples, desc="Converting"):
        try:
            sample = convert_oc20_sample(data)
            if sample is not None:
                mincatflow_samples.append(sample)
            else:
                failed += 1
        except Exception as e:
            print(f"Warning: Failed to convert sample: {e}")
            failed += 1
            continue

    print()
    print(f"Successfully converted: {len(mincatflow_samples)} samples")
    if failed > 0:
        print(f"Failed/skipped: {failed} samples")

    # Write output
    print()
    print("Step 3: Writing MinCatFlow LMDB...")
    write_mincatflow_lmdb(mincatflow_samples, output_path)

    print()
    print("=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print()
    print("Output format:")
    print("  - primitive_slab: ASE Atoms (unrelaxed primitive cell)")
    print("  - supercell_matrix: (3,3) transformation matrix")
    print("  - ads_pos: (N,3) relaxed adsorbate positions")
    print("  - ref_ads_pos: (N,3) reference positions")
    print("  - ads_atomic_numbers: (N,) adsorbate elements")
    print("  - n_slab, n_vac: layer counts")
    print("  - ref_energy: relaxed energy")


if __name__ == "__main__":
    main()
