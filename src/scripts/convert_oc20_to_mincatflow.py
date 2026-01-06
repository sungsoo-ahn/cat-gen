#!/usr/bin/env python3
"""Convert OC20 IS2RE LMDB data to MinCatFlow format.

MinCatFlow uses a custom LMDB format where catalyst structures are decomposed into:
- primitive_slab: ASE Atoms (primitive unit cell)
- supercell_matrix: transformation to reconstruct supercell
- adsorbate positions and types

This script provides a framework for the conversion, but the actual
primitive cell finding algorithm requires careful implementation.

Usage:
    uv run python src/scripts/convert_oc20_to_mincatflow.py \
        --input dataset/oc20_raw/is2re/train \
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
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm

from ase import Atoms
from ase.io import read as ase_read
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor


def identify_adsorbate_atoms(atoms: Atoms, tag_value: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify adsorbate atoms based on tags.
    
    In OC20, atoms are typically tagged as:
    - 0: bulk atoms
    - 1: surface atoms  
    - 2: adsorbate atoms
    
    Args:
        atoms: ASE Atoms object
        tag_value: Tag value for adsorbate atoms (default: 2)
    
    Returns:
        Tuple of (adsorbate_indices, slab_indices)
    """
    tags = atoms.get_tags()
    adsorbate_idx = np.where(tags == tag_value)[0]
    slab_idx = np.where(tags != tag_value)[0]
    return adsorbate_idx, slab_idx


def find_primitive_cell(slab_atoms: Atoms) -> Tuple[Atoms, np.ndarray]:
    """
    Find the primitive cell of a slab structure.
    
    This is a complex operation that may require:
    1. Converting to pymatgen Structure
    2. Using SpacegroupAnalyzer to find primitive cell
    3. Computing the transformation matrix
    
    Note: This is a simplified implementation. Full MinCatFlow data
    preparation likely uses more sophisticated methods.
    
    Args:
        slab_atoms: ASE Atoms of the slab (without adsorbate)
    
    Returns:
        Tuple of (primitive_slab, supercell_matrix)
    """
    adaptor = AseAtomsAdaptor()
    
    # Convert to pymatgen Structure
    structure = adaptor.get_structure(slab_atoms)
    
    # Try to find primitive cell using symmetry analysis
    # This may not work for all slab structures
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
        primitive_struct = analyzer.find_primitive()
        
        if primitive_struct is None:
            # Fallback: use original as primitive
            primitive_struct = structure
            supercell_matrix = np.eye(3)
        else:
            # Compute supercell matrix
            # This is approximate - full implementation would be more rigorous
            prim_lattice = primitive_struct.lattice.matrix
            super_lattice = structure.lattice.matrix
            # supercell_matrix @ prim_lattice = super_lattice
            supercell_matrix = np.linalg.solve(prim_lattice.T, super_lattice.T).T
            
    except Exception as e:
        print(f"Warning: Primitive cell finding failed: {e}")
        primitive_struct = structure
        supercell_matrix = np.eye(3)
    
    # Convert back to ASE Atoms
    primitive_atoms = adaptor.get_atoms(primitive_struct)
    
    return primitive_atoms, supercell_matrix.astype(np.float32)


def convert_sample(atoms: Atoms, energy: Optional[float] = None) -> Dict[str, Any]:
    """
    Convert a single OC20 sample to MinCatFlow format.
    
    Args:
        atoms: ASE Atoms object (catalyst + adsorbate system)
        energy: Reference energy (optional)
    
    Returns:
        Dictionary in MinCatFlow format
    """
    # Identify adsorbate and slab atoms
    ads_idx, slab_idx = identify_adsorbate_atoms(atoms)
    
    # Extract adsorbate information
    ads_atomic_numbers = atoms.numbers[ads_idx]
    ads_pos = atoms.positions[ads_idx]
    
    # Extract slab atoms
    slab_atoms = atoms[slab_idx]
    
    # Find primitive cell
    primitive_slab, supercell_matrix = find_primitive_cell(slab_atoms)
    
    # Estimate n_slab and n_vac from cell dimensions
    # This is approximate - real values should come from metadata
    cell_z = atoms.cell[2, 2]
    slab_z = slab_atoms.positions[:, 2].max() - slab_atoms.positions[:, 2].min()
    n_slab = 3  # Default assumption
    n_vac = max(1, int((cell_z - slab_z) / slab_z))
    
    sample = {
        "primitive_slab": primitive_slab,
        "supercell_matrix": supercell_matrix,
        "ads_atomic_numbers": ads_atomic_numbers.astype(np.int64),
        "ads_pos": ads_pos.astype(np.float32),
        "ref_ads_pos": ads_pos.astype(np.float32),  # Use same as ads_pos
        "n_slab": n_slab,
        "n_vac": n_vac,
        "ref_energy": energy if energy is not None else 0.0,
    }
    
    return sample


def read_oc20_lmdb(lmdb_path: Path) -> list:
    """
    Read samples from OC20 LMDB format.
    
    Args:
        lmdb_path: Path to LMDB directory
    
    Returns:
        List of (atoms, energy) tuples
    """
    samples = []
    
    env = lmdb.open(
        str(lmdb_path),
        subdir=True,  # OC20 uses directory format
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
    )
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            if key == b"length":
                continue
            
            try:
                data = pickle.loads(value)
                
                # OC20 format typically has 'pos', 'atomic_numbers', 'cell', etc.
                if isinstance(data, dict):
                    pos = data.get("pos", data.get("positions"))
                    numbers = data.get("atomic_numbers", data.get("numbers"))
                    cell = data.get("cell")
                    tags = data.get("tags", np.zeros(len(numbers)))
                    energy = data.get("y", data.get("energy", None))
                    
                    atoms = Atoms(
                        numbers=numbers,
                        positions=pos,
                        cell=cell,
                        pbc=True,
                        tags=tags,
                    )
                    samples.append((atoms, energy))
                    
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
        help="Path to OC20 LMDB directory",
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
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"Reading OC20 data from: {input_path}")
    print()
    
    # Check if input exists
    if not input_path.exists():
        print(f"ERROR: Input path does not exist: {input_path}")
        print()
        print("To download OC20 data, run:")
        print("  bash scripts/data/download_data.sh oc20")
        return
    
    # Read OC20 samples
    print("Loading OC20 samples...")
    oc20_samples = read_oc20_lmdb(input_path)
    print(f"Loaded {len(oc20_samples)} samples")
    
    if args.max_samples:
        oc20_samples = oc20_samples[:args.max_samples]
        print(f"Using first {len(oc20_samples)} samples")
    
    # Convert samples
    print()
    print("Converting to MinCatFlow format...")
    mincatflow_samples = []
    
    for atoms, energy in tqdm(oc20_samples, desc="Converting"):
        try:
            sample = convert_sample(atoms, energy)
            mincatflow_samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed to convert sample: {e}")
            continue
    
    print(f"Successfully converted {len(mincatflow_samples)} samples")
    
    # Write output
    print()
    print(f"Writing to: {output_path}")
    write_mincatflow_lmdb(mincatflow_samples, output_path)
    
    print()
    print("Conversion complete!")
    print()
    print("NOTE: This conversion uses a simplified primitive cell finding algorithm.")
    print("For production use, you may need:")
    print("  - More sophisticated primitive cell identification")
    print("  - Proper n_slab and n_vac from metadata")
    print("  - Correct reference energy computation")


if __name__ == "__main__":
    main()
