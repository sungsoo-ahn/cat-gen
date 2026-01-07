#!/usr/bin/env python3
"""Convert OC20 IS2RE LMDB directly to MinCatFlow LMDB format.

Usage:
    uv run python src/scripts/oc20_to_mincatflow.py \
        --lmdb-path dataset/oc20_raw/is2re/all/train/data.lmdb \
        --mapping-path resources/oc20_data_mapping.pkl \
        --adsorbates-path resources/adsorbates.pkl \
        --output-path dataset/train/dataset.lmdb
"""

import argparse
import json
import lmdb
import math
import os
import pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from ase import Atoms
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from src.helpers import calculate_rmsd_pymatgen


def load_mapping(mapping_path: str) -> dict:
    """Load OC20 data mapping file."""
    with open(mapping_path, "rb") as f:
        return pickle.load(f)


def get_slab_params(mapping: dict, sid: int) -> dict | None:
    """Get slab parameters from mapping for a given system ID."""
    key = f"random{sid}"
    if key not in mapping:
        return None

    value = mapping[key]
    if not isinstance(value, dict):
        return None

    return {
        "bulk_mpid": value.get("bulk_mpid"),
        "miller_index": value.get("miller_index"),
        "ads_id": value.get("ads_id"),
    }


def compute_slab_layers(bulk_mpid: str, miller_index: tuple) -> tuple[int, int] | None:
    """Compute n_slab and n_vac layers using fairchem/pymatgen."""
    try:
        from fairchem.data.oc.core import Bulk
        from fairchem.data.oc.core.slab import standardize_bulk
        from pymatgen.core.surface import SlabGenerator

        bulk = Bulk(bulk_src_id_from_db=bulk_mpid)
        initial_structure = standardize_bulk(bulk.atoms)

        slab_gen = SlabGenerator(
            initial_structure=initial_structure,
            miller_index=miller_index,
            min_slab_size=7.0,
            min_vacuum_size=20.0,
            lll_reduce=False,
            center_slab=False,
            primitive=True,
            max_normal_search=1,
        )

        height = slab_gen._proj_height
        n_slab = math.ceil(slab_gen.min_slab_size / height)
        n_vac = math.ceil(slab_gen.min_vac_size / height)

        return n_slab, n_vac
    except Exception:
        return None


def extract_from_lmdb(lmdb_path: str, index: int) -> dict | None:
    """Extract data from OC20 LMDB at given index."""
    db = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    try:
        with db.begin() as txn:
            key = f"{index}".encode("ascii")
            value = txn.get(key)
            if value is None:
                return None
            data = pickle.loads(value)

        # Bypass PyG's property system by accessing __dict__ directly
        # This avoids version compatibility issues with older PyG data
        internal = object.__getattribute__(data, '__dict__')

        def to_numpy(val):
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                return val.cpu().numpy()
            return np.array(val)

        def to_scalar(val):
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                return val.item() if val.numel() == 1 else int(val[0])
            if isinstance(val, (list, tuple)):
                return val[0]
            return val

        sid = to_scalar(internal.get('sid'))
        tags = to_numpy(internal.get('tags'))
        atomic_numbers = to_numpy(internal.get('atomic_numbers'))
        pos = to_numpy(internal.get('pos'))
        pos_relaxed = to_numpy(internal.get('pos_relaxed'))
        cell = to_numpy(internal.get('cell'))
        y_relaxed = to_scalar(internal.get('y_relaxed'))

        # Validate required fields
        if any(x is None for x in [tags, atomic_numbers, pos, cell, y_relaxed]):
            return None

        if cell.shape == (1, 3, 3):
            cell = cell[0]

        # Filter atoms (tags 0, 1, 2 only)
        atom_mask = (tags == 0) | (tags == 1) | (tags == 2)
        ads_mask = (tags == 2)

        return {
            'sid': sid,
            'atomic_numbers': atomic_numbers[atom_mask],
            'positions': pos[atom_mask],
            'cell': cell,
            'tags': tags[atom_mask],
            'ads_pos_relaxed': pos_relaxed[ads_mask] if pos_relaxed is not None and np.any(ads_mask) else None,
            'ref_energy': y_relaxed,
        }
    except Exception:
        return None
    finally:
        db.close()


def process_to_mincatflow(data: dict, n_slab: int, n_vac: int, ref_ads_pos_canonical: np.ndarray) -> dict | None:
    """Convert extracted data to MinCatFlow format with primitive slab.

    Args:
        data: Extracted OC20 data
        n_slab: Number of slab layers
        n_vac: Number of vacuum layers
        ref_ads_pos_canonical: Canonical reference adsorbate positions from adsorbates.pkl
    """
    try:
        # Create ASE Atoms
        true_system = Atoms(
            numbers=data['atomic_numbers'],
            positions=data['positions'],
            cell=data['cell'],
            tags=data['tags'],
            pbc=True
        )

        # Separate slab and adsorbate
        slab_mask = (data['tags'] == 0) | (data['tags'] == 1)
        true_slab = true_system[slab_mask]

        adaptor = AseAtomsAdaptor()
        n_layers = n_slab + n_vac

        # Create tight slab (remove vacuum)
        tight_slab = true_slab.copy()
        tight_cell = tight_slab.get_cell()
        tight_cell[2] = tight_cell[2] * (n_slab / n_layers)
        tight_slab.set_cell(tight_cell)
        tight_slab.center()

        # Find primitive cell
        tight_slab_struct = adaptor.get_structure(tight_slab)
        prim_slab_struct = tight_slab_struct.get_primitive_structure(tolerance=0.1)

        # Standardize orientation
        a, b, c, alpha, beta, gamma = prim_slab_struct.lattice.parameters
        standard_lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

        L_orig = prim_slab_struct.lattice.matrix
        L_std = standard_lattice.matrix
        R = (np.linalg.inv(L_orig) @ L_std).T

        rotated_prim_struct = Structure(
            standard_lattice,
            prim_slab_struct.species,
            prim_slab_struct.frac_coords
        )

        # Compute supercell matrix
        rotated_tight_lattice = Lattice(tight_slab_struct.lattice.matrix @ R.T)
        sc_matrix = np.round(np.dot(
            rotated_tight_lattice.matrix,
            np.linalg.inv(standard_lattice.matrix)
        )).astype(int)

        # Verify reconstruction
        recon_struct = rotated_prim_struct.copy()
        recon_struct.make_supercell(sc_matrix, to_unit_cell=False)
        recon_tight_slab = adaptor.get_atoms(recon_struct)

        rmsd_tight, _ = calculate_rmsd_pymatgen(
            struct1=recon_tight_slab,
            struct2=tight_slab,
            ltol=0.2, stol=0.3, angle_tol=5,
            primitive_cell=False,
        )

        if rmsd_tight is None or rmsd_tight >= 1e-4:
            return None

        # Add vacuum back
        scaling_factor = n_layers / n_slab
        recon_slab = recon_tight_slab.copy()
        recon_cell = recon_slab.get_cell()
        recon_cell[2] = recon_cell[2] * scaling_factor
        recon_slab.set_cell(recon_cell)

        # Transform adsorbate for RMSD verification (using original positions)
        adsorbate_orig = true_system[data['tags'] == 2].copy()
        rotated_true_slab = true_slab.copy()
        rotated_true_slab.set_cell(true_slab.cell @ R.T)
        rotated_true_slab.set_positions(true_slab.positions @ R.T)

        rotated_adsorbate = adsorbate_orig.copy()
        rotated_adsorbate.set_positions(adsorbate_orig.positions @ R.T)
        diff_vector = recon_slab.get_center_of_mass() - rotated_true_slab.get_center_of_mass()
        rotated_adsorbate.translate(diff_vector)

        # Final verification with original positions
        rmsd_system, _ = calculate_rmsd_pymatgen(
            struct1=recon_slab + rotated_adsorbate,
            struct2=true_system,
            ltol=0.2, stol=0.3, angle_tol=5,
            primitive_cell=False,
        )

        if rmsd_system is None or rmsd_system >= 1e-4:
            return None

        # Use relaxed positions for the dataset output
        if data['ads_pos_relaxed'] is not None:
            relaxed_adsorbate = adsorbate_orig.copy()
            relaxed_adsorbate.set_positions(data['ads_pos_relaxed'] @ R.T)
            relaxed_adsorbate.translate(diff_vector)
            final_adsorbate = relaxed_adsorbate
        else:
            final_adsorbate = rotated_adsorbate

        # Convert primitive slab to ASE Atoms
        primitive_slab_atoms = adaptor.get_atoms(rotated_prim_struct)

        return {
            "sid": data['sid'],
            "primitive_slab": primitive_slab_atoms,
            "supercell_matrix": sc_matrix,
            "n_slab": n_slab,
            "n_vac": n_vac,
            "ads_atomic_numbers": final_adsorbate.numbers,
            "ads_pos": final_adsorbate.positions,  # Relaxed adsorbate positions
            "ref_ads_pos": ref_ads_pos_canonical,  # Canonical reference positions from adsorbates.pkl
            "ref_energy": data['ref_energy'],
        }
    except Exception:
        return None


# Global variables for worker processes
_mapping = None
_lmdb_path = None
_adsorbates = None
_slab_cache = {}


def init_worker(mapping_path: str, lmdb_path: str, adsorbates_path: str):
    """Initialize worker with shared data."""
    global _mapping, _lmdb_path, _adsorbates
    _mapping = load_mapping(mapping_path)
    _lmdb_path = lmdb_path
    with open(adsorbates_path, "rb") as f:
        _adsorbates = pickle.load(f)


def process_single(index: int) -> dict:
    """Process a single sample."""
    global _mapping, _lmdb_path, _adsorbates, _slab_cache

    result = {'index': index, 'status': 'error', 'data': None}

    # Extract from LMDB
    data = extract_from_lmdb(_lmdb_path, index)
    if data is None:
        result['status'] = 'extract_failed'
        return result

    sid = data['sid']
    if sid is None:
        result['status'] = 'no_sid'
        return result

    # Get slab parameters
    slab_params = get_slab_params(_mapping, sid)
    if slab_params is None or slab_params['bulk_mpid'] is None:
        result['status'] = 'no_mapping'
        return result

    bulk_mpid = slab_params['bulk_mpid']
    miller_index = slab_params['miller_index']
    ads_id = slab_params['ads_id']

    if miller_index is None:
        result['status'] = 'no_miller'
        return result

    if ads_id is None:
        result['status'] = 'no_ads_id'
        return result

    # Get canonical reference adsorbate positions from adsorbates.pkl
    if ads_id not in _adsorbates:
        result['status'] = 'ads_id_not_found'
        return result
    ref_ads_pos_canonical = _adsorbates[ads_id][0].get_positions()

    # Convert miller_index to tuple if needed
    if isinstance(miller_index, list):
        miller_index = tuple(miller_index)

    # Compute or get cached slab layers
    cache_key = (bulk_mpid, miller_index)
    if cache_key not in _slab_cache:
        layers = compute_slab_layers(bulk_mpid, miller_index)
        _slab_cache[cache_key] = layers
    else:
        layers = _slab_cache[cache_key]

    if layers is None:
        result['status'] = 'slab_compute_failed'
        return result

    n_slab, n_vac = layers

    # Convert to MinCatFlow format
    mincatflow_data = process_to_mincatflow(data, n_slab, n_vac, ref_ads_pos_canonical)
    if mincatflow_data is None:
        result['status'] = 'conversion_failed'
        return result

    result['status'] = 'success'
    result['data'] = mincatflow_data
    return result


def main():
    parser = argparse.ArgumentParser(description="Convert OC20 to MinCatFlow LMDB")
    parser.add_argument("--lmdb-path", type=str, required=True, help="OC20 IS2RE LMDB path")
    parser.add_argument("--mapping-path", type=str, required=True, help="oc20_data_mapping.pkl path")
    parser.add_argument("--adsorbates-path", type=str, default="resources/adsorbates.pkl",
                        help="adsorbates.pkl path from fairchem")
    parser.add_argument("--output-path", type=str, required=True, help="Output LMDB path")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--chunksize", type=int, default=50, help="Chunk size for multiprocessing")
    args = parser.parse_args()

    num_workers = args.num_workers or cpu_count()

    # Count total samples in LMDB
    db = lmdb.open(args.lmdb_path, subdir=False, readonly=True, lock=False)
    with db.begin() as txn:
        total_samples = txn.stat()['entries']
    db.close()

    end_index = args.end if args.end is not None else total_samples
    indices = list(range(args.start, end_index))

    print(f"Processing {len(indices)} samples (index {args.start} to {end_index-1})")
    print(f"Using {num_workers} workers")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Open output LMDB
    out_db = lmdb.open(
        args.output_path,
        map_size=1099511627776 * 2,  # 2TB
        subdir=False,
        meminit=False,
        map_async=True,
    )

    stats = {'success': 0, 'failed': 0, 'errors': {}}
    out_idx = 0

    # Process in parallel
    with Pool(num_workers, initializer=init_worker, initargs=(args.mapping_path, args.lmdb_path, args.adsorbates_path)) as pool:
        for result in tqdm(pool.imap(process_single, indices, chunksize=args.chunksize),
                          total=len(indices), desc="Processing"):
            if result['status'] == 'success':
                # Save to LMDB
                with out_db.begin(write=True) as txn:
                    txn.put(f"{out_idx}".encode("ascii"),
                           pickle.dumps(result['data'], protocol=-1))
                out_idx += 1
                stats['success'] += 1
            else:
                stats['failed'] += 1
                stats['errors'][result['status']] = stats['errors'].get(result['status'], 0) + 1

    # Save length
    with out_db.begin(write=True) as txn:
        txn.put("length".encode("ascii"), pickle.dumps(out_idx, protocol=-1))

    out_db.sync()
    out_db.close()

    # Save stats
    stats_path = args.output_path.replace('.lmdb', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nResults saved to: {args.output_path}")
    print(f"Successful: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    if stats['errors']:
        print("Error breakdown:")
        for err, count in sorted(stats['errors'].items(), key=lambda x: -x[1]):
            print(f"  {err}: {count}")


if __name__ == "__main__":
    main()
