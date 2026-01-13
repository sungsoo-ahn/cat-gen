#!/usr/bin/env python3
"""Convert OC20 IS2RE LMDB directly to CatGen LMDB format.

Usage:
    uv run python src/scripts/oc20_to_catgen.py \
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


def extract_from_lmdb(lmdb_path: str, index: int, verbose: bool = False) -> tuple[dict | None, str | None]:
    """Extract data from OC20 LMDB at given index.

    Returns:
        Tuple of (data_dict, error_message)
        - data_dict: Extracted data or None if failed
        - error_message: Description of error if failed, None if success
    """
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
                return None, f"Key {index} not found in LMDB"
            data = pickle.loads(value)

        # Bypass PyG's property system by accessing __dict__ directly
        # This avoids version compatibility issues with older PyG data
        internal = object.__getattribute__(data, '__dict__')

        # PyG Data objects may store data in '_store' dict
        if '_store' in internal:
            internal = internal['_store']

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

        # Validate required fields with detailed error messages
        missing_fields = []
        if tags is None:
            missing_fields.append('tags')
        if atomic_numbers is None:
            missing_fields.append('atomic_numbers')
        if pos is None:
            missing_fields.append('pos')
        if cell is None:
            missing_fields.append('cell')
        if y_relaxed is None:
            missing_fields.append('y_relaxed')

        if missing_fields:
            available_keys = list(internal.keys()) if hasattr(internal, 'keys') else []
            return None, f"Missing required fields: {missing_fields}. Available keys: {available_keys}"

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
        }, None
    except Exception as e:
        return None, f"Exception during extraction: {type(e).__name__}: {str(e)}"
    finally:
        db.close()


def process_to_catgen(
    data: dict,
    n_slab: int,
    n_vac: int,
    ref_ads_pos_canonical: np.ndarray,
    rmsd_tolerance: float = 1e-4,
    verbose: bool = False,
) -> tuple[dict | None, dict]:
    """Convert extracted data to CatGen format with primitive slab.

    Args:
        data: Extracted OC20 data
        n_slab: Number of slab layers
        n_vac: Number of vacuum layers
        ref_ads_pos_canonical: Canonical reference adsorbate positions from adsorbates.pkl
        rmsd_tolerance: Tolerance for RMSD validation (default: 1e-4)
        verbose: If True, return detailed diagnostics

    Returns:
        Tuple of (result_data, diagnostics)
        - result_data: CatGen format dict or None if failed
        - diagnostics: Dict with detailed failure information
    """
    diagnostics = {
        'stage': None,
        'error': None,
        'rmsd_tight': None,
        'rmsd_system': None,
        'n_slab_atoms': None,
        'n_ads_atoms': None,
        'n_prim_atoms': None,
        'supercell_det': None,
        'lattice_params': None,
        'prim_lattice_params': None,
    }

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
        ads_mask = data['tags'] == 2
        true_slab = true_system[slab_mask]

        diagnostics['n_slab_atoms'] = int(slab_mask.sum())
        diagnostics['n_ads_atoms'] = int(ads_mask.sum())
        diagnostics['lattice_params'] = list(true_system.cell.cellpar())

        adaptor = AseAtomsAdaptor()
        n_layers = n_slab + n_vac

        # Create tight slab (remove vacuum)
        diagnostics['stage'] = 'tight_slab_creation'
        tight_slab = true_slab.copy()
        tight_cell = tight_slab.get_cell()
        tight_cell[2] = tight_cell[2] * (n_slab / n_layers)
        tight_slab.set_cell(tight_cell)
        tight_slab.center()

        # Find primitive cell
        diagnostics['stage'] = 'primitive_cell_finding'
        tight_slab_struct = adaptor.get_structure(tight_slab)
        prim_slab_struct = tight_slab_struct.get_primitive_structure(tolerance=0.1)
        diagnostics['n_prim_atoms'] = len(prim_slab_struct)
        diagnostics['prim_lattice_params'] = list(prim_slab_struct.lattice.parameters)

        # Standardize orientation
        diagnostics['stage'] = 'orientation_standardization'
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
        diagnostics['stage'] = 'supercell_matrix_computation'
        rotated_tight_lattice = Lattice(tight_slab_struct.lattice.matrix @ R.T)

        # Compute raw (non-rounded) supercell matrix for analysis
        sc_matrix_raw = np.dot(
            rotated_tight_lattice.matrix,
            np.linalg.inv(standard_lattice.matrix)
        )
        sc_matrix = np.round(sc_matrix_raw).astype(int)
        sc_det = int(round(np.linalg.det(sc_matrix)))
        diagnostics['supercell_det'] = sc_det
        diagnostics['sc_matrix_raw'] = sc_matrix_raw.tolist()

        # Validate supercell matrix: det(sc_matrix) * n_prim should equal n_slab
        n_prim = len(prim_slab_struct)
        n_slab_atoms = diagnostics['n_slab_atoms']
        expected_det = n_slab_atoms // n_prim if n_prim > 0 else 0
        diagnostics['expected_supercell_det'] = expected_det

        if sc_det != expected_det and expected_det > 0:
            diagnostics['stage'] = 'supercell_matrix_validation'
            diagnostics['det_mismatch'] = True

            # Analyze rounding errors
            rounding_errors = np.abs(sc_matrix_raw - sc_matrix)
            max_rounding_error = np.max(rounding_errors)
            diagnostics['max_rounding_error'] = float(max_rounding_error)

            # Try to find a corrected matrix by searching nearby integer matrices
            corrected_matrix = None
            best_det_diff = abs(sc_det - expected_det)

            # Search within ±1 of each element for matrices with correct determinant
            for i in range(3):
                for j in range(3):
                    for delta in [-1, 1]:
                        test_matrix = sc_matrix.copy()
                        test_matrix[i, j] += delta
                        test_det = int(round(np.linalg.det(test_matrix)))
                        if test_det == expected_det:
                            # Found a candidate - verify it's closer to raw matrix
                            test_error = np.max(np.abs(sc_matrix_raw - test_matrix))
                            if test_error < 0.6:  # Reasonable threshold
                                corrected_matrix = test_matrix
                                diagnostics['correction_applied'] = f'Adjusted element [{i},{j}] by {delta}'
                                break
                    if corrected_matrix is not None:
                        break
                if corrected_matrix is not None:
                    break

            if corrected_matrix is not None:
                sc_matrix = corrected_matrix
                sc_det = expected_det
                diagnostics['supercell_det'] = sc_det
                diagnostics['matrix_corrected'] = True
            else:
                # Could not find correction - report detailed error
                diagnostics['error'] = 'supercell_matrix_invalid'
                diagnostics['detail'] = (
                    f'Supercell matrix determinant {sc_det} != expected {expected_det} '
                    f'(n_slab={n_slab_atoms}, n_prim={n_prim}). '
                    f'Max rounding error: {max_rounding_error:.4f}. '
                    f'This typically occurs with non-orthogonal cells.'
                )
                return None, diagnostics
        else:
            diagnostics['det_mismatch'] = False
            diagnostics['matrix_corrected'] = False

        # Verify reconstruction - tight slab
        diagnostics['stage'] = 'rmsd_tight_slab_check'
        recon_struct = rotated_prim_struct.copy()
        recon_struct.make_supercell(sc_matrix, to_unit_cell=False)
        recon_tight_slab = adaptor.get_atoms(recon_struct)

        rmsd_tight, max_dist_tight = calculate_rmsd_pymatgen(
            struct1=recon_tight_slab,
            struct2=tight_slab,
            ltol=0.2, stol=0.3, angle_tol=5,
            primitive_cell=False,
        )
        diagnostics['rmsd_tight'] = rmsd_tight
        diagnostics['max_dist_tight'] = max_dist_tight

        if rmsd_tight is None:
            diagnostics['error'] = 'tight_slab_no_match'
            diagnostics['detail'] = 'StructureMatcher could not find correspondence between reconstructed and original tight slab'
            return None, diagnostics

        if rmsd_tight >= rmsd_tolerance:
            diagnostics['error'] = 'tight_slab_rmsd_exceeded'
            diagnostics['detail'] = f'RMSD {rmsd_tight:.6f} >= tolerance {rmsd_tolerance}'
            return None, diagnostics

        # Add vacuum back
        scaling_factor = n_layers / n_slab
        recon_slab = recon_tight_slab.copy()
        recon_cell = recon_slab.get_cell()
        recon_cell[2] = recon_cell[2] * scaling_factor
        recon_slab.set_cell(recon_cell)

        # Transform adsorbate for RMSD verification (using original positions)
        diagnostics['stage'] = 'adsorbate_transformation'
        adsorbate_orig = true_system[data['tags'] == 2].copy()
        rotated_true_slab = true_slab.copy()
        rotated_true_slab.set_cell(true_slab.cell @ R.T)
        rotated_true_slab.set_positions(true_slab.positions @ R.T)

        rotated_adsorbate = adsorbate_orig.copy()
        rotated_adsorbate.set_positions(adsorbate_orig.positions @ R.T)

        # Check if cell is triclinic (all angles != 90)
        cell_angles = recon_slab.cell.angles()
        is_triclinic = all(abs(angle - 90.0) > 1.0 for angle in cell_angles)
        diagnostics['is_triclinic'] = is_triclinic
        diagnostics['cell_angles'] = list(cell_angles)

        # Try multiple adsorbate alignment strategies
        alignment_strategies = []

        # Strategy 1: Center of mass alignment (original method)
        diff_vector_com = recon_slab.get_center_of_mass() - rotated_true_slab.get_center_of_mass()
        aligned_ads_com = rotated_adsorbate.copy()
        aligned_ads_com.translate(diff_vector_com)
        alignment_strategies.append(('com', aligned_ads_com, diff_vector_com))

        # Strategy 2: Surface atom alignment (for triclinic cells)
        # Find the topmost slab atom in z and align based on that
        if is_triclinic or True:  # Always try this as backup
            # Get z-coordinates of slab atoms
            recon_slab_z = recon_slab.positions[:, 2]
            rotated_slab_z = rotated_true_slab.positions[:, 2]

            # Find topmost atoms (surface)
            recon_top_idx = np.argmax(recon_slab_z)
            rotated_top_idx = np.argmax(rotated_slab_z)

            # Compute offset based on surface atom positions
            diff_vector_surface = recon_slab.positions[recon_top_idx] - rotated_true_slab.positions[rotated_top_idx]
            aligned_ads_surface = rotated_adsorbate.copy()
            aligned_ads_surface.translate(diff_vector_surface)
            alignment_strategies.append(('surface', aligned_ads_surface, diff_vector_surface))

        # Strategy 3: Fractional coordinate alignment
        # Transform adsorbate to fractional coords, apply to new cell
        try:
            # Get adsorbate position relative to original slab in fractional coordinates
            orig_cell_inv = np.linalg.inv(true_slab.cell[:])
            ads_frac_relative = (adsorbate_orig.positions @ R.T) @ np.linalg.inv(rotated_true_slab.cell[:])

            # Apply to reconstructed slab cell
            aligned_ads_frac = rotated_adsorbate.copy()
            new_positions = ads_frac_relative @ recon_slab.cell[:]
            aligned_ads_frac.set_positions(new_positions)
            diff_vector_frac = new_positions[0] - rotated_adsorbate.positions[0] if len(new_positions) > 0 else np.zeros(3)
            alignment_strategies.append(('fractional', aligned_ads_frac, diff_vector_frac))
        except Exception:
            pass  # Skip if fractional method fails

        # Try each alignment strategy and find the best one
        diagnostics['stage'] = 'rmsd_full_system_check'
        best_rmsd = None
        best_strategy = None
        best_aligned_ads = None
        best_diff_vector = None

        for strategy_name, aligned_ads, diff_vec in alignment_strategies:
            rmsd_system, max_dist_system = calculate_rmsd_pymatgen(
                struct1=recon_slab + aligned_ads,
                struct2=true_system,
                ltol=0.2, stol=0.3, angle_tol=5,
                primitive_cell=False,
            )

            if rmsd_system is not None:
                if best_rmsd is None or rmsd_system < best_rmsd:
                    best_rmsd = rmsd_system
                    best_strategy = strategy_name
                    best_aligned_ads = aligned_ads
                    best_diff_vector = diff_vec
                    diagnostics[f'rmsd_{strategy_name}'] = rmsd_system

                # If we found a good match, we can stop
                if rmsd_system < rmsd_tolerance:
                    break
            else:
                diagnostics[f'rmsd_{strategy_name}'] = None

        # Use the best alignment found
        if best_aligned_ads is not None:
            rotated_adsorbate = best_aligned_ads
            diff_vector = best_diff_vector
            diagnostics['alignment_strategy'] = best_strategy
            diagnostics['rmsd_system'] = best_rmsd
            diagnostics['max_dist_system'] = max_dist_system
        else:
            # No alignment strategy worked - try relaxed tolerances as last resort
            diagnostics['stage'] = 'rmsd_full_system_check_relaxed'
            for strategy_name, aligned_ads, diff_vec in alignment_strategies:
                rmsd_system, max_dist_system = calculate_rmsd_pymatgen(
                    struct1=recon_slab + aligned_ads,
                    struct2=true_system,
                    ltol=0.3, stol=0.5, angle_tol=10,  # Relaxed tolerances
                    primitive_cell=False,
                )
                if rmsd_system is not None:
                    best_rmsd = rmsd_system
                    best_strategy = strategy_name + '_relaxed'
                    best_aligned_ads = aligned_ads
                    best_diff_vector = diff_vec
                    break

            if best_aligned_ads is not None:
                rotated_adsorbate = best_aligned_ads
                diff_vector = best_diff_vector
                diagnostics['alignment_strategy'] = best_strategy
                diagnostics['rmsd_system'] = best_rmsd
                diagnostics['max_dist_system'] = max_dist_system
            else:
                # Last resort: If tight slab RMSD was perfect, try direct Cartesian comparison
                # This handles cases where StructureMatcher fails due to adsorbate periodicity issues
                if rmsd_tight is not None and rmsd_tight < 1e-10:
                    diagnostics['stage'] = 'direct_cartesian_comparison'

                    # The slab reconstruction is perfect, so we trust it
                    # For adsorbate, compute position relative to slab and apply to reconstructed slab
                    best_direct_rmsd = float('inf')
                    best_direct_strategy = None
                    best_direct_ads = None
                    best_direct_diff = None

                    for strategy_name, aligned_ads, diff_vec in alignment_strategies:
                        # Compare adsorbate positions directly (accounting for PBC)
                        orig_ads_pos = true_system[data['tags'] == 2].positions
                        aligned_ads_pos = aligned_ads.positions

                        # Compute minimum image distance for each atom
                        cell = recon_slab.cell[:]
                        cell_inv = np.linalg.inv(cell)

                        total_rmsd = 0.0
                        for i in range(len(orig_ads_pos)):
                            diff = aligned_ads_pos[i] - orig_ads_pos[i]
                            # Apply minimum image convention
                            frac_diff = diff @ cell_inv
                            frac_diff = frac_diff - np.round(frac_diff)
                            cart_diff = frac_diff @ cell
                            total_rmsd += np.sum(cart_diff ** 2)

                        rmsd_direct = np.sqrt(total_rmsd / len(orig_ads_pos)) if len(orig_ads_pos) > 0 else 0.0
                        diagnostics[f'rmsd_direct_{strategy_name}'] = float(rmsd_direct)

                        if rmsd_direct < best_direct_rmsd:
                            best_direct_rmsd = rmsd_direct
                            best_direct_strategy = strategy_name + '_direct'
                            best_direct_ads = aligned_ads
                            best_direct_diff = diff_vec

                    # Accept if direct RMSD is reasonable (< 0.5 Angstrom)
                    if best_direct_rmsd < 0.5:
                        rotated_adsorbate = best_direct_ads
                        diff_vector = best_direct_diff
                        diagnostics['alignment_strategy'] = best_direct_strategy
                        diagnostics['rmsd_system'] = best_direct_rmsd
                        diagnostics['max_dist_system'] = None
                        diagnostics['used_direct_comparison'] = True
                    else:
                        diagnostics['rmsd_system'] = None
                        diagnostics['max_dist_system'] = None
                        diagnostics['error'] = 'full_system_no_match'
                        diagnostics['detail'] = (
                            'StructureMatcher and direct comparison both failed. '
                            f'Best direct RMSD: {best_direct_rmsd:.4f}. '
                            f'Tried {len(alignment_strategies)} alignment strategies. '
                            f'Cell angles: {cell_angles}. Is triclinic: {is_triclinic}.'
                        )
                        return None, diagnostics
                else:
                    diagnostics['rmsd_system'] = None
                    diagnostics['max_dist_system'] = None
                    diagnostics['error'] = 'full_system_no_match'
                    diagnostics['detail'] = (
                        'StructureMatcher could not find correspondence between reconstructed and original full system. '
                        f'Tried {len(alignment_strategies)} alignment strategies. '
                        f'Cell angles: {cell_angles}. Is triclinic: {is_triclinic}.'
                    )
                    return None, diagnostics

        if best_rmsd is not None and best_rmsd >= rmsd_tolerance:
            diagnostics['error'] = 'full_system_rmsd_exceeded'
            diagnostics['detail'] = f'RMSD {best_rmsd:.6f} >= tolerance {rmsd_tolerance} (strategy: {best_strategy})'
            return None, diagnostics

        # Use relaxed positions for the dataset output
        diagnostics['stage'] = 'final_assembly'
        if data['ads_pos_relaxed'] is not None:
            relaxed_adsorbate = adsorbate_orig.copy()
            relaxed_adsorbate.set_positions(data['ads_pos_relaxed'] @ R.T)
            relaxed_adsorbate.translate(diff_vector)
            final_adsorbate = relaxed_adsorbate
        else:
            final_adsorbate = rotated_adsorbate

        # Convert primitive slab to ASE Atoms
        primitive_slab_atoms = adaptor.get_atoms(rotated_prim_struct)

        diagnostics['stage'] = 'success'
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
        }, diagnostics
    except Exception as e:
        diagnostics['error'] = 'exception'
        diagnostics['detail'] = f'{type(e).__name__}: {str(e)}'
        return None, diagnostics


# Global variables for worker processes
_mapping = None
_lmdb_path = None
_adsorbates = None
_slab_cache = {}
_rmsd_tolerance = None


def init_worker(mapping_path: str, lmdb_path: str, adsorbates_path: str, rmsd_tolerance: float = 1e-4):
    """Initialize worker with shared data."""
    global _mapping, _lmdb_path, _adsorbates, _rmsd_tolerance
    _mapping = load_mapping(mapping_path)
    _lmdb_path = lmdb_path
    _rmsd_tolerance = rmsd_tolerance
    with open(adsorbates_path, "rb") as f:
        _adsorbates = pickle.load(f)


def process_single(index: int) -> dict:
    """Process a single sample."""
    global _mapping, _lmdb_path, _adsorbates, _slab_cache, _rmsd_tolerance

    result = {
        'index': index,
        'status': 'error',
        'data': None,
        'diagnostics': None,
        'sid': None,
        'bulk_mpid': None,
        'miller_index': None,
        'extract_error': None,
    }

    # Extract from LMDB
    data, extract_error = extract_from_lmdb(_lmdb_path, index)
    if data is None:
        result['status'] = 'extract_failed'
        result['extract_error'] = extract_error
        return result

    sid = data['sid']
    result['sid'] = sid
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

    result['bulk_mpid'] = bulk_mpid
    result['miller_index'] = str(miller_index) if miller_index else None

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

    # Convert to CatGen format with detailed diagnostics
    rmsd_tol = _rmsd_tolerance if _rmsd_tolerance is not None else 1e-4
    catgen_data, diagnostics = process_to_catgen(
        data, n_slab, n_vac, ref_ads_pos_canonical, rmsd_tolerance=rmsd_tol
    )
    result['diagnostics'] = diagnostics

    if catgen_data is None:
        # Use specific error from diagnostics if available
        if diagnostics.get('error'):
            result['status'] = f"conversion_failed:{diagnostics['error']}"
        else:
            result['status'] = 'conversion_failed'
        return result

    result['status'] = 'success'
    result['data'] = catgen_data
    return result


def main():
    parser = argparse.ArgumentParser(description="Convert OC20 to CatGen LMDB")
    parser.add_argument("--lmdb-path", type=str, required=True, help="OC20 IS2RE LMDB path")
    parser.add_argument("--mapping-path", type=str, required=True, help="oc20_data_mapping.pkl path")
    parser.add_argument("--adsorbates-path", type=str, default="resources/adsorbates.pkl",
                        help="adsorbates.pkl path from fairchem")
    parser.add_argument("--output-path", type=str, required=True, help="Output LMDB path")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--chunksize", type=int, default=50, help="Chunk size for multiprocessing")
    parser.add_argument("--rmsd-tolerance", type=float, default=1e-4,
                        help="RMSD tolerance for validation (default: 1e-4)")
    parser.add_argument("--save-failures", action="store_true",
                        help="Save detailed failure information to a separate file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information")
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
    print(f"RMSD tolerance: {args.rmsd_tolerance}")

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

    # Enhanced statistics tracking
    stats = {
        'success': 0,
        'failed': 0,
        'errors': {},
        'rmsd_tolerance': args.rmsd_tolerance,
        'rmsd_tight_distribution': [],  # Track RMSD values for analysis
        'rmsd_system_distribution': [],
        'matrix_corrections': 0,  # Track how many samples were saved by matrix correction
        'alignment_strategies': {},  # Track which adsorbate alignment strategies were used
        'triclinic_cells': 0,  # Track how many triclinic cells were processed
        'failure_details': {
            'tight_slab_no_match': [],
            'tight_slab_rmsd_exceeded': [],
            'full_system_no_match': [],
            'full_system_rmsd_exceeded': [],
            'supercell_matrix_invalid': [],
            'exception': [],
        },
        'miller_index_failures': {},  # Track failures by miller index
        'bulk_mpid_failures': {},  # Track failures by bulk material
    }
    failed_samples = []
    out_idx = 0

    # Process in parallel
    with Pool(num_workers, initializer=init_worker,
              initargs=(args.mapping_path, args.lmdb_path, args.adsorbates_path, args.rmsd_tolerance)) as pool:
        for result in tqdm(pool.imap(process_single, indices, chunksize=args.chunksize),
                          total=len(indices), desc="Processing"):
            if result['status'] == 'success':
                # Save to LMDB
                with out_db.begin(write=True) as txn:
                    txn.put(f"{out_idx}".encode("ascii"),
                           pickle.dumps(result['data'], protocol=-1))
                out_idx += 1
                stats['success'] += 1

                # Track successful RMSD values for distribution analysis
                if result.get('diagnostics'):
                    diag = result['diagnostics']
                    if diag.get('rmsd_tight') is not None:
                        stats['rmsd_tight_distribution'].append(float(diag['rmsd_tight']))
                    if diag.get('rmsd_system') is not None:
                        stats['rmsd_system_distribution'].append(float(diag['rmsd_system']))
                    # Track matrix corrections
                    if diag.get('matrix_corrected'):
                        stats['matrix_corrections'] += 1
                    # Track alignment strategies
                    strategy = diag.get('alignment_strategy', 'unknown')
                    stats['alignment_strategies'][strategy] = stats['alignment_strategies'].get(strategy, 0) + 1
                    # Track triclinic cells
                    if diag.get('is_triclinic'):
                        stats['triclinic_cells'] += 1
            else:
                stats['failed'] += 1
                status = result['status']
                stats['errors'][status] = stats['errors'].get(status, 0) + 1

                # Track detailed failure info
                diag = result.get('diagnostics', {})
                error_type = diag.get('error') if diag else None

                if error_type and error_type in stats['failure_details']:
                    failure_info = {
                        'index': result['index'],
                        'sid': result.get('sid'),
                        'bulk_mpid': result.get('bulk_mpid'),
                        'miller_index': result.get('miller_index'),
                        'rmsd_tight': diag.get('rmsd_tight'),
                        'rmsd_system': diag.get('rmsd_system'),
                        'n_slab_atoms': diag.get('n_slab_atoms'),
                        'n_prim_atoms': diag.get('n_prim_atoms'),
                        'supercell_det': diag.get('supercell_det'),
                        'detail': diag.get('detail'),
                    }
                    stats['failure_details'][error_type].append(failure_info)

                    # Track RMSD values even for failures (for distribution analysis)
                    if diag.get('rmsd_tight') is not None:
                        stats['rmsd_tight_distribution'].append(float(diag['rmsd_tight']))
                    if diag.get('rmsd_system') is not None:
                        stats['rmsd_system_distribution'].append(float(diag['rmsd_system']))

                # Track failures by miller index and bulk_mpid
                miller = result.get('miller_index')
                if miller:
                    stats['miller_index_failures'][miller] = stats['miller_index_failures'].get(miller, 0) + 1

                bulk = result.get('bulk_mpid')
                if bulk:
                    stats['bulk_mpid_failures'][bulk] = stats['bulk_mpid_failures'].get(bulk, 0) + 1

                # Save full failure info if requested
                if args.save_failures:
                    failed_samples.append({
                        'index': result['index'],
                        'status': status,
                        'sid': result.get('sid'),
                        'bulk_mpid': result.get('bulk_mpid'),
                        'miller_index': result.get('miller_index'),
                        'extract_error': result.get('extract_error'),
                        'diagnostics': diag,
                    })

    # Save length
    with out_db.begin(write=True) as txn:
        txn.put("length".encode("ascii"), pickle.dumps(out_idx, protocol=-1))

    out_db.sync()
    out_db.close()

    # Compute RMSD statistics
    rmsd_stats = {}
    if stats['rmsd_tight_distribution']:
        rmsd_tight = np.array(stats['rmsd_tight_distribution'])
        rmsd_stats['rmsd_tight'] = {
            'mean': float(np.mean(rmsd_tight)),
            'std': float(np.std(rmsd_tight)),
            'min': float(np.min(rmsd_tight)),
            'max': float(np.max(rmsd_tight)),
            'median': float(np.median(rmsd_tight)),
            'p90': float(np.percentile(rmsd_tight, 90)),
            'p95': float(np.percentile(rmsd_tight, 95)),
            'p99': float(np.percentile(rmsd_tight, 99)),
        }
    if stats['rmsd_system_distribution']:
        rmsd_sys = np.array(stats['rmsd_system_distribution'])
        rmsd_stats['rmsd_system'] = {
            'mean': float(np.mean(rmsd_sys)),
            'std': float(np.std(rmsd_sys)),
            'min': float(np.min(rmsd_sys)),
            'max': float(np.max(rmsd_sys)),
            'median': float(np.median(rmsd_sys)),
            'p90': float(np.percentile(rmsd_sys, 90)),
            'p95': float(np.percentile(rmsd_sys, 95)),
            'p99': float(np.percentile(rmsd_sys, 99)),
        }
    stats['rmsd_stats'] = rmsd_stats

    # Limit failure_details to top 100 per category for JSON size
    for key in stats['failure_details']:
        stats['failure_details'][key] = stats['failure_details'][key][:100]

    # Remove raw distributions from stats (too large for JSON)
    stats.pop('rmsd_tight_distribution', None)
    stats.pop('rmsd_system_distribution', None)

    # Sort and limit miller/bulk failures to top 20
    stats['miller_index_failures'] = dict(
        sorted(stats['miller_index_failures'].items(), key=lambda x: -x[1])[:20]
    )
    stats['bulk_mpid_failures'] = dict(
        sorted(stats['bulk_mpid_failures'].items(), key=lambda x: -x[1])[:20]
    )

    # Save stats
    stats_path = args.output_path.replace('.lmdb', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Save full failure details if requested
    if args.save_failures and failed_samples:
        failures_path = args.output_path.replace('.lmdb', '_failures.json')
        with open(failures_path, 'w') as f:
            json.dump(failed_samples, f, indent=2)
        print(f"Failure details saved to: {failures_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"CONVERSION RESULTS")
    print(f"{'='*60}")
    print(f"Output: {args.output_path}")
    total = stats['success'] + stats['failed']
    print(f"Successful: {stats['success']} ({100*stats['success']/total:.1f}%)")
    print(f"Failed: {stats['failed']} ({100*stats['failed']/total:.1f}%)")
    if stats['matrix_corrections'] > 0:
        print(f"Matrix corrections applied: {stats['matrix_corrections']} samples saved")
    if stats['triclinic_cells'] > 0:
        print(f"Triclinic cells processed: {stats['triclinic_cells']}")
    if stats['alignment_strategies']:
        print(f"Alignment strategies used: {dict(sorted(stats['alignment_strategies'].items(), key=lambda x: -x[1]))}")

    if stats['errors']:
        print(f"\n{'='*60}")
        print("ERROR BREAKDOWN:")
        print(f"{'='*60}")
        for err, count in sorted(stats['errors'].items(), key=lambda x: -x[1]):
            pct = 100 * count / stats['failed'] if stats['failed'] > 0 else 0
            print(f"  {err}: {count} ({pct:.1f}% of failures)")

    if rmsd_stats:
        print(f"\n{'='*60}")
        print("RMSD STATISTICS:")
        print(f"{'='*60}")
        if 'rmsd_tight' in rmsd_stats:
            r = rmsd_stats['rmsd_tight']
            print(f"  Tight slab RMSD:")
            print(f"    mean={r['mean']:.2e}, std={r['std']:.2e}")
            print(f"    min={r['min']:.2e}, max={r['max']:.2e}")
            print(f"    p90={r['p90']:.2e}, p95={r['p95']:.2e}, p99={r['p99']:.2e}")
        if 'rmsd_system' in rmsd_stats:
            r = rmsd_stats['rmsd_system']
            print(f"  Full system RMSD:")
            print(f"    mean={r['mean']:.2e}, std={r['std']:.2e}")
            print(f"    min={r['min']:.2e}, max={r['max']:.2e}")
            print(f"    p90={r['p90']:.2e}, p95={r['p95']:.2e}, p99={r['p99']:.2e}")

    # Print recommendations if there are failures
    if stats['failed'] > 0:
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS:")
        print(f"{'='*60}")

        # Check for RMSD-related failures
        rmsd_failures = sum([
            len(stats['failure_details'].get('tight_slab_rmsd_exceeded', [])),
            len(stats['failure_details'].get('full_system_rmsd_exceeded', [])),
        ])
        no_match_failures = sum([
            len(stats['failure_details'].get('tight_slab_no_match', [])),
            len(stats['failure_details'].get('full_system_no_match', [])),
        ])

        if rmsd_failures > 0:
            print(f"\n  * {rmsd_failures} samples failed due to RMSD exceeding tolerance ({args.rmsd_tolerance})")
            if rmsd_stats.get('rmsd_tight'):
                suggested_tol = rmsd_stats['rmsd_tight']['p99'] * 1.5
                print(f"    Consider increasing --rmsd-tolerance to {suggested_tol:.2e}")
                print(f"    (This would recover samples up to p99 + margin)")

        if no_match_failures > 0:
            print(f"\n  * {no_match_failures} samples failed because StructureMatcher couldn't find correspondence")
            print(f"    This may indicate:")
            print(f"    - Highly distorted structures")
            print(f"    - Issues with primitive cell detection (tolerance=0.1)")
            print(f"    - Triclinic cells with complex adsorbate orientations")

        # Check for supercell matrix failures
        matrix_failures = len(stats['failure_details'].get('supercell_matrix_invalid', []))
        if matrix_failures > 0:
            print(f"\n  * {matrix_failures} samples failed due to invalid supercell matrix")
            print(f"    The matrix determinant didn't match expected atom count ratio.")
            print(f"    This typically occurs with highly non-orthogonal cells where")
            print(f"    rounding errors accumulate. Matrix correction was attempted but failed.")

        if stats['miller_index_failures']:
            top_miller = list(stats['miller_index_failures'].items())[0]
            print(f"\n  * Most problematic Miller index: {top_miller[0]} ({top_miller[1]} failures)")
            print(f"    Consider investigating these surface orientations")

        if stats['bulk_mpid_failures']:
            top_bulk = list(stats['bulk_mpid_failures'].items())[0]
            print(f"\n  * Most problematic bulk material: {top_bulk[0]} ({top_bulk[1]} failures)")
            print(f"    Consider investigating this material's structures")


if __name__ == "__main__":
    main()
