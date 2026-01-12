"""Conversion functions between lattice parameters and virtual atom coordinates.

This module provides functions to convert between:
- Lattice parameters (a, b, c, alpha, beta, gamma) and lattice vectors [3, 3]
- Virtual atom coordinates and lattice/supercell representations

The virtual atoms architecture uses 6 virtual atoms:
- 3 for primitive lattice vectors (a, b, c)
- 3 for supercell lattice vectors (a', b', c')
"""

import numpy as np
import torch
from torch import Tensor
from pymatgen.core import Lattice


def lattice_params_to_vectors_single(lattice_params: np.ndarray) -> np.ndarray:
    """Convert lattice parameters to lattice vectors (row vectors) for a single sample.

    Args:
        lattice_params: [6] array of (a, b, c, alpha, beta, gamma)
                       lengths in Angstrom, angles in degrees

    Returns:
        lattice_vectors: [3, 3] array where each row is a lattice vector
    """
    a, b, c, alpha, beta, gamma = lattice_params
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    # lattice.matrix has rows as lattice vectors in pymatgen
    return lattice.matrix.copy()


def lattice_vectors_to_params_single(lattice_vectors: np.ndarray) -> np.ndarray:
    """Convert lattice vectors to lattice parameters for a single sample.

    Args:
        lattice_vectors: [3, 3] array where each row is a lattice vector

    Returns:
        lattice_params: [6] array of (a, b, c, alpha, beta, gamma)
    """
    lattice = Lattice(lattice_vectors)
    return np.array(lattice.parameters)


def compute_supercell_matrix_from_vectors(
    prim_vectors: np.ndarray,
    supercell_vectors: np.ndarray,
) -> np.ndarray:
    """Recover supercell matrix from primitive and supercell lattice vectors.

    The relationship is: supercell_vectors = supercell_matrix @ prim_vectors
    So: supercell_matrix = supercell_vectors @ prim_vectors^(-1)

    Args:
        prim_vectors: [3, 3] array of primitive lattice vectors (row vectors)
        supercell_vectors: [3, 3] array of supercell lattice vectors (row vectors)

    Returns:
        supercell_matrix: [3, 3] transformation matrix
    """
    # supercell_vectors = supercell_matrix @ prim_vectors
    # supercell_matrix = supercell_vectors @ inv(prim_vectors)
    return supercell_vectors @ np.linalg.inv(prim_vectors)


def lattice_params_to_vectors(lattice_params: Tensor) -> Tensor:
    """Convert lattice parameters to lattice vectors (batched).

    Args:
        lattice_params: [B, 6] or [6] tensor of (a, b, c, alpha, beta, gamma)

    Returns:
        lattice_vectors: [B, 3, 3] or [3, 3] tensor where each row is a lattice vector
    """
    single_sample = lattice_params.ndim == 1
    if single_sample:
        lattice_params = lattice_params.unsqueeze(0)

    B = lattice_params.shape[0]
    device = lattice_params.device
    dtype = lattice_params.dtype

    vectors = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    for i in range(B):
        params_np = lattice_params[i].detach().cpu().numpy()
        vectors_np = lattice_params_to_vectors_single(params_np)
        vectors[i] = torch.from_numpy(vectors_np).to(device=device, dtype=dtype)

    if single_sample:
        return vectors[0]
    return vectors


def lattice_vectors_to_params(lattice_vectors: Tensor) -> Tensor:
    """Convert lattice vectors to lattice parameters (batched).

    Args:
        lattice_vectors: [B, 3, 3] or [3, 3] tensor where each row is a lattice vector

    Returns:
        lattice_params: [B, 6] or [6] tensor of (a, b, c, alpha, beta, gamma)
    """
    single_sample = lattice_vectors.ndim == 2
    if single_sample:
        lattice_vectors = lattice_vectors.unsqueeze(0)

    B = lattice_vectors.shape[0]
    device = lattice_vectors.device
    dtype = lattice_vectors.dtype

    params = torch.zeros(B, 6, device=device, dtype=dtype)
    for i in range(B):
        vectors_np = lattice_vectors[i].detach().cpu().numpy()
        params_np = lattice_vectors_to_params_single(vectors_np)
        params[i] = torch.from_numpy(params_np).to(device=device, dtype=dtype)

    if single_sample:
        return params[0]
    return params


def compute_virtual_coords(
    lattice_params: Tensor,
    supercell_matrix: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute primitive and supercell virtual atom coordinates.

    Args:
        lattice_params: [B, 6] or [6] tensor of lattice parameters
        supercell_matrix: [B, 3, 3] or [3, 3] tensor of supercell transformation matrices

    Returns:
        prim_virtual_coords: [B, 3, 3] or [3, 3] primitive lattice vectors (row vectors)
        supercell_virtual_coords: [B, 3, 3] or [3, 3] supercell lattice vectors (row vectors)
    """
    single_sample = lattice_params.ndim == 1
    if single_sample:
        lattice_params = lattice_params.unsqueeze(0)
        supercell_matrix = supercell_matrix.unsqueeze(0)

    # Convert lattice params to primitive vectors
    prim_virtual_coords = lattice_params_to_vectors(lattice_params)

    # Compute supercell vectors: supercell_vectors = supercell_matrix @ prim_vectors
    # Using batched matrix multiplication
    supercell_virtual_coords = torch.bmm(supercell_matrix, prim_virtual_coords)

    if single_sample:
        return prim_virtual_coords[0], supercell_virtual_coords[0]
    return prim_virtual_coords, supercell_virtual_coords


def virtual_coords_to_lattice_and_supercell(
    prim_virtual_coords: Tensor,
    supercell_virtual_coords: Tensor,
) -> tuple[Tensor, Tensor]:
    """Convert virtual coordinates back to lattice parameters and supercell matrix.

    Args:
        prim_virtual_coords: [B, 3, 3] or [3, 3] primitive lattice vectors
        supercell_virtual_coords: [B, 3, 3] or [3, 3] supercell lattice vectors

    Returns:
        lattice_params: [B, 6] or [6] lattice parameters
        supercell_matrix: [B, 3, 3] or [3, 3] supercell transformation matrix
    """
    single_sample = prim_virtual_coords.ndim == 2
    if single_sample:
        prim_virtual_coords = prim_virtual_coords.unsqueeze(0)
        supercell_virtual_coords = supercell_virtual_coords.unsqueeze(0)

    B = prim_virtual_coords.shape[0]
    device = prim_virtual_coords.device
    dtype = prim_virtual_coords.dtype

    # Convert primitive vectors to lattice params
    lattice_params = lattice_vectors_to_params(prim_virtual_coords)

    # Compute supercell matrix: supercell_matrix = supercell_vectors @ inv(prim_vectors)
    supercell_matrices = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    for i in range(B):
        # Convert to float32 for numpy compatibility (bfloat16 not supported)
        prim_np = prim_virtual_coords[i].detach().cpu().float().numpy()
        supercell_np = supercell_virtual_coords[i].detach().cpu().float().numpy()
        sm_np = compute_supercell_matrix_from_vectors(prim_np, supercell_np)
        supercell_matrices[i] = torch.from_numpy(sm_np).to(device=device, dtype=dtype)

    if single_sample:
        return lattice_params[0], supercell_matrices[0]
    return lattice_params, supercell_matrices
