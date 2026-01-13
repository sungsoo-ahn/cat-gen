"""LDDT (Local Distance Difference Test) loss implementation.

Based on SimpleFold (https://arxiv.org/abs/2509.18480) and the original LDDT paper
(Mariani et al., 2013). LDDT measures the fraction of preserved local distances
within tolerance thresholds, providing a structure-aware loss for crystal generation.

Key features:
- Supports periodic boundary conditions via minimum image convention
- Uses standard LDDT thresholds: 0.5, 1.0, 2.0, 4.0 Angstroms
- 15 Angstrom cutoff for neighbor atoms
"""

import torch
from torch import Tensor

LDDT_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]  # Angstroms
LDDT_CUTOFF = 15.0  # Angstroms


def compute_pairwise_distances(coords: Tensor) -> Tensor:
    """Compute pairwise Euclidean distances.

    Args:
        coords: (B, N, 3) atom coordinates

    Returns:
        distances: (B, N, N) pairwise distance matrix
    """
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 3)
    distances = torch.norm(diff, dim=-1)
    return distances


def compute_pbc_pairwise_distances(
    coords: Tensor,
    lattice_vectors: Tensor,
) -> Tensor:
    """Compute pairwise distances with minimum image convention.

    For periodic crystals, the minimum image convention finds the shortest
    distance between atoms considering periodic boundary conditions.

    Args:
        coords: (B, N, 3) Cartesian coordinates
        lattice_vectors: (B, 3, 3) lattice vectors (row vectors)

    Returns:
        distances: (B, N, N) minimum image distances
    """
    # Compute inverse lattice for Cartesian -> fractional conversion
    lattice_inv = torch.linalg.inv(lattice_vectors)  # (B, 3, 3)

    # Pairwise displacement vectors in Cartesian
    diff_cart = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 3)

    # Convert to fractional coordinates
    # diff_frac[b, i, j, :] = diff_cart[b, i, j, :] @ lattice_inv[b].T
    diff_frac = torch.einsum("bnmk,bkl->bnml", diff_cart, lattice_inv.transpose(-1, -2))

    # Apply minimum image: wrap to [-0.5, 0.5)
    diff_frac = diff_frac - torch.round(diff_frac)

    # Convert back to Cartesian
    # diff_cart_mic[b, i, j, :] = diff_frac[b, i, j, :] @ lattice_vectors[b]
    diff_cart_mic = torch.einsum("bnmk,bkl->bnml", diff_frac, lattice_vectors)

    # Compute distances
    distances = torch.norm(diff_cart_mic, dim=-1)
    return distances


def compute_lddt_score(
    pred_dists: Tensor,
    true_dists: Tensor,
    pair_mask: Tensor,
    cutoff: float = LDDT_CUTOFF,
    thresholds: list[float] | None = None,
) -> Tensor:
    """Compute LDDT score from distance matrices.

    LDDT measures the fraction of interatomic distances that are preserved
    within tolerance thresholds. The score is averaged across four thresholds:
    0.5, 1.0, 2.0, and 4.0 Angstroms.

    Args:
        pred_dists: (B, N, N) predicted pairwise distances
        true_dists: (B, N, N) ground truth pairwise distances
        pair_mask: (B, N, N) valid atom pair mask (True for valid)
        cutoff: neighbor cutoff distance in Angstroms
        thresholds: distance tolerance thresholds (default: [0.5, 1.0, 2.0, 4.0])

    Returns:
        lddt_score: (B,) LDDT score per sample [0, 1]
    """
    if thresholds is None:
        thresholds = LDDT_THRESHOLDS

    # Neighbor mask: only consider pairs within cutoff in ground truth
    neighbor_mask = (true_dists < cutoff) & pair_mask

    # Exclude self-distances (diagonal)
    B, N, _ = pred_dists.shape
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=pred_dists.device).unsqueeze(0)
    neighbor_mask = neighbor_mask & diag_mask

    # Distance errors
    dist_errors = torch.abs(pred_dists - true_dists)  # (B, N, N)

    # For each threshold, compute fraction of preserved distances
    preserved_fractions = []
    for thresh in thresholds:
        preserved = (dist_errors < thresh).float()  # (B, N, N)
        # Masked mean over valid neighbor pairs
        num_preserved = (preserved * neighbor_mask).sum(dim=(1, 2))
        num_neighbors = neighbor_mask.float().sum(dim=(1, 2)).clamp(min=1)
        preserved_fractions.append(num_preserved / num_neighbors)

    # Average across thresholds
    lddt_score = torch.stack(preserved_fractions, dim=0).mean(dim=0)  # (B,)
    return lddt_score


def compute_lddt_loss(
    pred_coords: Tensor,
    true_coords: Tensor,
    atom_mask: Tensor,
    lattice_vectors: Tensor | None = None,
    cutoff: float = LDDT_CUTOFF,
    thresholds: list[float] | None = None,
    use_pbc: bool = True,
) -> Tensor:
    """Compute LDDT loss for coordinates.

    LDDT loss = 1 - LDDT score, so a perfect structure has loss 0.

    Args:
        pred_coords: (B, N, 3) predicted coordinates
        true_coords: (B, N, 3) ground truth coordinates
        atom_mask: (B, N) valid atom mask (True for valid atoms)
        lattice_vectors: (B, 3, 3) lattice vectors for PBC (required if use_pbc=True)
        cutoff: neighbor cutoff distance in Angstroms
        thresholds: LDDT tolerance thresholds
        use_pbc: whether to use periodic boundary conditions

    Returns:
        lddt_loss: (B,) LDDT loss per sample (1 - lddt_score)
    """
    if thresholds is None:
        thresholds = LDDT_THRESHOLDS

    # Compute pairwise distances
    if use_pbc and lattice_vectors is not None:
        pred_dists = compute_pbc_pairwise_distances(pred_coords, lattice_vectors)
        true_dists = compute_pbc_pairwise_distances(true_coords, lattice_vectors)
    else:
        pred_dists = compute_pairwise_distances(pred_coords)
        true_dists = compute_pairwise_distances(true_coords)

    # Create pair mask from atom mask: valid if both atoms valid
    # Convert to bool if necessary (mask might be float 0/1)
    if atom_mask.dtype != torch.bool:
        atom_mask = atom_mask.bool()
    pair_mask = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1)  # (B, N, N)

    # Compute LDDT score
    lddt_score = compute_lddt_score(
        pred_dists, true_dists, pair_mask, cutoff, thresholds
    )

    # Loss = 1 - score (so perfect structure has loss 0)
    lddt_loss = 1.0 - lddt_score
    return lddt_loss
