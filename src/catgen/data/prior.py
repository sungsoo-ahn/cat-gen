"""Prior samplers for flow matching."""

from typing import Any, Dict, List, Protocol, Union

import torch
from torch import Tensor
from torch.distributions import LogNormal, Uniform

from src.catgen.constants import EPS_NUMERICAL


class Normalizer:
    """Generic normalizer for tensors using mean and standard deviation.

    Provides normalize and denormalize operations:
    - normalize: (x - mean) / (std + eps)
    - denormalize: x * std + mean

    Handles automatic device and dtype conversion.
    """

    def __init__(
        self,
        mean: Union[List[float], List[List[float]], float],
        std: Union[List[float], List[List[float]], float],
    ):
        """Initialize normalizer with mean and std.

        Args:
            mean: Mean value(s) for normalization. Can be scalar, 1D list, or 2D list.
            std: Standard deviation(s) for normalization. Same shape as mean.
        """
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def normalize(self, x: Tensor) -> Tensor:
        """Normalize tensor: (x - mean) / (std + eps)."""
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        return (x - mean) / (std + EPS_NUMERICAL)

    def denormalize(self, x: Tensor) -> Tensor:
        """Denormalize tensor: x * std + mean."""
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        return x * std + mean


class PriorSampler(Protocol):
    """Protocol for prior samplers used in flow matching."""

    def sample(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Sample from the prior distribution.

        Args:
            data: Dictionary containing at least:
                - prim_slab_cart_coords: (B, N, 3) target primitive slab coordinates
                - prim_slab_atom_mask: (B, N) boolean mask for valid prim_slab atoms
                - ads_cart_coords: (B, M, 3) target adsorbate coordinates
                - ads_atom_mask: (B, M) boolean mask for valid adsorbate atoms

        Returns:
            Dictionary containing:
                - prim_slab_coords_0: (B, N, 3) sampled prior primitive slab coordinates
                - ads_coords_0: (B, M, 3) sampled prior adsorbate coordinates
                - lattice_0: (B, 6) sampled prior lattice parameters (a, b, c, alpha, beta, gamma)
                - supercell_matrix_0: (B, 3, 3) sampled prior supercell matrix (normalized)
        """
        ...


class CatPriorSampler:
    """Prior sampler for flow matching.
    
    Samples all priors needed for flow matching:
    - prim_slab_coords_0: N(0, coord_std^2) in normalized space, then denormalized to raw space (Angstrom)
    - ads_coords_0: N(0, coord_std^2) in normalized space, then denormalized to raw space (Angstrom)
    - lattice_0: LogNormal (lengths) + Uniform (angles) - raw space (Angstrom, degrees)
    - supercell_matrix_0: N(0, 1) in normalized space, then denormalized to raw space
    - scaling_factor_0: N(0, coord_std^2) - Gaussian (raw space, no normalization)
    
    Note: prim_slab_coords, ads_coords, and supercell_matrix are sampled in normalized space and then
    denormalized to raw space. Lattice uses non-Gaussian distributions in raw space. Scaling factor uses raw Gaussian prior.
    """

    def __init__(
        self,
        coord_std: float = 1.0,
        lognormal_loc: List[float] = None,
        lognormal_scale: List[float] = None,
        uniform_low: float = 60.0,
        uniform_high: float = 120.0,
        uniform_eps: float = 0.1,
        # Supercell matrix normalization parameters
        supercell_mean: List[List[float]] = None,
        supercell_std: List[List[float]] = None,
        # Coordinate standardization parameters (computed from train set)
        prim_slab_coord_mean: List[float] = None,
        prim_slab_coord_std: List[float] = None,
        ads_coord_mean: List[float] = None,
        ads_coord_std: List[float] = None,
    ):
        self.coord_std = coord_std

        # LogNormal distribution for lattice lengths (a, b, c)
        self._lognormal = LogNormal(
            loc=torch.tensor(lognormal_loc),
            scale=torch.tensor(lognormal_scale),
        )

        # Uniform distribution for lattice angles (alpha, beta, gamma)
        self._uniform = Uniform(
            low=uniform_low - uniform_eps,
            high=uniform_high + uniform_eps,
        )

        # Create normalizers using the generic Normalizer class
        self.supercell_normalizer = Normalizer(supercell_mean, supercell_std)
        self.prim_slab_coord_normalizer = Normalizer(prim_slab_coord_mean, prim_slab_coord_std)
        self.ads_coord_normalizer = Normalizer(ads_coord_mean, ads_coord_std)

        # Angle normalization parameters (degrees)
        # Mean is center of uniform range, std is sqrt(variance) of uniform distribution
        angle_mean = (uniform_low + uniform_high) / 2.0  # 90.0
        angle_std = (uniform_high - uniform_low) / (2 * 3**0.5)  # ~17.32 for uniform
        self.angle_normalizer = Normalizer(angle_mean, angle_std)

    def sample(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Sample from prior distributions for coords, lattice, and supercell matrix."""
        prim_slab_cart_coords = data["prim_slab_cart_coords"]
        prim_slab_atom_mask = data["prim_slab_atom_mask"]
        ads_cart_coords = data["ads_cart_coords"]
        ads_atom_mask = data["ads_atom_mask"]

        batch_size, num_prim_slab_atoms, _ = prim_slab_cart_coords.shape
        _, num_ads_atoms, _ = ads_cart_coords.shape
        device = prim_slab_cart_coords.device
        dtype = prim_slab_cart_coords.dtype

        # Sample primitive slab coordinates from Gaussian N(0, coord_std^2) in normalized space, then denormalize to raw space
        prim_slab_coords_0_normalized = (
            torch.randn(batch_size, num_prim_slab_atoms, 3, device=device, dtype=dtype)
            * self.coord_std
        )
        prim_slab_coords_0_normalized = prim_slab_coords_0_normalized * prim_slab_atom_mask.unsqueeze(-1)
        # Denormalize to raw space (Angstrom)
        prim_slab_coords_0 = self.prim_slab_coord_normalizer.denormalize(prim_slab_coords_0_normalized)

        # Sample adsorbate coordinates from Gaussian N(0, coord_std^2) in normalized space, then denormalize to raw space
        ads_coords_0_normalized = (
            torch.randn(batch_size, num_ads_atoms, 3, device=device, dtype=dtype)
            * self.coord_std
        )
        ads_coords_0_normalized = ads_coords_0_normalized * ads_atom_mask.unsqueeze(-1)
        # Denormalize to raw space (Angstrom)
        ads_coords_0 = self.ads_coord_normalizer.denormalize(ads_coords_0_normalized)

        # Sample lattice lengths from LogNormal distribution
        lengths_0 = self._lognormal.sample((batch_size,))  # (B, 3)
        lengths_0 = lengths_0.to(device=device, dtype=dtype)

        # Sample lattice angles from Uniform distribution
        angles_0 = self._uniform.sample((batch_size, 3))  # (B, 3)
        angles_0 = angles_0.to(device=device, dtype=dtype)

        # Combine into lattice_0: (a, b, c, alpha, beta, gamma)
        lattice_0 = torch.cat([lengths_0, angles_0], dim=-1)  # (B, 6)

        # # Sample supercell matrix from standard normal N(0, 1) in normalized space, then denormalize to raw space
        # supercell_matrix_0_normalized = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)  # (B, 3, 3)
        # supercell_matrix_0 = self.denormalize_supercell(supercell_matrix_0_normalized)  # (B, 3, 3) raw space
        
        identity = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        noise = torch.randn(batch_size, 3, 3, device=device, dtype=dtype) * 0.1
        supercell_matrix_0 = identity + noise

        # Sample scaling factor from Gaussian N(0, coord_std^2) - no normalization
        scaling_factor_0 = torch.randn(batch_size, device=device, dtype=dtype) * self.coord_std  # (B,)

        return {
            "prim_slab_coords_0": prim_slab_coords_0,  # (B, N, 3) raw space (Angstrom)
            "ads_coords_0": ads_coords_0,  # (B, M, 3) raw space (Angstrom)
            "lattice_0": lattice_0,  # (B, 6) raw space (Angstrom, degrees)
            "supercell_matrix_0": supercell_matrix_0,  # (B, 3, 3) raw space
            "scaling_factor_0": scaling_factor_0,  # (B,) raw value
        }

    # Backward-compatible properties for direct attribute access
    @property
    def ads_coord_mean(self) -> torch.Tensor:
        """Access ads_coord_mean from normalizer for backward compatibility."""
        return self.ads_coord_normalizer.mean

    @property
    def ads_coord_std(self) -> torch.Tensor:
        """Access ads_coord_std from normalizer for backward compatibility."""
        return self.ads_coord_normalizer.std

    @property
    def prim_slab_coord_mean(self) -> torch.Tensor:
        """Access prim_slab_coord_mean from normalizer for backward compatibility."""
        return self.prim_slab_coord_normalizer.mean

    @property
    def prim_slab_coord_std(self) -> torch.Tensor:
        """Access prim_slab_coord_std from normalizer for backward compatibility."""
        return self.prim_slab_coord_normalizer.std

    @property
    def supercell_mean(self) -> torch.Tensor:
        """Access supercell_mean from normalizer for backward compatibility."""
        return self.supercell_normalizer.mean

    @property
    def supercell_std(self) -> torch.Tensor:
        """Access supercell_std from normalizer for backward compatibility."""
        return self.supercell_normalizer.std

    # Backward-compatible wrapper methods that delegate to normalizers
    def normalize_supercell(self, supercell_matrix: torch.Tensor) -> torch.Tensor:
        """Normalize supercell matrix. Delegates to supercell_normalizer."""
        return self.supercell_normalizer.normalize(supercell_matrix)

    def denormalize_supercell(self, supercell_matrix_normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize supercell matrix. Delegates to supercell_normalizer."""
        return self.supercell_normalizer.denormalize(supercell_matrix_normalized)

    def normalize_prim_slab_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalize prim_slab coordinates. Delegates to prim_slab_coord_normalizer."""
        return self.prim_slab_coord_normalizer.normalize(coords)

    def denormalize_prim_slab_coords(self, coords_normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize prim_slab coordinates. Delegates to prim_slab_coord_normalizer."""
        return self.prim_slab_coord_normalizer.denormalize(coords_normalized)

    def normalize_ads_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalize adsorbate coordinates. Delegates to ads_coord_normalizer."""
        return self.ads_coord_normalizer.normalize(coords)

    def denormalize_ads_coords(self, coords_normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize adsorbate coordinates. Delegates to ads_coord_normalizer."""
        return self.ads_coord_normalizer.denormalize(coords_normalized)

    def normalize_angles(self, angles: torch.Tensor) -> torch.Tensor:
        """Normalize lattice angles. Delegates to angle_normalizer."""
        return self.angle_normalizer.normalize(angles)

    def denormalize_angles(self, angles_normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize lattice angles. Delegates to angle_normalizer."""
        return self.angle_normalizer.denormalize(angles_normalized)
