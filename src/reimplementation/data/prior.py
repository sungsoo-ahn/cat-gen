"""Prior samplers for flow matching."""

from typing import Any, Dict, List, Protocol

import torch
from torch.distributions import LogNormal, Uniform


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
    - scaling_factor_0: N(0, coord_std^2) in normalized space, then denormalized to raw space

    Note: All continuous variables except lattice are sampled in normalized space and then
    denormalized to raw space. Lattice uses domain-specific distributions (LogNormal + Uniform)
    in raw space for physical plausibility.

    Normalization methods are provided for all variables:
    - normalize_prim_slab_coords / denormalize_prim_slab_coords: Z-score for coordinates
    - normalize_ads_coords / denormalize_ads_coords: Z-score for coordinates
    - normalize_supercell / denormalize_supercell: Z-score for supercell matrix
    - normalize_lattice / denormalize_lattice: Unit conversion + Z-score
    - normalize_scaling_factor / denormalize_scaling_factor: Z-score
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
        # Lattice normalization parameters (after unit conversion: nm, radians)
        lattice_length_mean: List[float] = None,
        lattice_length_std: List[float] = None,
        lattice_angle_mean: List[float] = None,
        lattice_angle_std: List[float] = None,
        # Scaling factor normalization parameters
        scaling_factor_mean: float = 0.0,
        scaling_factor_std: float = 1.0,
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
        
        self.supercell_mean = torch.tensor(supercell_mean, dtype=torch.float32)  # (3, 3)
        self.supercell_std = torch.tensor(supercell_std, dtype=torch.float32)  # (3, 3)
        
        self.prim_slab_coord_mean = torch.tensor(prim_slab_coord_mean, dtype=torch.float32)  # (3,)
        self.prim_slab_coord_std = torch.tensor(prim_slab_coord_std, dtype=torch.float32)  # (3,)
        
        self.ads_coord_mean = torch.tensor(ads_coord_mean, dtype=torch.float32)  # (3,)
        self.ads_coord_std = torch.tensor(ads_coord_std, dtype=torch.float32)  # (3,)

        # Lattice normalization parameters (after unit conversion)
        if lattice_length_mean is not None:
            self.lattice_length_mean = torch.tensor(lattice_length_mean, dtype=torch.float32)  # (3,) nm
            self.lattice_length_std = torch.tensor(lattice_length_std, dtype=torch.float32)  # (3,) nm
            self.lattice_angle_mean = torch.tensor(lattice_angle_mean, dtype=torch.float32)  # (3,) radians
            self.lattice_angle_std = torch.tensor(lattice_angle_std, dtype=torch.float32)  # (3,) radians
        else:
            # Default: no normalization (mean=0, std=1 after unit conversion)
            self.lattice_length_mean = torch.zeros(3, dtype=torch.float32)
            self.lattice_length_std = torch.ones(3, dtype=torch.float32)
            self.lattice_angle_mean = torch.zeros(3, dtype=torch.float32)
            self.lattice_angle_std = torch.ones(3, dtype=torch.float32)

        # Scaling factor normalization parameters
        self.scaling_factor_mean = scaling_factor_mean
        self.scaling_factor_std = scaling_factor_std

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
        prim_slab_coords_0 = self.denormalize_prim_slab_coords(prim_slab_coords_0_normalized)

        # Sample adsorbate coordinates from Gaussian N(0, coord_std^2) in normalized space, then denormalize to raw space
        ads_coords_0_normalized = (
            torch.randn(batch_size, num_ads_atoms, 3, device=device, dtype=dtype)
            * self.coord_std
        )
        ads_coords_0_normalized = ads_coords_0_normalized * ads_atom_mask.unsqueeze(-1)
        # Denormalize to raw space (Angstrom)
        ads_coords_0 = self.denormalize_ads_coords(ads_coords_0_normalized)

        # Sample lattice lengths from LogNormal distribution
        lengths_0 = self._lognormal.sample((batch_size,))  # (B, 3)
        lengths_0 = lengths_0.to(device=device, dtype=dtype)

        # Sample lattice angles from Uniform distribution
        angles_0 = self._uniform.sample((batch_size, 3))  # (B, 3)
        angles_0 = angles_0.to(device=device, dtype=dtype)

        # Combine into lattice_0: (a, b, c, alpha, beta, gamma)
        lattice_0 = torch.cat([lengths_0, angles_0], dim=-1)  # (B, 6)

        # Sample supercell matrix from standard normal N(0, 1) in normalized space, then denormalize to raw space
        supercell_matrix_0_normalized = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)  # (B, 3, 3)
        supercell_matrix_0 = self.denormalize_supercell(supercell_matrix_0_normalized)  # (B, 3, 3) raw space

        # Sample scaling factor from Gaussian N(0, coord_std^2) in normalized space, then denormalize
        scaling_factor_0_normalized = torch.randn(batch_size, device=device, dtype=dtype) * self.coord_std
        scaling_factor_0 = self.denormalize_scaling_factor(scaling_factor_0_normalized)  # (B,) raw value

        return {
            "prim_slab_coords_0": prim_slab_coords_0,  # (B, N, 3) raw space (Angstrom)
            "ads_coords_0": ads_coords_0,  # (B, M, 3) raw space (Angstrom)
            "lattice_0": lattice_0,  # (B, 6) raw space (Angstrom, degrees)
            "supercell_matrix_0": supercell_matrix_0,  # (B, 3, 3) raw space
            "scaling_factor_0": scaling_factor_0,  # (B,) raw value
        }

    def normalize_supercell(self, supercell_matrix: torch.Tensor) -> torch.Tensor:
        """Normalize supercell matrix using mean and std.
        
        Args:
            supercell_matrix: (B, 3, 3) raw supercell matrix
            
        Returns:
            (B, 3, 3) normalized supercell matrix
        """
        mean = self.supercell_mean.to(supercell_matrix.device, supercell_matrix.dtype)
        std = self.supercell_std.to(supercell_matrix.device, supercell_matrix.dtype)
        return (supercell_matrix - mean) / (std + 1e-8)

    def denormalize_supercell(self, supercell_matrix_normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize supercell matrix back to original space.
        
        Args:
            supercell_matrix_normalized: (B, 3, 3) normalized supercell matrix
            
        Returns:
            (B, 3, 3) denormalized supercell matrix
        """
        mean = self.supercell_mean.to(supercell_matrix_normalized.device, supercell_matrix_normalized.dtype)
        std = self.supercell_std.to(supercell_matrix_normalized.device, supercell_matrix_normalized.dtype)
        return supercell_matrix_normalized * std + mean

    def normalize_prim_slab_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Standardize prim_slab coordinates using mean and std.
        
        Args:
            coords: (B, N, 3) raw prim_slab coordinates in Angstrom
            
        Returns:
            (B, N, 3) standardized coordinates
        """
        mean = self.prim_slab_coord_mean.to(coords.device, coords.dtype)
        std = self.prim_slab_coord_std.to(coords.device, coords.dtype)
        return (coords - mean) / (std + 1e-8)

    def denormalize_prim_slab_coords(self, coords_normalized: torch.Tensor) -> torch.Tensor:
        """Destandardize prim_slab coordinates back to original space.
        
        Args:
            coords_normalized: (B, N, 3) standardized coordinates
            
        Returns:
            (B, N, 3) denormalized coordinates in Angstrom
        """
        mean = self.prim_slab_coord_mean.to(coords_normalized.device, coords_normalized.dtype)
        std = self.prim_slab_coord_std.to(coords_normalized.device, coords_normalized.dtype)
        return coords_normalized * std + mean

    def normalize_ads_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Standardize adsorbate coordinates using mean and std.
        
        Args:
            coords: (B, M, 3) raw adsorbate coordinates in Angstrom
            
        Returns:
            (B, M, 3) standardized coordinates
        """
        mean = self.ads_coord_mean.to(coords.device, coords.dtype)
        std = self.ads_coord_std.to(coords.device, coords.dtype)
        return (coords - mean) / (std + 1e-8)

    def denormalize_ads_coords(self, coords_normalized: torch.Tensor) -> torch.Tensor:
        """Destandardize adsorbate coordinates back to original space.

        Args:
            coords_normalized: (B, M, 3) standardized coordinates

        Returns:
            (B, M, 3) denormalized coordinates in Angstrom
        """
        mean = self.ads_coord_mean.to(coords_normalized.device, coords_normalized.dtype)
        std = self.ads_coord_std.to(coords_normalized.device, coords_normalized.dtype)
        return coords_normalized * std + mean

    def normalize_lattice(self, lattice: torch.Tensor) -> torch.Tensor:
        """Normalize lattice parameters: raw (Ang, deg) -> normalized (Z-score nm, Z-score rad).

        Args:
            lattice: (B, 6) raw lattice [a, b, c in Angstrom, alpha, beta, gamma in degrees]

        Returns:
            (B, 6) normalized lattice [Z-score lengths in nm, Z-score angles in radians]
        """
        # Unit conversion
        lengths_nm = lattice[:, :3] * 0.1  # Angstrom -> nm
        angles_rad = lattice[:, 3:] * (torch.pi / 180.0)  # degrees -> radians

        # Z-score normalization
        length_mean = self.lattice_length_mean.to(lattice.device, lattice.dtype)
        length_std = self.lattice_length_std.to(lattice.device, lattice.dtype)
        angle_mean = self.lattice_angle_mean.to(lattice.device, lattice.dtype)
        angle_std = self.lattice_angle_std.to(lattice.device, lattice.dtype)

        lengths_norm = (lengths_nm - length_mean) / (length_std + 1e-8)
        angles_norm = (angles_rad - angle_mean) / (angle_std + 1e-8)

        return torch.cat([lengths_norm, angles_norm], dim=-1)

    def denormalize_lattice(self, lattice_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize lattice parameters: normalized -> raw (Ang, deg).

        Args:
            lattice_norm: (B, 6) normalized lattice

        Returns:
            (B, 6) raw lattice [a, b, c in Angstrom, alpha, beta, gamma in degrees]
        """
        length_mean = self.lattice_length_mean.to(lattice_norm.device, lattice_norm.dtype)
        length_std = self.lattice_length_std.to(lattice_norm.device, lattice_norm.dtype)
        angle_mean = self.lattice_angle_mean.to(lattice_norm.device, lattice_norm.dtype)
        angle_std = self.lattice_angle_std.to(lattice_norm.device, lattice_norm.dtype)

        # Undo Z-score
        lengths_nm = lattice_norm[:, :3] * length_std + length_mean
        angles_rad = lattice_norm[:, 3:] * angle_std + angle_mean

        # Undo unit conversion
        lengths_ang = lengths_nm * 10.0  # nm -> Angstrom
        angles_deg = angles_rad * (180.0 / torch.pi)  # radians -> degrees

        return torch.cat([lengths_ang, angles_deg], dim=-1)

    def normalize_scaling_factor(self, sf: torch.Tensor) -> torch.Tensor:
        """Normalize scaling factor using mean and std.

        Args:
            sf: (B,) raw scaling factors

        Returns:
            (B,) normalized scaling factors
        """
        return (sf - self.scaling_factor_mean) / (self.scaling_factor_std + 1e-8)

    def denormalize_scaling_factor(self, sf_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize scaling factor back to raw space.

        Args:
            sf_norm: (B,) normalized scaling factors

        Returns:
            (B,) raw scaling factors
        """
        return sf_norm * self.scaling_factor_std + self.scaling_factor_mean
