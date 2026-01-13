from einops import rearrange
from typing import Optional
import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

from src.catgen.models.utils import LinearNoBias
from src.catgen.models.transformers import DiT, PositionalEmbedder
from src.catgen.constants import (
    NUM_ELEMENTS,
    EPS_NUMERICAL,
)

logger = logging.getLogger(__name__)


def _clamp_element_indices(
    tensor: torch.Tensor,
    min_val: int,
    max_val: int,
    context: str = "element",
) -> torch.Tensor:
    """Clamp element indices to valid range with optional warning."""
    out_of_range = (tensor < min_val) | (tensor > max_val)
    if out_of_range.any():
        num_invalid = out_of_range.sum().item()
        logger.warning(
            f"{context}: {num_invalid} values out of range [{min_val}, {max_val}], clamping"
        )
    return tensor.clamp(min_val, max_val)


# =============================================================================
# Architecture: Input Embedder → Transformer → Output Projection
# =============================================================================


class InputEmbedder(Module):
    """
    Embeds all input features into hidden representations.

    Handles:
    - Primitive slab atoms: element one-hot + mask + noisy positions
    - Adsorbate atoms: element one-hot + mask + ref position + binding atom + noisy positions
    - Virtual atoms: one-hot ID (0-5) + noisy lattice vectors
    - Scaling factor: projected to hidden dim and broadcast

    Output: (B, N+M+6, hidden_dim) joint atom representations
    """

    NUM_VIRTUAL_ATOMS = 6

    def __init__(self, hidden_dim: int, positional_encoding: bool = False):
        super().__init__()

        # Prim slab features: atom_pad_mask (1) + element one-hot (NUM_ELEMENTS = 100)
        prim_slab_feature_dim = 1 + NUM_ELEMENTS  # 101
        self.embed_prim_slab_features = LinearNoBias(prim_slab_feature_dim, hidden_dim)

        # Adsorbate features: ref_pos (3) + atom_pad_mask (1) + bind_atomic_num (NUM_ELEMENTS) + element one-hot (NUM_ELEMENTS)
        ads_feature_dim = 3 + 1 + NUM_ELEMENTS + NUM_ELEMENTS
        self.embed_ads_features = LinearNoBias(ads_feature_dim, hidden_dim)

        # Virtual atom features: one-hot encoding of virtual_id (0-5)
        virtual_feature_dim = self.NUM_VIRTUAL_ATOMS  # 6
        self.embed_virtual_features = LinearNoBias(virtual_feature_dim, hidden_dim)

        # Position embedding for noisy coords (applies to all atom types including virtual)
        self.x_to_hidden = LinearNoBias(3, hidden_dim)

        # Scaling factor embedding (scalar -> hidden_dim, broadcast to all atoms)
        self.sf_to_hidden = LinearNoBias(1, hidden_dim)

        # Optional positional encoding
        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            self.pos_emb = PositionalEmbedder(hidden_dim)

    def forward(
        self,
        prim_slab_x_t,
        ads_x_t,
        prim_virtual_t,
        supercell_virtual_t,
        sf_t,
        feats,
        multiplicity=1,
    ):
        """
        Args:
            prim_slab_x_t: (B*mult, N, 3) noisy prim_slab coordinates
            ads_x_t: (B*mult, M, 3) noisy adsorbate coordinates
            prim_virtual_t: (B*mult, 3, 3) noisy primitive lattice vectors
            supercell_virtual_t: (B*mult, 3, 3) noisy supercell lattice vectors
            sf_t: (B*mult,) noisy scaling factor
            feats: feature dictionary (not yet repeated for multiplicity)
            multiplicity: number of samples per input

        Returns:
            h: (B*mult, N+M+6, hidden_dim) joint atom representations
            mask: (B*mult, N+M+6) joint padding mask
            n_prim_slab: int, number of prim_slab atoms (N)
            n_ads: int, number of adsorbate atoms (M)
        """
        B, N, _ = feats["prim_slab_cart_coords"].shape
        M = feats["ads_cart_coords"].shape[1]
        B_mult = prim_slab_x_t.shape[0]
        device = prim_slab_x_t.device

        # === Prim slab features ===
        clamped_element = _clamp_element_indices(
            feats["ref_prim_slab_element"], 0, NUM_ELEMENTS - 1, "ref_prim_slab_element"
        )
        prim_slab_element_onehot = F.one_hot(clamped_element, num_classes=NUM_ELEMENTS).float()
        prim_slab_mask = feats["prim_slab_atom_pad_mask"]

        prim_slab_feats = torch.cat([
            prim_slab_mask.unsqueeze(-1),
            prim_slab_element_onehot,
        ], dim=-1)
        h_prim_slab = self.embed_prim_slab_features(prim_slab_feats)
        h_prim_slab = h_prim_slab.repeat_interleave(multiplicity, 0)
        prim_slab_mask = prim_slab_mask.repeat_interleave(multiplicity, 0)

        # === Adsorbate features ===
        clamped_ads_element = _clamp_element_indices(
            feats["ref_ads_element"], 0, NUM_ELEMENTS - 1, "ref_ads_element"
        )
        ads_element_onehot = F.one_hot(clamped_ads_element, num_classes=NUM_ELEMENTS).float()
        ref_ads_pos = feats["ref_ads_pos"]
        bind_atom_idx = _clamp_element_indices(
            feats["bind_ads_atom"], 0, NUM_ELEMENTS - 1, "bind_ads_atom"
        )
        bind_atom_onehot = F.one_hot(bind_atom_idx, num_classes=NUM_ELEMENTS).float()
        bind_atom_expanded = bind_atom_onehot.unsqueeze(1).expand(-1, M, -1)
        ads_mask = feats["ads_atom_pad_mask"]

        ads_feats = torch.cat([
            ref_ads_pos,
            ads_mask.unsqueeze(-1),
            bind_atom_expanded,
            ads_element_onehot,
        ], dim=-1)
        h_ads = self.embed_ads_features(ads_feats)
        h_ads = h_ads.repeat_interleave(multiplicity, 0)
        ads_mask = ads_mask.repeat_interleave(multiplicity, 0)

        # === Virtual atom features ===
        virtual_ids = torch.arange(self.NUM_VIRTUAL_ATOMS, device=device)
        virtual_onehot = F.one_hot(virtual_ids, num_classes=self.NUM_VIRTUAL_ATOMS).float()
        virtual_feats = virtual_onehot.unsqueeze(0).expand(B_mult, -1, -1)
        h_virtual = self.embed_virtual_features(virtual_feats)

        # === Concatenate all atom embeddings ===
        h = torch.cat([h_prim_slab, h_ads, h_virtual], dim=1)

        # === Add noisy position embeddings ===
        x_t = torch.cat([
            prim_slab_x_t,
            ads_x_t,
            prim_virtual_t,
            supercell_virtual_t,
        ], dim=1)
        h = h + self.x_to_hidden(x_t)

        # === Add scaling factor embedding (broadcast) ===
        sf_embed = self.sf_to_hidden(sf_t.unsqueeze(-1))
        h = h + rearrange(sf_embed, "b d -> b 1 d")

        # === Create joint mask ===
        virtual_mask = torch.ones(B_mult, self.NUM_VIRTUAL_ATOMS, device=device, dtype=prim_slab_mask.dtype)
        mask = torch.cat([prim_slab_mask, ads_mask, virtual_mask], dim=1)

        return h, mask, N, M


class TransformerBackbone(Module):
    """
    Single transformer backbone for processing all atoms.

    Uses DiT (Diffusion Transformer) architecture with time conditioning.
    """

    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        heads: int,
        attention_impl: str = "pytorch",
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.transformer = DiT(
            dim=hidden_dim,
            depth=depth,
            heads=heads,
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(self, h, t, mask):
        """
        Args:
            h: (B, N+M+6, hidden_dim) joint atom representations
            t: (B,) timesteps
            mask: (B, N+M+6) padding mask (1=valid, 0=padding)

        Returns:
            h: (B, N+M+6, hidden_dim) updated representations
        """
        return self.transformer(h, t, mask)


class OutputProjection(Module):
    """
    Projects hidden representations to output predictions.

    Outputs:
    - prim_slab_coords: (B, N, 3) per-atom coordinate predictions
    - ads_coords: (B, M, 3) per-atom coordinate predictions
    - prim_virtual_coords: (B, 3, 3) primitive lattice vectors
    - supercell_virtual_coords: (B, 3, 3) supercell lattice vectors
    - scaling_factor: (B,) scalar from global pooling
    """

    NUM_VIRTUAL_ATOMS = 6

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Per-atom coordinate projections
        self.feats_to_prim_slab_coords = nn.Sequential(
            nn.LayerNorm(hidden_dim), LinearNoBias(hidden_dim, 3)
        )
        self.feats_to_ads_coords = nn.Sequential(
            nn.LayerNorm(hidden_dim), LinearNoBias(hidden_dim, 3)
        )
        self.feats_to_prim_virtual_coords = nn.Sequential(
            nn.LayerNorm(hidden_dim), LinearNoBias(hidden_dim, 3)
        )
        self.feats_to_supercell_virtual_coords = nn.Sequential(
            nn.LayerNorm(hidden_dim), LinearNoBias(hidden_dim, 3)
        )

        # Scaling factor from global pooling
        self.feats_to_scaling_factor = nn.Sequential(
            nn.LayerNorm(hidden_dim), LinearNoBias(hidden_dim, 1)
        )

    def forward(self, h, n_prim_slab, n_ads, prim_slab_mask):
        """
        Args:
            h: (B, N+M+6, hidden_dim) joint atom representations
            n_prim_slab: int, number of prim_slab atoms (N)
            n_ads: int, number of adsorbate atoms (M)
            prim_slab_mask: (B, N) mask for prim_slab atoms

        Returns:
            Dictionary with all predictions
        """
        # Split representations by atom type
        h_prim_slab = h[:, :n_prim_slab, :]
        h_ads = h[:, n_prim_slab:n_prim_slab + n_ads, :]
        h_prim_virtual = h[:, n_prim_slab + n_ads:n_prim_slab + n_ads + 3, :]
        h_supercell_virtual = h[:, n_prim_slab + n_ads + 3:, :]

        # Coordinate predictions
        prim_slab_r = self.feats_to_prim_slab_coords(h_prim_slab)
        ads_r = self.feats_to_ads_coords(h_ads)
        prim_virtual = self.feats_to_prim_virtual_coords(h_prim_virtual)
        supercell_virtual = self.feats_to_supercell_virtual_coords(h_supercell_virtual)

        # Scaling factor from global pooling (prim_slab atoms only)
        num_atoms = prim_slab_mask.sum(dim=1, keepdim=True)
        h_global = torch.sum(h_prim_slab * prim_slab_mask[..., None], dim=1) / (num_atoms + EPS_NUMERICAL)
        sf = self.feats_to_scaling_factor(h_global).squeeze(-1)

        return {
            "prim_slab_r_update": prim_slab_r,
            "ads_r_update": ads_r,
            "prim_virtual_update": prim_virtual,
            "supercell_virtual_update": supercell_virtual,
            "sf_update": sf,
        }
