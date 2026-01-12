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
    MASK_TOKEN_INDEX,
    NUM_ELEMENTS_WITH_MASK,
    EPS_NUMERICAL,
)

logger = logging.getLogger(__name__)


def _clamp_element_indices(
    tensor: torch.Tensor,
    min_val: int,
    max_val: int,
    context: str = "element",
) -> torch.Tensor:
    """Clamp element indices to valid range with optional warning.

    Args:
        tensor: Input tensor with element indices
        min_val: Minimum valid value (inclusive)
        max_val: Maximum valid value (inclusive)
        context: Description for logging (e.g., "prim_slab_element")

    Returns:
        Clamped tensor
    """
    out_of_range = (tensor < min_val) | (tensor > max_val)
    if out_of_range.any():
        num_invalid = out_of_range.sum().item()
        logger.warning(
            f"{context}: {num_invalid} values out of range [{min_val}, {max_val}], clamping"
        )
    return tensor.clamp(min_val, max_val)


class AtomAttentionEncoder(Module):
    """
    Encoder that jointly processes primitive slab, adsorbate, and virtual atoms.

    Virtual atoms represent lattice information:
    - 3 primitive lattice virtual atoms (vectors a, b, c)
    - 3 supercell lattice virtual atoms (vectors a', b', c')

    Input:
        - prim_slab_x_t: (B, N, 3) noisy prim_slab coordinates
        - ads_x_t: (B, M, 3) noisy adsorbate coordinates
        - prim_virtual_t: (B, 3, 3) noisy primitive lattice vectors
        - supercell_virtual_t: (B, 3, 3) noisy supercell lattice vectors
        - sf_t: (B,) noisy scaling factor
        - t: timesteps
        - feats: dictionary containing ref_prim_slab_element, ref_ads_element, masks, etc.

    Output:
        - joint representation: (B, N+M+6, atom_s)
    """

    # Number of virtual atoms (3 primitive + 3 supercell)
    NUM_VIRTUAL_ATOMS = 6

    def __init__(
        self,
        atom_s,
        atom_encoder_depth,
        atom_encoder_heads,
        attention_impl,
        positional_encoding=False,
        activation_checkpointing=False,
        dng: bool = False,
    ):
        super().__init__()

        self.dng = dng

        # Prim slab features: atom_pad_mask (1) + element one-hot
        # dng=True: includes MASK token (NUM_ELEMENTS_WITH_MASK = 101)
        # dng=False: no MASK token (NUM_ELEMENTS = 100)
        if dng:
            prim_slab_feature_dim = 1 + NUM_ELEMENTS_WITH_MASK  # 102
        else:
            prim_slab_feature_dim = 1 + NUM_ELEMENTS  # 101
        self.embed_prim_slab_features = LinearNoBias(prim_slab_feature_dim, atom_s)

        # Adsorbate features: ref_pos (3) + atom_pad_mask (1) + bind_atomic_num (1) + element one-hot (NUM_ELEMENTS)
        ads_feature_dim = 3 + 1 + NUM_ELEMENTS + NUM_ELEMENTS
        self.embed_ads_features = LinearNoBias(ads_feature_dim, atom_s)

        # Virtual atom features: one-hot encoding of virtual_id (0-5)
        # IDs 0, 1, 2: primitive lattice vectors (a, b, c)
        # IDs 3, 4, 5: supercell lattice vectors (a', b', c')
        virtual_feature_dim = self.NUM_VIRTUAL_ATOMS  # 6
        self.embed_virtual_features = LinearNoBias(virtual_feature_dim, atom_s)

        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            self.pos_emb = PositionalEmbedder(atom_s)

        # Position embedding for noisy coords (applies to all atom types including virtual)
        self.x_to_q_trans = LinearNoBias(3, atom_s)

        # Scaling factor is scalar (1-dimensional)
        self.sf_to_q_trans = LinearNoBias(1, atom_s)

        self.atom_encoder = DiT(
            dim=atom_s,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(self, prim_slab_x_t, ads_x_t, prim_virtual_t, supercell_virtual_t, sf_t, t, feats, multiplicity=1, prim_slab_element_t: Optional[torch.Tensor] = None):
        """
        Args:
            prim_slab_x_t: (B*mult, N, 3) noisy prim_slab coordinates
            ads_x_t: (B*mult, M, 3) noisy adsorbate coordinates
            prim_virtual_t: (B*mult, 3, 3) noisy primitive lattice vectors
            supercell_virtual_t: (B*mult, 3, 3) noisy supercell lattice vectors
            sf_t: (B*mult,) noisy scaling factor (normalized)
            t: (B*mult,) timesteps
            feats: feature dictionary (not yet repeated for multiplicity)
            multiplicity: number of samples per input

        Returns:
            q: (B*mult, N+M+6, atom_s) joint atom representations
            n_prim_slab: int, number of prim_slab atoms (N)
            n_ads: int, number of adsorbate atoms (M)
        """
        B, N, _ = feats["prim_slab_cart_coords"].shape
        M = feats["ads_cart_coords"].shape[1]
        B_mult = prim_slab_x_t.shape[0]
        device = prim_slab_x_t.device
        dtype = prim_slab_x_t.dtype

        prim_slab_mask = feats["prim_slab_atom_pad_mask"].bool()
        ads_mask = feats["ads_atom_pad_mask"].bool()

        # === Prim slab features ===
        # Features: atom_pad_mask (1) + element one-hot
        # dng=True: prim_slab_element_t is integer tensor (0=MASK, 1-100=elements) → one-hot with MASK
        # dng=False: use ref_prim_slab_element directly → one-hot without MASK
        n_prim_slab = N  # Track actual prim_slab atom count for return value
        if prim_slab_element_t is not None and self.dng:
            # dng=True: prim_slab_element_t is integer tensor (B*mult, N_actual)
            # Values: 0=MASK, 1-100=element atomic numbers
            N_actual = prim_slab_element_t.shape[1]
            B_x, N_x, _ = prim_slab_x_t.shape

            # Ensure N_actual matches prim_slab_x_t shape
            if N_actual != N_x:
                n_prim_slab = N_x
                prim_slab_element_t = prim_slab_element_t[:, :n_prim_slab]
            else:
                n_prim_slab = N_actual

            # Convert integer tensor to one-hot with MASK token
            # prim_slab_element_t: (B*mult, n_prim_slab) integer tensor (0=MASK, 1-100=elements)
            clamped_element = _clamp_element_indices(
                prim_slab_element_t, 0, NUM_ELEMENTS_WITH_MASK - 1, "prim_slab_element_t"
            )
            prim_slab_element_onehot = F.one_hot(
                clamped_element.long(),
                num_classes=NUM_ELEMENTS_WITH_MASK
            ).float()  # (B*mult, n_prim_slab, NUM_ELEMENTS_WITH_MASK)

            # mask needs multiplicity applied and sliced to match n_prim_slab
            prim_slab_mask_for_feats = feats["prim_slab_atom_pad_mask"].repeat_interleave(multiplicity, 0)
            prim_slab_mask_for_feats = prim_slab_mask_for_feats[:, :n_prim_slab]
        else:
            # dng=False: existing approach (no MASK token)
            clamped_element = _clamp_element_indices(
                feats["ref_prim_slab_element"], 0, NUM_ELEMENTS - 1, "ref_prim_slab_element"
            )
            prim_slab_element_onehot = F.one_hot(
                clamped_element,
                num_classes=NUM_ELEMENTS
            ).float()  # (B, N, NUM_ELEMENTS)
            prim_slab_mask_for_feats = feats["prim_slab_atom_pad_mask"]  # (B, N)

        prim_slab_feats = torch.cat([
            prim_slab_mask_for_feats.unsqueeze(-1),  # (B*mult, N, 1) or (B, N, 1)
            prim_slab_element_onehot,  # (B*mult, N, NUM_ELEMENTS) or (B, N, NUM_ELEMENTS)
        ], dim=-1)

        c_prim_slab = self.embed_prim_slab_features(prim_slab_feats)  # (B*mult, N, atom_s) or (B, N, atom_s)

        # Apply multiplicity only when not in dng mode with prim_slab_element_t
        if prim_slab_element_t is None or not self.dng:
            # Existing approach: (B, N, atom_s) -> (B*mult, N, atom_s)
            c_prim_slab = c_prim_slab.repeat_interleave(multiplicity, 0)
            # Also expand mask for joint_mask concatenation
            prim_slab_mask_for_feats = prim_slab_mask_for_feats.repeat_interleave(multiplicity, 0)

        # === Adsorbate features ===
        # Features: atom_pad_mask (1) + element one-hot (NUM_ELEMENTS)
        clamped_ads_element = _clamp_element_indices(
            feats["ref_ads_element"], 0, NUM_ELEMENTS - 1, "ref_ads_element"
        )
        ads_element_onehot = F.one_hot(
            clamped_ads_element,
            num_classes=NUM_ELEMENTS
        ).float()  # (B, M, NUM_ELEMENTS)

        ref_ads_pos = feats["ref_ads_pos"] # (B, M, 3)

        bind_atom_idx = _clamp_element_indices(
            feats["bind_ads_atom"], 0, NUM_ELEMENTS - 1, "bind_ads_atom"
        )  # (B)
        bind_atom_onehot = F.one_hot(bind_atom_idx, num_classes=NUM_ELEMENTS).float() # (B, NUM_ELEMENTS)
        bind_atom_expanded = bind_atom_onehot.unsqueeze(1).expand(-1, M, -1) # (B, M, NUM_ELEMENTS)

        ads_feats = torch.cat([
            ref_ads_pos, # (B, M, 3)
            feats["ads_atom_pad_mask"].unsqueeze(-1),  # (B, M, 1)
            bind_atom_expanded, # (B, M, NUM_ELEMENTS)
            ads_element_onehot,  # (B, M, NUM_ELEMENTS)
        ], dim=-1)

        c_ads = self.embed_ads_features(ads_feats)  # (B, M, atom_s)
        # ads always needs multiplicity applied
        c_ads = c_ads.repeat_interleave(multiplicity, 0)  # (B*mult, M, atom_s)

        # === Virtual atom features ===
        # Create one-hot encoding for virtual atom IDs (0-5)
        # IDs 0, 1, 2: primitive lattice vectors; IDs 3, 4, 5: supercell lattice vectors
        virtual_ids = torch.arange(self.NUM_VIRTUAL_ATOMS, device=device)
        virtual_onehot = F.one_hot(virtual_ids, num_classes=self.NUM_VIRTUAL_ATOMS).float()  # (6, 6)
        # Expand to batch size
        virtual_feats = virtual_onehot.unsqueeze(0).expand(B_mult, -1, -1)  # (B*mult, 6, 6)
        c_virtual = self.embed_virtual_features(virtual_feats)  # (B*mult, 6, atom_s)

        # === Concat all atom types ===
        # Order: [prim_slab, ads, prim_virtual (3), supercell_virtual (3)]
        c = torch.cat([c_prim_slab, c_ads, c_virtual], dim=1)  # (B*mult, N+M+6, atom_s)

        # Create joint mask: virtual atoms always valid (no padding)
        virtual_mask = torch.ones(B_mult, self.NUM_VIRTUAL_ATOMS, device=device, dtype=prim_slab_mask_for_feats.dtype)
        joint_mask = torch.cat([
            prim_slab_mask_for_feats,
            ads_mask.repeat_interleave(multiplicity, 0),
            virtual_mask
        ], dim=1)  # (B*mult, N+M+6)

        q = c

        # === Add noisy positions ===
        # Concat noisy coords: prim_slab_x_t + ads_x_t + virtual coords
        if prim_slab_x_t.shape[1] != n_prim_slab:
            prim_slab_x_t = prim_slab_x_t[:, :n_prim_slab, :]

        # Concatenate all coordinates including virtual atoms
        # prim_virtual_t and supercell_virtual_t are both (B*mult, 3, 3) where each row is a vector
        x_t = torch.cat([
            prim_slab_x_t,  # (B*mult, n_prim_slab, 3)
            ads_x_t,  # (B*mult, M, 3)
            prim_virtual_t,  # (B*mult, 3, 3)
            supercell_virtual_t,  # (B*mult, 3, 3)
        ], dim=1)  # (B*mult, n_prim_slab+M+6, 3)
        q = q + self.x_to_q_trans(x_t)

        # Add noisy scaling factor (sf_t is (B*mult,) -> expand to (B*mult, 1))
        sf_embed = self.sf_to_q_trans(sf_t.unsqueeze(-1))  # (B*mult, atom_s)
        q = q + rearrange(sf_embed, "b d -> b 1 d")

        # Pass through transformer (joint attention over all atoms including virtual)
        q = self.atom_encoder(q, t, joint_mask)  # (B*mult, n_prim_slab+M+6, atom_s)

        return q, n_prim_slab, M


class AtomAttentionDecoder(Module):
    """
    Decoder that outputs predictions for all atom coordinates including virtual atoms.

    Uses unified per-atom coordinate prediction for all atom types.

    Input:
        - x: (B, N+M+6, atom_s) joint atom representations
        - n_prim_slab: int, number of prim_slab atoms (N)
        - n_ads: int, number of adsorbate atoms (M)

    Output:
        - prim_slab_r_update: (B, N, 3) prim_slab coordinate updates
        - ads_r_update: (B, M, 3) adsorbate coordinate updates
        - prim_virtual_update: (B, 3, 3) primitive virtual coordinate updates
        - supercell_virtual_update: (B, 3, 3) supercell virtual coordinate updates
        - sf_update: (B,) scaling factor updates
    """

    # Number of virtual atoms (3 primitive + 3 supercell)
    NUM_VIRTUAL_ATOMS = 6

    def __init__(
        self,
        atom_s,
        atom_decoder_depth,
        atom_decoder_heads,
        attention_impl,
        activation_checkpointing=False,
        dng: bool = False,
    ):
        super().__init__()
        self.atom_decoder = DiT(
            dim=atom_s,
            depth=atom_decoder_depth,
            heads=atom_decoder_heads,
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing,
        )

        # Projection heads for prim_slab coords (per-atom)
        self.feats_to_prim_slab_coords = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 3)
        )

        # Projection heads for adsorbate coords (per-atom, direct prediction)
        self.feats_to_ads_coords = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 3)
        )

        # Projection heads for primitive virtual coords (per-atom)
        self.feats_to_prim_virtual_coords = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 3)
        )

        # Projection heads for supercell virtual coords (per-atom)
        self.feats_to_supercell_virtual_coords = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 3)
        )

        # Scaling factor output is 1-dimensional (scalar) - from global pooling
        self.feats_to_scaling_factor = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 1)
        )

        # Output scales for tanh bounding (prevents gradient explosion)
        # These are applied to normalized outputs before denormalization
        self.coord_output_scale = 3.0  # coords: tanh * 3 -> [-3, 3] normalized
        self.virtual_coord_output_scale = 3.0  # virtual coords: tanh * 3 -> [-3, 3] normalized
        self.scaling_factor_output_scale = 3.0  # scaling factor: tanh * 3 -> [-3, 3] normalized

        # Add element prediction head when dng=True
        if dng:
            # Output logits (softmax is applied during loss/vector field computation)
            self.feats_to_prim_slab_element = nn.Sequential(
                nn.LayerNorm(atom_s),
                LinearNoBias(atom_s, NUM_ELEMENTS)  # output logits
            )

    def forward(self, x, t, feats, n_prim_slab, n_ads=None, multiplicity=1):
        """
        Args:
            x: (B*mult, N+M+6, atom_s) joint atom representations (including virtual atoms)
            t: (B*mult,) timesteps
            feats: feature dictionary (not yet repeated for multiplicity)
            n_prim_slab: int, number of prim_slab atoms (N)
            n_ads: int, optional, number of adsorbate atoms (M). If None, inferred from feats
            multiplicity: number of samples per input

        Returns:
            Dictionary with coordinate predictions for all atom types
        """
        prim_slab_mask = feats["prim_slab_atom_pad_mask"]
        ads_mask = feats["ads_atom_pad_mask"]

        prim_slab_mask = prim_slab_mask.repeat_interleave(multiplicity, 0)
        ads_mask = ads_mask.repeat_interleave(multiplicity, 0)

        # Get number of adsorbate atoms
        if n_ads is None:
            n_ads = ads_mask.shape[1]

        B_mult = x.shape[0]
        device = x.device

        # Virtual atoms always valid (no padding)
        virtual_mask = torch.ones(B_mult, self.NUM_VIRTUAL_ATOMS, device=device, dtype=prim_slab_mask.dtype)
        joint_mask = torch.cat([prim_slab_mask, ads_mask, virtual_mask], dim=1)

        x = self.atom_decoder(x, t, joint_mask)

        # Split back to prim_slab, adsorbate, and virtual atoms
        x_prim_slab = x[:, :n_prim_slab, :]  # (B*mult, N, atom_s)
        x_ads = x[:, n_prim_slab:n_prim_slab + n_ads, :]  # (B*mult, M, atom_s)
        x_prim_virtual = x[:, n_prim_slab + n_ads:n_prim_slab + n_ads + 3, :]  # (B*mult, 3, atom_s)
        x_supercell_virtual = x[:, n_prim_slab + n_ads + 3:, :]  # (B*mult, 3, atom_s)

        # Prim slab position prediction (bounded to prevent explosion)
        prim_slab_r_update = self.feats_to_prim_slab_coords(x_prim_slab)
        prim_slab_r_update = torch.tanh(prim_slab_r_update) * self.coord_output_scale

        # Adsorbate position prediction (per-atom, direct, bounded)
        ads_r_update = self.feats_to_ads_coords(x_ads)  # (B*mult, M, 3)
        ads_r_update = torch.tanh(ads_r_update) * self.coord_output_scale

        # Primitive virtual coords prediction (per-atom, bounded)
        prim_virtual_update = self.feats_to_prim_virtual_coords(x_prim_virtual)  # (B*mult, 3, 3)
        prim_virtual_update = torch.tanh(prim_virtual_update) * self.virtual_coord_output_scale

        # Supercell virtual coords prediction (per-atom, bounded)
        supercell_virtual_update = self.feats_to_supercell_virtual_coords(x_supercell_virtual)  # (B*mult, 3, 3)
        supercell_virtual_update = torch.tanh(supercell_virtual_update) * self.virtual_coord_output_scale

        # Global pooling (use prim_slab atoms only for scaling factor)
        num_prim_slab_atoms = prim_slab_mask.sum(dim=1, keepdim=True)  # (B*mult, 1)
        x_global = torch.sum(x_prim_slab * prim_slab_mask[..., None], dim=1) / (num_prim_slab_atoms + EPS_NUMERICAL)

        # Scaling factor prediction (bounded to prevent explosion)
        sf_update = self.feats_to_scaling_factor(x_global).squeeze(-1)  # (B*mult,)
        sf_update = torch.tanh(sf_update) * self.scaling_factor_output_scale

        # Add element prediction when dng=True
        if hasattr(self, 'feats_to_prim_slab_element'):
            prim_slab_element_update = self.feats_to_prim_slab_element(x_prim_slab)
            # (B*mult, N, NUM_ELEMENTS) - logits format (softmax is applied during loss/vector field computation)
            return {
                "prim_slab_r_update": prim_slab_r_update,
                "ads_r_update": ads_r_update,
                "prim_virtual_update": prim_virtual_update,
                "supercell_virtual_update": supercell_virtual_update,
                "sf_update": sf_update,
                "prim_slab_element_update": prim_slab_element_update,
            }
        else:
            return {
                "prim_slab_r_update": prim_slab_r_update,
                "ads_r_update": ads_r_update,
                "prim_virtual_update": prim_virtual_update,
                "supercell_virtual_update": supercell_virtual_update,
                "sf_update": sf_update,
            }


class TokenTransformer(Module):
    """
    Token-level transformer for joint prim_slab and adsorbate tokens.
    """
    
    def __init__(
        self,
        token_s,
        token_transformer_depth,
        token_transformer_heads,
        attention_impl,
        activation_checkpointing=False,
    ):
        super().__init__()

        self.s_init = nn.Linear(token_s, token_s, bias=False)

        # Normalization layers
        self.s_mlp = nn.Sequential(
            nn.LayerNorm(token_s), nn.Linear(token_s, token_s, bias=False)
        )

        self.token_transformer = DiT(
            dim=token_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(self, x, t, feats, multiplicity=1):
        # Joint token mask (prim_slab + adsorbate + 6 virtual tokens)
        prim_slab_token_mask = feats["prim_slab_token_pad_mask"]
        ads_token_mask = feats["ads_token_pad_mask"]

        # Create virtual token mask (always True, virtual atoms are never padded)
        B = prim_slab_token_mask.shape[0]
        NUM_VIRTUAL_TOKENS = 6  # 3 primitive + 3 supercell virtual atoms (each is its own token)
        virtual_token_mask = torch.ones(B, NUM_VIRTUAL_TOKENS, dtype=torch.bool, device=prim_slab_token_mask.device)

        # Concatenate all token masks
        token_mask = torch.cat([prim_slab_token_mask, ads_token_mask, virtual_token_mask], dim=1)
        token_mask = token_mask.repeat_interleave(multiplicity, 0)

        # Initialize single embeddings
        s_init = self.s_init(x)  # (batch_size, num_tokens, token_s)

        s = self.s_mlp(s_init)
        x = self.token_transformer(s, t, token_mask)

        return x
