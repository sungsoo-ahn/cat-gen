import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer normalization modulation.

    Modulates input x using learned shift and scale parameters:
        output = x * (1 + scale) + shift

    Args:
        x: Input tensor of shape (B, N, D)
        shift: Shift parameter of shape (B, D)
        scale: Scale parameter of shape (B, D)

    Returns:
        Modulated tensor of shape (B, N, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                         Embedding Layers for Timesteps                        #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_dim, frequency_embedding_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        # t: Float['b] -> t_emb: Float['b d]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


class PositionalEmbedder(nn.Module):
    """Embeds integer indices into vector representations using sine/cosine positional encoding."""

    def __init__(self, hidden_dim, frequency_embedding_dim=256, max_len=2048):
        super().__init__()

        if frequency_embedding_dim % 2 != 0:
            raise ValueError(
                f"frequency_embedding_dim must be even, but got {frequency_embedding_dim}"
            )

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim
        self.max_len = max_len

    @staticmethod
    def _create_positional_embedding(indices, dim, max_len=2048):
        """Creates the core sine/cosine positional embedding."""
        K = torch.arange(dim // 2, device=indices.device)
        div_term = max_len ** (2 * K / dim)
        angles = indices[..., None] * math.pi / div_term
        pos_embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return pos_embedding

    def forward(self, indices):
        # indices: Long['b n'] -> pos_emb: Float['b n d']
        pos_freq = self._create_positional_embedding(
            indices, self.frequency_embedding_dim, self.max_len
        )
        pos_emb = self.mlp(pos_freq)
        return pos_emb


class ScaledDotProductAttention(nn.Module):
    """Multi-head attention using PyTorch's scaled_dot_product_attention.

    This automatically uses Flash Attention when available (PyTorch 2.0+).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional key padding mask.

        Args:
            x: Input tensor of shape (B, N, D)
            key_padding_mask: Boolean mask of shape (B, N) where True = ignore position

        Returns:
            Output tensor of shape (B, N, D)
        """
        B, N, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Create attention mask from key_padding_mask
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, N) True = ignore
            # Convert to attention mask: (B, 1, 1, N) -> broadcasts to (B, H, N, N)
            # Where True positions get -inf attention
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn_mask = attn_mask.expand(-1, -1, N, -1)  # (B, 1, N, N)
            attn_mask = torch.where(attn_mask, float('-inf'), 0.0)

        # Use scaled_dot_product_attention (Flash Attention when available)
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

        # Reshape and project output
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        return out


class MLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

AttentionImpl = Literal["manual", "pytorch", "xformers", "flash"]


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(
        self,
        heads,
        dim,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_impl: AttentionImpl = "pytorch",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention_impl = attention_impl

        # Select attention implementation
        if attention_impl == "flash":
            self.attn = ScaledDotProductAttention(
                embed_dim=dim, num_heads=heads, dropout=dropout, bias=True
            )
        else:
            # Default: PyTorch MultiheadAttention
            self.attn = nn.MultiheadAttention(
                dim, num_heads=heads, dropout=dropout, bias=True, batch_first=True
            )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)

        def _approx_gelu() -> nn.Module:
            """Create GELU activation with tanh approximation (faster than exact)."""
            return nn.GELU(approximate="tanh")

        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=_approx_gelu,
            drop=dropout,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT encoder blocks:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, mask):
        # Generate modulation parameters (shift, scale gate) from condition c
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # Attention block
        _x = modulate(self.norm1(x), shift_msa, scale_msa)
        # key_padding_mask: True = ignore position. Input mask is True = valid, so invert.
        key_padding_mask = ~mask.bool() if mask is not None else None

        if self.attention_impl == "flash":
            # Flash attention via ScaledDotProductAttention
            attn_out = self.attn(_x, key_padding_mask=key_padding_mask)
        else:
            # Standard PyTorch MultiheadAttention
            attn_out = self.attn(_x, _x, _x, key_padding_mask=key_padding_mask, need_weights=False)[0]

        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP block
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class DiT(nn.Module):
    """Transformer with DiT blocks"""

    def __init__(
        self,
        depth,
        heads,
        dim,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_impl: AttentionImpl = "pytorch",
        activation_checkpointing=False,
    ):
        super().__init__()

        self.activation_checkpointing = activation_checkpointing
        self.t_embedder = TimestepEmbedder(dim)
        self.layers = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_impl=attention_impl,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, t, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D)
            t (torch.Tensor): Time step for each sample (B,)
            mask (torch.Tensor): True if valid token, False if padding (B, N)
        """
        # Embed t
        time_embed = self.t_embedder(t)  # (B, D)

        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, time_embed, mask, use_reentrant=False
                )
            else:
                x = layer(x, time_embed, mask)
        return x
