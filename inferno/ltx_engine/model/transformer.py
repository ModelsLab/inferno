from dataclasses import dataclass
import os
import glob
import json
import math
import logging
import numbers
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import numpy as np
from safetensors import safe_open
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.utils import TRANSFORMER_KEYS_RENAME_DICT, TimestepEmbedding, make_hashable_key, diffusers_and_inferno_config_mapping

try:
    from torch._dynamo import allow_in_graph as maybe_allow_in_graph
except (ImportError, ModuleNotFoundError):

    def maybe_allow_in_graph(cls):
        return cls

try:
    from torch_xla.experimental.custom_kernel import flash_attention
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

logger = logging.getLogger(__name__)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False):
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        self.weight = None
        self.bias = None

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class GELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate, approximate=self.approximate)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states



class GEGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate)

    def forward(self, hidden_states, *args, **kwargs):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


def get_3d_sincos_pos_embed(embed_dim, grid, w, h, f):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = rearrange(grid, "c (f h w) -> c f h w", h=h, w=w)
    grid = rearrange(grid, "c f h w -> c h w f", h=h, w=w)
    grid = grid.reshape([3, 1, w, h, f])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = pos_embed.transpose(1, 0, 2, 3)
    return rearrange(pos_embed, "h w f c -> (f h w) c")


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 3 != 0:
        raise ValueError("embed_dim must be divisible by 3")

    # use half of dimensions to encode grid_h
    emb_f = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W*T, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W*T, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W*T, D/3)

    emb = np.concatenate([emb_h, emb_w, emb_f], axis=-1)  # (H*W*T, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos_shape = pos.shape

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    out = out.reshape([*pos_shape, -1])[0]

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (M, D)
    return emb


class SinusoidalPositionalEmbedding(nn.Module):
    """Apply positional information to a sequence of embeddings.
    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them
    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings
    """

    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        _, seq_length, _ = x.shape
        x = x + self.pe[:, :seq_length]
        return x


def _chunked_feed_forward(ff_module, hidden_states, chunk_dim: int, chunk_size: int):
    """
    Computes feed forward layer outputs in chunks to save memory.
    """
    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    chunk_output = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        if chunk_dim == 0:
            chunk = hidden_states[start_idx:end_idx]
        elif chunk_dim == 1:
            chunk = hidden_states[:, start_idx:end_idx]
        else:
            raise ValueError(f"Unsupported chunk dimension {chunk_dim}")
        
        chunk_output.append(ff_module(chunk))

    if chunk_dim == 0:
        return torch.cat(chunk_output, dim=0)
    elif chunk_dim == 1:
        return torch.cat(chunk_output, dim=1)
    
def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


class FeedForward(nn.Module):
    """
    A feed-forward layer with flexible activation functions.
    """
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        inner_dim = inner_dim if inner_dim is not None else int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        # Choose activation function
        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        self.net = nn.ModuleList([
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=bias)
        ])

        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        for i, module in enumerate(self.net):
            if isinstance(module, (GEGLU, GELU)):
                hidden_states = module(hidden_states, scale)
            else:
                hidden_states = module(hidden_states)
        return hidden_states

class SpatialNorm(nn.Module):
    """
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f
    
class Attention(nn.Module):
    """
    Cross attention module supporting various attention mechanisms.
    """
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        qk_norm: Optional[str] = None,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        out_dim: Optional[int] = None,
        use_tpu_flash_attention: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5 if scale_qk else 1.0
        self.dropout = dropout
        self.use_tpu_flash_attention = use_tpu_flash_attention and XLA_AVAILABLE
        self.use_rope = use_rope
        
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.out_dim = out_dim if out_dim is not None else query_dim

        # Initialize query, key, value projections
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        if not only_cross_attention:
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None
            if added_kv_proj_dim is None:
                raise ValueError("added_kv_proj_dim must be defined when only_cross_attention is True")

        # Initialize normalization layers
        if qk_norm == "layer_norm":
            self.q_norm = nn.LayerNorm(self.inner_dim, eps=eps)
            self.k_norm = nn.LayerNorm(self.inner_dim, eps=eps)
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(self.inner_dim, eps=eps)
            self.k_norm = RMSNorm(self.inner_dim, eps=eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Optional group normalization
        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(
                num_channels=query_dim,
                num_groups=norm_num_groups,
                eps=eps,
                affine=True
            )
        else:
            self.group_norm = None

        # Optional spatial normalization
        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(query_dim, spatial_norm_dim)
        else:
            self.spatial_norm = None

        # Optional cross attention normalization
        if cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            channels = added_kv_proj_dim if added_kv_proj_dim is not None else self.cross_attention_dim
            self.norm_cross = nn.GroupNorm(
                num_channels=channels,
                num_groups=cross_attention_norm_num_groups,
                eps=eps,
                affine=True
            )
        else:
            self.norm_cross = None

        # Output projection
        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, self.out_dim, bias=out_bias),
            nn.Dropout(dropout)
        ])

        # Additional key/value projections if needed
        if added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
        else:
            self.add_k_proj = None
            self.add_v_proj = None

    def reshape_heads_to_batch_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, self.heads, self.dim_head)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(batch_size * self.heads, seq_len, self.dim_head)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // self.heads, self.heads, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(batch_size // self.heads, seq_len, dim * self.heads)
        return tensor

    def get_attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(query.dtype)

        return attention_probs

    def prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        target_length: int,
        batch_size: int
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None

        if attention_mask.shape[-1] != target_length:
            if attention_mask.device.type == "mps":
                # Handle MPS padding limitation
                padding_shape = (*attention_mask.shape[:-1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=-1)
            else:
                attention_mask = F.pad(attention_mask, (0, target_length - attention_mask.shape[-1]))

        head_size = self.heads
        attention_mask = attention_mask.view(batch_size, 1, -1, attention_mask.shape[-1])
        attention_mask = attention_mask.repeat_interleave(head_size, dim=1)
        return attention_mask

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        freqs_cis: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of attention module.
        """
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        residual = hidden_states if self.residual_connection else None

        if self.spatial_norm is not None:
            hidden_states = self.spatial_norm(hidden_states, temb)

        # Project query, key, value
        query = self.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.norm_cross is not None:
            if isinstance(self.norm_cross, nn.LayerNorm):
                encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            else:
                encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
                encoder_hidden_states = self.norm_cross(encoder_hidden_states)
                encoder_hidden_states = encoder_hidden_states.transpose(1, 2)

        key = self.to_k(encoder_hidden_states) if self.to_k is not None else None
        value = self.to_v(encoder_hidden_states) if self.to_v is not None else None

        if self.group_norm is not None:
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = self.group_norm(hidden_states)
            hidden_states = hidden_states.transpose(1, 2)

        # Apply rotary embeddings if used
        if self.use_rope and freqs_cis is not None:
            cos_sin = freqs_cis
            query = self.apply_rotary_emb(query, cos_sin)
            if key is not None:
                key = self.apply_rotary_emb(key, cos_sin)

        query = self.q_norm(query)
        key = self.k_norm(key) if key is not None else None

        # Handle dimensions
        inner_dim = key.shape[-1] if key is not None else query.shape[-1]
        head_dim = inner_dim // self.heads

        # Shape for attention computation
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        if key is not None:
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        if value is not None:
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # Prepare attention mask if needed
        if attention_mask is not None and not self.use_tpu_flash_attention:
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # Compute attention
        if self.use_tpu_flash_attention and XLA_AVAILABLE:
            # TPU Flash Attention path
            q_segment_ids = None
            if attention_mask is not None:
                attention_mask = attention_mask.to(torch.float32)
                q_segment_ids = torch.ones(batch_size, query.shape[2], device=query.device, dtype=torch.float32)
                if key is not None:
                    assert attention_mask.shape[1] == key.shape[2], f"Key shape mismatch: {key.shape[2]} vs {attention_mask.shape[1]}"

            # TPU limitations checks
            if query.shape[2] % 128 != 0 or (key is not None and key.shape[2] % 128 != 0):
                raise ValueError("Query and key sequence lengths must be divisible by 128 for TPU")

            hidden_states = flash_attention(
                q=query,
                k=key if key is not None else query,
                v=value if value is not None else query,
                q_segment_ids=q_segment_ids,
                kv_segment_ids=attention_mask,
                sm_scale=self.scale,
            )
        else:
            # Standard attention path
            attention_probs = self.get_attention_scores(
                query,
                key if key is not None else query,
                attention_mask
            )
            hidden_states = torch.bmm(attention_probs, value if value is not None else query)

        # Reshape and project output
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        # Apply residual connection if needed
        if residual is not None:
            hidden_states = hidden_states + residual

        # Rescale output if needed
        if self.rescale_output_factor != 1.0:
            hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states

    @staticmethod
    def apply_rotary_emb(
        x: torch.Tensor, 
        freqs_cis: Tuple[torch.FloatTensor, torch.FloatTensor]
    ) -> torch.Tensor:
        """Apply rotary embeddings to input tensor."""
        cos, sin = freqs_cis
        x_duplicate = rearrange(x, '... (d r) -> ... d r', r=2)
        x1, x2 = x_duplicate.unbind(dim=-1)
        x_rotated = torch.stack([-x2, x1], dim=-1)
        x_rotated = rearrange(x_rotated, '... d r -> ... (d r)')
        return x * cos + x_rotated * sin

@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    """
    A basic Transformer block with self-attention, cross-attention, and feed-forward layers.
    """
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        adaptive_norm: str = "single_scale_shift",
        standardization_norm: str = "layer_norm",
        norm_eps: float = 1e-5,
        qk_norm: Optional[str] = None,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        use_tpu_flash_attention: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        
        self.only_cross_attention = only_cross_attention
        self.use_tpu_flash_attention = use_tpu_flash_attention
        self.adaptive_norm = adaptive_norm

        # Validate norm types
        if standardization_norm not in ["layer_norm", "rms_norm"]:
            raise ValueError(f"Unknown standardization norm: {standardization_norm}")
        if adaptive_norm not in ["single_scale_shift", "single_scale", "none"]:
            raise ValueError(f"Unknown adaptive norm: {adaptive_norm}")

        # Create normalization layer based on type
        norm_layer = nn.LayerNorm if standardization_norm == "layer_norm" else RMSNorm

        # 1. Self-Attention block
        self.norm1 = norm_layer(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            use_tpu_flash_attention=use_tpu_flash_attention,
            qk_norm=qk_norm,
            use_rope=use_rope
        )

        # 2. Cross-Attention block (if needed)
        if cross_attention_dim is not None or double_self_attention:
            cross_attention_dim = None if double_self_attention else cross_attention_dim
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                use_tpu_flash_attention=use_tpu_flash_attention,
                qk_norm=qk_norm,
                use_rope=use_rope
            )
            
            # Add norm layer for cross-attention if needed
            if adaptive_norm == "none":
                self.attn2_norm = norm_layer(
                    dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine
                )
            else:
                self.attn2_norm = None
        else:
            self.attn2 = None
            self.attn2_norm = None

        # 3. Feed-forward block
        self.norm2 = norm_layer(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias
        )

        # 4. Adaptive parameters for PixArt-Alpha
        if adaptive_norm != "none":
            num_ada_params = 4 if adaptive_norm == "single_scale" else 6
            self.scale_shift_table = nn.Parameter(torch.randn(num_ada_params, dim) / dim**0.5)

        # Chunking parameters
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], chunk_dim: int = 0):
        """Configure chunked feed-forward computation."""
        self._chunk_size = chunk_size
        self._chunk_dim = chunk_dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        freqs_cis: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None
    ) -> torch.FloatTensor:
        """
        Forward pass of transformer block.
        """
        # Process input through self-attention
        batch_size = hidden_states.shape[0]
        norm_hidden_states = self.norm1(hidden_states)

        # Handle adaptive normalization
        if self.adaptive_norm in ["single_scale_shift", "single_scale"]:
            if timestep is None or timestep.ndim != 3:
                raise ValueError("Timestep must be provided for adaptive norm")

            num_ada_params = self.scale_shift_table.shape[0]
            ada_params = self.scale_shift_table[None, None] + timestep.reshape(
                batch_size, timestep.shape[1], num_ada_params, -1
            )

            if self.adaptive_norm == "single_scale_shift":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_params.unbind(dim=2)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            else:
                scale_msa, gate_msa, scale_mlp, gate_mlp = ada_params.unbind(dim=2)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa)
        else:
            scale_msa = gate_msa = scale_mlp = gate_mlp = None
            shift_msa = shift_mlp = None

        # Remove extra dimension if present
        norm_hidden_states = norm_hidden_states.squeeze(1)

        # Self-attention
        cross_attention_kwargs = cross_attention_kwargs or {}
        attn_output = self.attn1(
            norm_hidden_states,
            freqs_cis=freqs_cis,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs
        )

        if gate_msa is not None:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        hidden_states = hidden_states.squeeze(1)

        # Cross-attention block
        if self.attn2 is not None:
            attn_input = self.attn2_norm(hidden_states) if self.adaptive_norm == "none" else hidden_states
            attn_output = self.attn2(
                attn_input,
                freqs_cis=freqs_cis,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # Feed-forward block
        norm_hidden_states = self.norm2(hidden_states)

        # Apply adaptive normalization to feed-forward input if needed
        if self.adaptive_norm == "single_scale_shift":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        elif self.adaptive_norm == "single_scale":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp)

        # Handle chunked feed-forward if configured
        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(
                self.ff, 
                norm_hidden_states, 
                self._chunk_dim, 
                self._chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        # Apply gating to feed-forward output if used
        if gate_mlp is not None:
            ff_output = gate_mlp * ff_output

        # Residual connection with feed-forward output
        hidden_states = ff_output + hidden_states

        # Remove extra dimension if present
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


@dataclass
class Transformer3DModelOutput:
    """Output of Transformer3DModel."""
    sample: torch.FloatTensor


class AdaLayerNormSingle(nn.Module):
    """
    Adaptive Layer Norm that supports single conditioning.
    """
    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False):
        super().__init__()
        self.emb = TimestepEmbedding(embedding_dim)
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.use_additional_conditions = use_additional_conditions

    def forward(
        self,
        timestep: torch.Tensor,
        conditioning_embedding: Dict[str, torch.Tensor],
        batch_size: int,
        hidden_dtype: Optional[torch.dtype] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, batch_size, hidden_dtype)
        return timestep, embedded_timestep


class PixArtAlphaTextProjection(nn.Module):
    """
    Transforms text embeddings for conditioning.
    """
    def __init__(self, in_features: int, hidden_size: int, num_hidden_layers: int = 2):
        super().__init__()
        self.linear_layers = nn.ModuleList([])
        
        for i in range(num_hidden_layers):
            in_dim = in_features if i == 0 else hidden_size
            self.linear_layers.append(nn.Linear(in_dim, hidden_size))
            if i != num_hidden_layers - 1:
                self.linear_layers.append(nn.SiLU())

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        for layer in self.linear_layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class Transformer3DModel(ModelMixin, ConfigMixin):
    """
    Transformer model for 3D data with support for attention and cross-attention.
    """
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        num_vector_embeds: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        adaptive_norm: str = "single_scale_shift",
        standardization_norm: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: Optional[int] = None,
        project_to_2d_pos: bool = False,
        use_tpu_flash_attention: bool = False,
        qk_norm: Optional[str] = None,
        positional_embedding_type: str = "absolute",
        positional_embedding_theta: Optional[float] = None,
        positional_embedding_max_pos: Optional[List[int]] = None,
        timestep_scale_multiplier: Optional[float] = None,
    ):
        super().__init__()
        
        self.use_tpu_flash_attention = use_tpu_flash_attention
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.project_to_2d_pos = project_to_2d_pos

        # Input projection
        self.patchify_proj = nn.Linear(in_channels, self.inner_dim, bias=True)

        # Positional embedding configuration
        self.positional_embedding_type = positional_embedding_type
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.use_rope = self.positional_embedding_type == "rope"
        self.timestep_scale_multiplier = timestep_scale_multiplier

        # Handle 2D projection for positional embeddings
        if self.positional_embedding_type == "absolute":
            embed_dim_3d = math.ceil((self.inner_dim / 2) * 3) if project_to_2d_pos else self.inner_dim
            if self.project_to_2d_pos:
                self.to_2d_proj = nn.Linear(embed_dim_3d, self.inner_dim, bias=False)
                self._init_to_2d_proj_weights(self.to_2d_proj)

        # Initialize transformer blocks
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                only_cross_attention=only_cross_attention,
                double_self_attention=double_self_attention,
                upcast_attention=upcast_attention,
                adaptive_norm=adaptive_norm,
                standardization_norm=standardization_norm,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps,
                use_tpu_flash_attention=use_tpu_flash_attention,
                qk_norm=qk_norm,
                use_rope=self.use_rope,
            ) for _ in range(num_layers)
        ])

        # Output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels)

        # Adaptive normalization
        self.adaln_single = AdaLayerNormSingle(self.inner_dim, use_additional_conditions=False)
        if adaptive_norm == "single_scale":
            self.adaln_single.linear = nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=True)

        # Caption processing
        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels,
                hidden_size=self.inner_dim
            )

        self.gradient_checkpointing = False

    def set_use_tpu_flash_attention(self):
        """Enable TPU flash attention for all transformer blocks."""
        self.use_tpu_flash_attention = True
        for block in self.transformer_blocks:
            block.set_use_tpu_flash_attention()

    @staticmethod
    def _init_to_2d_proj_weights(linear_layer: nn.Linear):
        """Initialize the weights for 2D projection."""
        input_features = linear_layer.weight.data.size(1)
        output_features = linear_layer.weight.data.size(0)

        identity_like = torch.zeros((output_features, input_features))
        min_features = min(output_features, input_features)
        identity_like[:min_features, :min_features] = torch.eye(min_features)
        linear_layer.weight.data = identity_like.to(linear_layer.weight.data.device)

    def get_fractional_positions(self, indices_grid: torch.Tensor) -> torch.Tensor:
        """Compute fractional positions from indices grid."""
        return torch.stack([
            indices_grid[:, i] / self.positional_embedding_max_pos[i]
            for i in range(3)
        ], dim=-1)

    def precompute_freqs_cis(
        self,
        indices_grid: torch.Tensor,
        spacing: str = "exp"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute frequency components for rotary embeddings.
        """
        dtype = torch.float32
        dim = self.inner_dim
        theta = self.positional_embedding_theta
        device = indices_grid.device

        fractional_positions = self.get_fractional_positions(indices_grid)
        
        # Generate base frequencies based on spacing type
        if spacing == "exp":
            indices = theta ** torch.linspace(
                math.log(1, theta),
                math.log(theta, theta),
                dim // 6,
                device=device,
                dtype=dtype
            )
        elif spacing == "exp_2":
            indices = 1.0 / theta ** (torch.arange(0, dim, 6, device=device) / dim)
        elif spacing == "linear":
            indices = torch.linspace(1, theta, dim // 6, device=device, dtype=dtype)
        elif spacing == "sqrt":
            indices = torch.linspace(1, theta**2, dim // 6, device=device, dtype=dtype).sqrt()
        
        indices = indices * math.pi / 2

        # Compute frequencies
        if spacing == "exp_2":
            freqs = (indices * fractional_positions.unsqueeze(-1)).transpose(-1, -2).flatten(2)
        else:
            freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)

        # Generate cos and sin components
        cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freq = freqs.sin().repeat_interleave(2, dim=-1)

        # Handle padding if needed
        if dim % 6 != 0:
            padding_size = dim % 6
            cos_padding = torch.ones_like(cos_freq[:, :, :padding_size])
            sin_padding = torch.zeros_like(sin_freq[:, :, :padding_size])
            cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
            sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)

        return cos_freq.to(dtype), sin_freq.to(dtype)

    def initialize(self, embedding_std: float, mode: Literal["xora", "legacy"]):
        """Initialize model weights with specified standard deviation."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(
            self.adaln_single.emb.timestep_embedder.linear_1.weight,
            std=embedding_std
        )
        nn.init.normal_(
            self.adaln_single.emb.timestep_embedder.linear_2.weight,
            std=embedding_std
        )
        nn.init.normal_(
            self.adaln_single.linear.weight,
            std=embedding_std
        )

        # Initialize caption embedding MLP if present
        if self.caption_projection is not None:
            for layer in self.caption_projection.linear_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=embedding_std)

        # Initialize transformer blocks
        for block in self.transformer_blocks:
            if mode.lower() == "xora":
                nn.init.constant_(block.attn1.to_out[0].weight, 0)
                nn.init.constant_(block.attn1.to_out[0].bias, 0)

            nn.init.constant_(block.attn2.to_out[0].weight, 0)
            nn.init.constant_(block.attn2.to_out[0].bias, 0)

            if mode.lower() == "xora":
                nn.init.constant_(block.ff.net[2].weight, 0)
                nn.init.constant_(block.ff.net[2].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def load_state_dict(
        self,
        state_dict: Dict,
        *args,
        **kwargs,
    ):
        if any([key.startswith("model.diffusion_model.") for key in state_dict.keys()]):
            state_dict = {
                key.replace("model.diffusion_model.", ""): value
                for key, value in state_dict.items()
                if key.startswith("model.diffusion_model.")
            }
        super().load_state_dict(state_dict, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[Union[str, os.PathLike]],
        *args,
        **kwargs,
    ):
        pretrained_model_path = Path(pretrained_model_path)
        if pretrained_model_path.is_dir():
            config_path = pretrained_model_path / "transformer" / "config.json"
            with open(config_path, "r") as f:
                config = make_hashable_key(json.load(f))

            assert config in diffusers_and_inferno_config_mapping, (
                "Provided diffusers checkpoint config for transformer is not suppported. "
                "We only support diffusers configs found in Lightricks/LTX-Video."
            )

            config = diffusers_and_inferno_config_mapping[config]
            state_dict = {}
            ckpt_paths = (
                pretrained_model_path
                / "transformer"
                / "diffusion_pytorch_model*.safetensors"
            )
            dict_list = glob.glob(str(ckpt_paths))
            for dict_path in dict_list:
                part_dict = {}
                with safe_open(dict_path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        part_dict[k] = f.get_tensor(k)
                state_dict.update(part_dict)

            for key in list(state_dict.keys()):
                new_key = key
                for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
                    new_key = new_key.replace(replace_key, rename_key)
                state_dict[new_key] = state_dict.pop(key)

            transformer = cls.from_config(config)
            transformer.load_state_dict(state_dict, strict=True)
        elif pretrained_model_path.is_file() and str(pretrained_model_path).endswith(
            ".safetensors"
        ):
            comfy_single_file_state_dict = {}
            with safe_open(pretrained_model_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                for k in f.keys():
                    comfy_single_file_state_dict[k] = f.get_tensor(k)
            configs = json.loads(metadata["config"])
            transformer_config = configs["transformer"]
            transformer = Transformer3DModel.from_config(transformer_config)
            transformer.load_state_dict(comfy_single_file_state_dict)
        return transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        indices_grid: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Transformer3DModelOutput, Tuple]:
        """
        Forward pass of the model.
        
        Args:
            hidden_states: Input tensor
            indices_grid: Grid of positional indices
            encoder_hidden_states: Optional context for cross-attention
            timestep: Optional timestep for conditioning
            class_labels: Optional class labels for conditioning
            cross_attention_kwargs: Optional kwargs for cross attention
            attention_mask: Optional attention mask
            encoder_attention_mask: Optional mask for encoder states
            return_dict: Whether to return output as a dataclass
        """
        # Handle attention masks for TPU
        if not self.use_tpu_flash_attention:
            if attention_mask is not None and attention_mask.ndim == 2:
                attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (
                    1 - encoder_attention_mask.to(hidden_states.dtype)
                ) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # Project input
        hidden_states = self.patchify_proj(hidden_states)

        # Scale timestep if needed
        if self.timestep_scale_multiplier and timestep is not None:
            timestep = self.timestep_scale_multiplier * timestep

        # Handle positional embeddings
        if self.positional_embedding_type == "absolute":
            pos_embed_3d = self.get_absolute_pos_embed(indices_grid).to(hidden_states.device)
            if self.project_to_2d_pos:
                pos_embed = self.to_2d_proj(pos_embed_3d)
            hidden_states = (hidden_states + pos_embed).to(hidden_states.dtype)
            freqs_cis = None
        elif self.positional_embedding_type == "rope":
            freqs_cis = self.precompute_freqs_cis(indices_grid)

        # Process timestep embedding
        batch_size = hidden_states.shape[0]
        timestep, embedded_timestep = self.adaln_single(
            timestep.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        
        # Reshape timestep tensors
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])

        # Project caption/context if needed
        if self.caption_projection is not None and encoder_hidden_states is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        # Process transformer blocks
        cross_attention_kwargs = cross_attention_kwargs or {}
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    freqs_cis,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    use_reentrant=False
                )
            else:
                hidden_states = block(
                    hidden_states,
                    freqs_cis=freqs_cis,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # Output processing
        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)

        if not return_dict:
            return (hidden_states,)

        return Transformer3DModelOutput(sample=hidden_states)

    def get_absolute_pos_embed(self, grid: torch.Tensor) -> torch.Tensor:
        """Get absolute positional embeddings from grid."""
        grid_np = grid[0].cpu().numpy()
        embed_dim_3d = (
            math.ceil((self.inner_dim / 2) * 3)
            if self.project_to_2d_pos
            else self.inner_dim
        )
        pos_embed = get_3d_sincos_pos_embed(
            embed_dim_3d,
            grid_np,
            h=int(max(grid_np[1]) + 1),
            w=int(max(grid_np[2]) + 1),
            f=int(max(grid_np[0] + 1)),
        )
        return torch.from_numpy(pos_embed).float().unsqueeze(0)