from dataclasses import dataclass
import inspect
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
from diffusers.utils import BaseOutput, is_torch_version
import numpy as np
from safetensors import safe_open
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.utils import TRANSFORMER_KEYS_RENAME_DICT, PixArtAlphaCombinedTimestepSizeEmbeddings, TimestepEmbedding, make_hashable_key, diffusers_and_inferno_config_mapping, SkipLayerStrategy

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
    
@maybe_allow_in_graph
class Attention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        qk_norm (`str`, *optional*, defaults to None):
            Set to 'layer_norm' or `rms_norm` to perform query and key normalization.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
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
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor2_0"] = None,
        out_dim: int = None,
        use_tpu_flash_attention: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.use_tpu_flash_attention = use_tpu_flash_attention
        self.use_rope = use_rope

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        if qk_norm is None:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head * heads, eps=1e-5)
            self.k_norm = RMSNorm(dim_head * heads, eps=1e-5)
        elif qk_norm == "layer_norm":
            self.q_norm = nn.LayerNorm(dim_head * heads, eps=1e-5)
            self.k_norm = nn.LayerNorm(dim_head * heads, eps=1e-5)
        else:
            raise ValueError(f"Unsupported qk_norm method: {qk_norm}")

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(
                num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True
            )
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(
                f_channels=query_dim, zq_channels=spatial_norm_dim
            )
        else:
            self.spatial_norm = None

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels,
                num_groups=cross_attention_norm_num_groups,
                eps=1e-5,
                affine=True,
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        linear_cls = nn.Linear

        self.linear_cls = linear_cls
        self.to_q = linear_cls(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = linear_cls(self.cross_attention_dim, self.inner_dim, bias=bias)
            self.to_v = linear_cls(self.cross_attention_dim, self.inner_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = linear_cls(added_kv_proj_dim, self.inner_dim)
            self.add_v_proj = linear_cls(added_kv_proj_dim, self.inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(linear_cls(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
        if processor is None:
            processor = AttnProcessor2_0()
        self.set_processor(processor)

    def set_use_tpu_flash_attention(self):
        r"""
        Function sets the flag in this object. The flag will enforce the usage of TPU attention kernel.
        """
        self.use_tpu_flash_attention = True

    def set_processor(self, processor: "AttnProcessor2_0") -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(
                f"You are removing possibly trained weights of {self.processor} with {processor}"
            )
            self._modules.pop("processor")

        self.processor = processor

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        freqs_cis: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        skip_layer_mask: Optional[torch.Tensor] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            skip_layer_mask (`torch.Tensor`, *optional*):
                The skip layer mask to use. If `None`, no mask is applied.
            skip_layer_strategy (`SkipLayerStrategy`, *optional*, defaults to `None`):
                Controls which layers to skip for spatiotemporal guidance.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        attn_parameters = set(
            inspect.signature(self.processor.__call__).parameters.keys()
        )
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by"
                f" {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {
            k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters
        }

        return self.processor(
            self,
            hidden_states,
            freqs_cis=freqs_cis,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            skip_layer_mask=skip_layer_mask,
            skip_layer_strategy=skip_layer_strategy,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size // head_size, seq_len, dim * head_size
        )
        return tensor

    def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """

        head_size = self.heads
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(
            batch_size, seq_len * extra_dim, head_size, dim // head_size
        )
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(
                batch_size * head_size, seq_len * extra_dim, dim // head_size
            )

        return tensor

    def get_attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        target_length: int,
        batch_size: int,
        out_dim: int = 3,
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (
                    attention_mask.shape[0],
                    attention_mask.shape[1],
                    target_length,
                )
                padding = torch.zeros(
                    padding_shape,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def norm_encoder_hidden_states(
        self, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`torch.Tensor`): Hidden states of the encoder.

        Returns:
            `torch.Tensor`: The normalized encoder hidden states.
        """
        assert (
            self.norm_cross is not None
        ), "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False

        return encoder_hidden_states

    @staticmethod
    def apply_rotary_emb(
        input_tensor: torch.Tensor,
        freqs_cis: Tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_freqs = freqs_cis[0]
        sin_freqs = freqs_cis[1]

        t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
        t1, t2 = t_dup.unbind(dim=-1)
        t_dup = torch.stack((-t2, t1), dim=-1)
        input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")

        out = input_tensor * cos_freqs + input_tensor_rot * sin_freqs

        return out
    
class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        freqs_cis: Tuple[torch.FloatTensor, torch.FloatTensor],
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        skip_layer_mask: Optional[torch.FloatTensor] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if skip_layer_mask is not None:
            skip_layer_mask = skip_layer_mask.reshape(batch_size, 1, 1)

        if (attention_mask is not None) and (not attn.use_tpu_flash_attention):
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)
        query = attn.q_norm(query)

        if encoder_hidden_states is not None:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            key = attn.to_k(encoder_hidden_states)
            key = attn.k_norm(key)
        else:  # if no context provided do self-attention
            encoder_hidden_states = hidden_states
            key = attn.to_k(hidden_states)
            key = attn.k_norm(key)
            if attn.use_rope:
                key = attn.apply_rotary_emb(key, freqs_cis)
                query = attn.apply_rotary_emb(query, freqs_cis)

        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)

        if attn.use_tpu_flash_attention:  # use tpu attention offload 'flash attention'
            q_segment_indexes = None
            if (
                attention_mask is not None
            ):  # if mask is required need to tune both segmenIds fields
                # attention_mask = torch.squeeze(attention_mask).to(torch.float32)
                attention_mask = attention_mask.to(torch.float32)
                q_segment_indexes = torch.ones(
                    batch_size, query.shape[2], device=query.device, dtype=torch.float32
                )
                assert (
                    attention_mask.shape[1] == key.shape[2]
                ), f"ERROR: KEY SHAPE must be same as attention mask [{key.shape[2]}, {attention_mask.shape[1]}]"

            assert (
                query.shape[2] % 128 == 0
            ), f"ERROR: QUERY SHAPE must be divisible by 128 (TPU limitation) [{query.shape[2]}]"
            assert (
                key.shape[2] % 128 == 0
            ), f"ERROR: KEY SHAPE must be divisible by 128 (TPU limitation) [{key.shape[2]}]"

            # run the TPU kernel implemented in jax with pallas
            hidden_states_a = flash_attention(
                q=query,
                k=key,
                v=value,
                q_segment_ids=q_segment_indexes,
                kv_segment_ids=attention_mask,
                sm_scale=attn.scale,
            )
        else:
            hidden_states_a = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        hidden_states_a = hidden_states_a.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states_a = hidden_states_a.to(query.dtype)

        if (
            skip_layer_mask is not None
            and skip_layer_strategy == SkipLayerStrategy.Attention
        ):
            hidden_states = hidden_states_a * skip_layer_mask + hidden_states * (
                1.0 - skip_layer_mask
            )
        else:
            hidden_states = hidden_states_a

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
            if (
                skip_layer_mask is not None
                and skip_layer_strategy == SkipLayerStrategy.Residual
            ):
                skip_layer_mask = skip_layer_mask.reshape(batch_size, 1, 1, 1)

        if attn.residual_connection:
            if (
                skip_layer_mask is not None
                and skip_layer_strategy == SkipLayerStrategy.Residual
            ):
                hidden_states = hidden_states + residual * skip_layer_mask
            else:
                hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

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
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        skip_layer_mask: Optional[torch.Tensor] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
    ) -> torch.FloatTensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` to `cross_attention_kwargs` is depcrecated. `scale` will be ignored."
                )

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        norm_hidden_states = self.norm1(hidden_states)

        # Apply ada_norm_single
        if self.adaptive_norm in ["single_scale_shift", "single_scale"]:
            assert timestep.ndim == 3  # [batch, 1 or num_tokens, embedding_dim]
            num_ada_params = self.scale_shift_table.shape[0]
            ada_values = self.scale_shift_table[None, None] + timestep.reshape(
                batch_size, timestep.shape[1], num_ada_params, -1
            )
            if self.adaptive_norm == "single_scale_shift":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    ada_values.unbind(dim=2)
                )
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            else:
                scale_msa, gate_msa, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa)
        elif self.adaptive_norm == "none":
            scale_msa, gate_msa, scale_mlp, gate_mlp = None, None, None, None
        else:
            raise ValueError(f"Unknown adaptive norm type: {self.adaptive_norm}")

        norm_hidden_states = norm_hidden_states.squeeze(
            1
        )  # TODO: Check if this is needed

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )

        attn_output = self.attn1(
            norm_hidden_states,
            freqs_cis=freqs_cis,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            skip_layer_mask=skip_layer_mask,
            skip_layer_strategy=skip_layer_strategy,
            **cross_attention_kwargs,
        )
        if gate_msa is not None:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.adaptive_norm == "none":
                attn_input = self.attn2_norm(hidden_states)
            else:
                attn_input = hidden_states
            attn_output = self.attn2(
                attn_input,
                freqs_cis=freqs_cis,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        if self.adaptive_norm == "single_scale_shift":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        elif self.adaptive_norm == "single_scale":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp)
        elif self.adaptive_norm == "none":
            pass
        else:
            raise ValueError(f"Unknown adaptive norm type: {self.adaptive_norm}")

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states)
        if gate_mlp is not None:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


@dataclass
class Transformer3DModelOutput(BaseOutput):
    """Output of Transformer3DModel."""
    sample: torch.FloatTensor


class AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # No modulation happening here.
        added_cond_kwargs = added_cond_kwargs or {"resolution": None, "aspect_ratio": None}
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
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

    def initialize(self, embedding_std: float, mode: Literal["ltx_video", "legacy"]):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(
            self.adaln_single.emb.timestep_embedder.linear_1.weight, std=embedding_std
        )
        nn.init.normal_(
            self.adaln_single.emb.timestep_embedder.linear_2.weight, std=embedding_std
        )
        nn.init.normal_(self.adaln_single.linear.weight, std=embedding_std)

        if hasattr(self.adaln_single.emb, "resolution_embedder"):
            nn.init.normal_(
                self.adaln_single.emb.resolution_embedder.linear_1.weight,
                std=embedding_std,
            )
            nn.init.normal_(
                self.adaln_single.emb.resolution_embedder.linear_2.weight,
                std=embedding_std,
            )
        if hasattr(self.adaln_single.emb, "aspect_ratio_embedder"):
            nn.init.normal_(
                self.adaln_single.emb.aspect_ratio_embedder.linear_1.weight,
                std=embedding_std,
            )
            nn.init.normal_(
                self.adaln_single.emb.aspect_ratio_embedder.linear_2.weight,
                std=embedding_std,
            )

        # Initialize caption embedding MLP:
        nn.init.normal_(self.caption_projection.linear_1.weight, std=embedding_std)
        nn.init.normal_(self.caption_projection.linear_1.weight, std=embedding_std)

        for block in self.transformer_blocks:
            if mode.lower() == "ltx_video":
                nn.init.constant_(block.attn1.to_out[0].weight, 0)
                nn.init.constant_(block.attn1.to_out[0].bias, 0)

            nn.init.constant_(block.attn2.to_out[0].weight, 0)
            nn.init.constant_(block.attn2.to_out[0].bias, 0)

            if mode.lower() == "ltx_video":
                nn.init.constant_(block.ff.net[2].weight, 0)
                nn.init.constant_(block.ff.net[2].bias, 0)

        # Zero-out output layers:
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
        skip_layer_mask: Optional[torch.Tensor] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
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
        # for tpu attention offload 2d token masks are used. No need to transform.
        if not self.use_tpu_flash_attention:
            # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
            #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
            #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
            # expects mask of shape:
            #   [batch, key_tokens]
            # adds singleton query_tokens dimension:
            #   [batch,                    1, key_tokens]
            # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
            #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
            #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
            if attention_mask is not None and attention_mask.ndim == 2:
                # assume that mask is expressed as:
                #   (1 = keep,      0 = discard)
                # convert mask into a bias that can be added to attention scores:
                #       (keep = +0,     discard = -10000.0)
                attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # convert encoder_attention_mask to a bias the same way we do for attention_mask
            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (
                    1 - encoder_attention_mask.to(hidden_states.dtype)
                ) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        hidden_states = self.patchify_proj(hidden_states)

        if self.timestep_scale_multiplier:
            timestep = self.timestep_scale_multiplier * timestep

        if self.positional_embedding_type == "absolute":
            pos_embed_3d = self.get_absolute_pos_embed(indices_grid).to(
                hidden_states.device
            )
            if self.project_to_2d_pos:
                pos_embed = self.to_2d_proj(pos_embed_3d)
            hidden_states = (hidden_states + pos_embed).to(hidden_states.dtype)
            freqs_cis = None
        elif self.positional_embedding_type == "rope":
            freqs_cis = self.precompute_freqs_cis(indices_grid)

        batch_size = hidden_states.shape[0]
        timestep, embedded_timestep = self.adaln_single(
            timestep.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.shape[-1]
        )

        if skip_layer_mask is None:
            skip_layer_mask = torch.ones(
                len(self.transformer_blocks), batch_size, device=hidden_states.device
            )

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        for block_idx, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    freqs_cis,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    skip_layer_mask[block_idx],
                    skip_layer_strategy,
                    **ckpt_kwargs,
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
                    skip_layer_mask=skip_layer_mask[block_idx],
                    skip_layer_strategy=skip_layer_strategy,
                )

        # 3. Output
        scale_shift_values = (
            self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        if not return_dict:
            return (hidden_states,)

        return Transformer3DModelOutput(sample=hidden_states)

    def get_absolute_pos_embed(self, grid):
        grid_np = grid[0].cpu().numpy()
        embed_dim_3d = (
            math.ceil((self.inner_dim / 2) * 3)
            if self.project_to_2d_pos
            else self.inner_dim
        )
        pos_embed = get_3d_sincos_pos_embed(  # (f h w)
            embed_dim_3d,
            grid_np,
            h=int(max(grid_np[1]) + 1),
            w=int(max(grid_np[2]) + 1),
            f=int(max(grid_np[0] + 1)),
        )
        return torch.from_numpy(pos_embed).float().unsqueeze(0)