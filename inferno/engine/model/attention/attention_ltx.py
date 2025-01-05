import logging
import inspect
from typing import Optional, Tuple
from einops import rearrange
import torch
from utils.ltx_utils import SkipLayerStrategy
import kernels.ltx_video.q8_kernels.functional as Q8F
import torch.nn as nn
import torch.nn.functional as F
from blocks.normalization import RMSNorm, SpatialNorm
from utils.ltx_utils import maybe_allow_in_graph

try:
    from torch_xla.experimental.custom_kernel import flash_attention
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

NON_MM_PRECISION_TYPE = torch.bfloat16
MM_PRECISION_TYPE = torch.bfloat16

logger = logging.getLogger(__name__)

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
        processor: Optional["LTXVideoAttentionProcessor"] = None,
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
            processor = LTXVideoAttentionProcessor()
        self.set_processor(processor)

    def set_use_tpu_flash_attention(self):
        r"""
        Function sets the flag in this object. The flag will enforce the usage of TPU attention kernel.
        """
        self.use_tpu_flash_attention = True

    def set_processor(self, processor: "LTXVideoAttentionProcessor") -> None:
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

class LTXVideoQ8AttentionProcessor:
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

        if attention_mask is not None and attention_mask.ndim > 1:
            attention_mask = attention_mask.argmin(-1).squeeze().int()

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is not None:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = attn.norm_q(query, NON_MM_PRECISION_TYPE)
        key = attn.norm_k(key, NON_MM_PRECISION_TYPE)

        if attn.use_rope and freqs_cis is not None:
            query = attn.apply_rotary_emb(query, freqs_cis)
            key = attn.apply_rotary_emb(key, freqs_cis)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        hidden_states = Q8F.flash_attention.flash_attn_func(
            query, key, value, batch_mask=attention_mask, apply_qk_hadamard=True
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
            if skip_layer_mask is not None and skip_layer_strategy == SkipLayerStrategy.Residual:
                skip_layer_mask = skip_layer_mask.reshape(batch_size, 1, 1, 1)

        if attn.residual_connection:
            if skip_layer_mask is not None and skip_layer_strategy == SkipLayerStrategy.Residual:
                hidden_states = hidden_states + residual * skip_layer_mask
            else:
                hidden_states = hidden_states + residual

        if hasattr(attn, 'rescale_output_factor'):
            hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states.to(NON_MM_PRECISION_TYPE)
    
class LTXVideoAttentionProcessor:
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

        if attn.upcast_attention:
            value = value.to(query.dtype)

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
                value.to(query.dtype),
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

        hidden_states = hidden_states.to(attn.to_out[0].weight.dtype)

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