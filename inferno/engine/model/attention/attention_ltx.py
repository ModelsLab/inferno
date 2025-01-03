from typing import Optional, Tuple
import torch
from inferno.ltx_engine.infer import SkipLayerStrategy
import q8_kernels.functional as Q8F
import torch.nn.functional as F
from inferno.ltx_engine.model.transformer import Attention

try:
    from torch_xla.experimental.custom_kernel import flash_attention
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

NON_MM_PRECISION_TYPE = torch.bfloat16
MM_PRECISION_TYPE = torch.bfloat16

class LTXVideoQ8AttentionProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        if attention_mask is not None and attention_mask.ndim > 1:
            attention_mask = attention_mask.argmin(-1).squeeze().int()

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query, NON_MM_PRECISION_TYPE)
        key = attn.norm_k(key, NON_MM_PRECISION_TYPE)

        if image_rotary_emb is not None:
            query = attn.apply_rotary_emb(query, image_rotary_emb)
            key = attn.apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        hidden_states = Q8F.flash_attention.flash_attn_func(
            query, key, value, batch_mask=attention_mask, apply_qk_hadamard=True
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
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