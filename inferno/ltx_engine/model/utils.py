import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum, auto

ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}

class SkipLayerStrategy(Enum):
    Attention = auto()
    Residual = auto()


def append_dims(x: torch.Tensor, target_dims: int):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    elif dims_to_append == 0:
        return x
    return x[(...,) + (None,) * dims_to_append]

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb
    
class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = ACTIVATION_FUNCTIONS[act_fn]

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = ACTIVATION_FUNCTIONS[post_act_fn]

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample
    
class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """
    For PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.aspect_ratio_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)

    def forward(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            aspect_ratio_emb = self.additional_condition_proj(aspect_ratio.flatten()).to(hidden_dtype)
            aspect_ratio_emb = self.aspect_ratio_embedder(aspect_ratio_emb).reshape(batch_size, -1)
            conditioning = timesteps_emb + torch.cat([resolution_emb, aspect_ratio_emb], dim=1)
        else:
            conditioning = timesteps_emb

        return conditioning
    
def make_hashable_key(dict_key):
    def convert_value(value):
        if isinstance(value, list):
            return tuple(value)
        elif isinstance(value, dict):
            return tuple(sorted((k, convert_value(v)) for k, v in value.items()))
        else:
            return value

    return tuple(sorted((k, convert_value(v)) for k, v in dict_key.items()))


DIFFUSERS_SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.32.0.dev0",
    "base_image_seq_len": 1024,
    "base_shift": 0.95,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 2.05,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.1,
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
DIFFUSERS_TRANSFORMER_CONFIG = {
    "_class_name": "LTXVideoTransformer3DModel",
    "_diffusers_version": "0.32.0.dev0",
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 64,
    "attention_out_bias": True,
    "caption_channels": 4096,
    "cross_attention_dim": 2048,
    "in_channels": 128,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-06,
    "num_attention_heads": 32,
    "num_layers": 28,
    "out_channels": 128,
    "patch_size": 1,
    "patch_size_t": 1,
    "qk_norm": "rms_norm_across_heads",
}
DIFFUSERS_VAE_CONFIG = {
    "_class_name": "AutoencoderKLLTXVideo",
    "_diffusers_version": "0.32.0.dev0",
    "block_out_channels": [128, 256, 512, 512],
    "decoder_causal": False,
    "encoder_causal": True,
    "in_channels": 3,
    "latent_channels": 128,
    "layers_per_block": [4, 3, 3, 3, 4],
    "out_channels": 3,
    "patch_size": 4,
    "patch_size_t": 1,
    "resnet_norm_eps": 1e-06,
    "scaling_factor": 1.0,
    "spatio_temporal_scaling": [True, True, True, False],
}

OURS_SCHEDULER_CONFIG = {
    "_class_name": "RectifiedFlowScheduler",
    "_diffusers_version": "0.25.1",
    "num_train_timesteps": 1000,
    "shifting": "SD3",
    "base_resolution": None,
    "target_shift_terminal": 0.1,
}

OURS_TRANSFORMER_CONFIG = {
    "_class_name": "Transformer3DModel",
    "_diffusers_version": "0.25.1",
    "_name_or_path": "PixArt-alpha/PixArt-XL-2-256x256",
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 64,
    "attention_type": "default",
    "caption_channels": 4096,
    "cross_attention_dim": 2048,
    "double_self_attention": False,
    "dropout": 0.0,
    "in_channels": 128,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-06,
    "norm_num_groups": 32,
    "num_attention_heads": 32,
    "num_embeds_ada_norm": 1000,
    "num_layers": 28,
    "num_vector_embeds": None,
    "only_cross_attention": False,
    "out_channels": 128,
    "project_to_2d_pos": True,
    "upcast_attention": False,
    "use_linear_projection": False,
    "qk_norm": "rms_norm",
    "standardization_norm": "rms_norm",
    "positional_embedding_type": "rope",
    "positional_embedding_theta": 10000.0,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "timestep_scale_multiplier": 1000,
}
OURS_VAE_CONFIG = {
    "_class_name": "CausalVideoAutoencoder",
    "dims": 3,
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 128,
    "blocks": [
        ["res_x", 4],
        ["compress_all", 1],
        ["res_x_y", 1],
        ["res_x", 3],
        ["compress_all", 1],
        ["res_x_y", 1],
        ["res_x", 3],
        ["compress_all", 1],
        ["res_x", 3],
        ["res_x", 4],
    ],
    "scaling_factor": 1.0,
    "norm_layer": "pixel_norm",
    "patch_size": 4,
    "latent_log_var": "uniform",
    "use_quant_conv": False,
    "causal_decoder": False,
}


diffusers_and_inferno_config_mapping = {
    make_hashable_key(DIFFUSERS_SCHEDULER_CONFIG): OURS_SCHEDULER_CONFIG,
    make_hashable_key(DIFFUSERS_TRANSFORMER_CONFIG): OURS_TRANSFORMER_CONFIG,
    make_hashable_key(DIFFUSERS_VAE_CONFIG): OURS_VAE_CONFIG,
}


TRANSFORMER_KEYS_RENAME_DICT = {
    "proj_in": "patchify_proj",
    "time_embed": "adaln_single",
    "norm_q": "q_norm",
    "norm_k": "k_norm",
}


VAE_KEYS_RENAME_DICT = {
    "decoder.up_blocks.3.conv_in": "decoder.up_blocks.7",
    "decoder.up_blocks.3.upsamplers.0": "decoder.up_blocks.8",
    "decoder.up_blocks.3": "decoder.up_blocks.9",
    "decoder.up_blocks.2.upsamplers.0": "decoder.up_blocks.5",
    "decoder.up_blocks.2.conv_in": "decoder.up_blocks.4",
    "decoder.up_blocks.2": "decoder.up_blocks.6",
    "decoder.up_blocks.1.upsamplers.0": "decoder.up_blocks.2",
    "decoder.up_blocks.1": "decoder.up_blocks.3",
    "decoder.up_blocks.0": "decoder.up_blocks.1",
    "decoder.mid_block": "decoder.up_blocks.0",
    "encoder.down_blocks.3": "encoder.down_blocks.8",
    "encoder.down_blocks.2.downsamplers.0": "encoder.down_blocks.7",
    "encoder.down_blocks.2": "encoder.down_blocks.6",
    "encoder.down_blocks.1.downsamplers.0": "encoder.down_blocks.4",
    "encoder.down_blocks.1.conv_out": "encoder.down_blocks.5",
    "encoder.down_blocks.1": "encoder.down_blocks.3",
    "encoder.down_blocks.0.conv_out": "encoder.down_blocks.2",
    "encoder.down_blocks.0.downsamplers.0": "encoder.down_blocks.1",
    "encoder.down_blocks.0": "encoder.down_blocks.0",
    "encoder.mid_block": "encoder.down_blocks.9",
    "conv_shortcut.conv": "conv_shortcut",
    "resnets": "res_blocks",
    "norm3": "norm3.norm",
    "latents_mean": "per_channel_statistics.mean-of-means",
    "latents_std": "per_channel_statistics.std-of-means",
}
