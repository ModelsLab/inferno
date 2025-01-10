from dataclasses import dataclass
import glob
import json
import logging
import math
import os
from pathlib import Path
from typing import Literal, Optional, Tuple, Dict, Any, List, Union

from safetensors import safe_open
from ...utils.ltx_utils import TRANSFORMER_KEYS_RENAME_DICT, make_hashable_key, maybe_allow_in_graph, SkipLayerStrategy, diffusers_and_inferno_config_mapping
import torch
import torch.nn as nn
from ..blocks.normalization import AdaLayerNormSingle, RMSNorm
from ..attention.attention_ltx import Attention
from blocks.feed_forward import FeedForward, _chunked_feed_forward
from diffusers.utils import BaseOutput, is_torch_version
from ..embeddings.timestep_embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from ..embeddings.positional_embeddings import get_3d_sincos_pos_embed
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from ..embeddings.caption_embeddings import PixArtAlphaTextProjection

logger = logging.getLogger(__name__)

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


class LTXTransformer3DModel(ModelMixin, ConfigMixin):
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
            transformer =   LTXTransformer3DModel.from_config(transformer_config)
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