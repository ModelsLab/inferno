import torch
from model.autoencoder import AutoencoderKL
from einops import rearrange
from torch import Tensor


from model.autoencoder import CausalVideoAutoencoder

import json
import os
from functools import partial
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Tuple, Union

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional

from diffusers.utils import logging

from model.autoencoder import make_conv_nd, make_linear_nd
from model.autoencoder import PixelNorm
from model.autoencoder import AutoencoderKL as AutoencoderKLWrapper

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None

logger = logging.get_logger(__name__)

class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive."""

    def __init__(self, *args, **kwargs) -> None:  # pylint: disable=unused-argument
        super().__init__()

    # pylint: disable=unused-argument
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x

class VideoAutoencoder(AutoencoderKLWrapper):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        **kwargs,
    ):
        config_local_path = pretrained_model_name_or_path / "config.json"
        config = cls.load_config(config_local_path, **kwargs)
        video_vae = cls.from_config(config)
        video_vae.to(kwargs["torch_dtype"])

        model_local_path = pretrained_model_name_or_path / "autoencoder.pth"
        ckpt_state_dict = torch.load(model_local_path)
        video_vae.load_state_dict(ckpt_state_dict)

        statistics_local_path = (
            pretrained_model_name_or_path / "per_channel_statistics.json"
        )
        if statistics_local_path.exists():
            with open(statistics_local_path, "r") as file:
                data = json.load(file)
            transposed_data = list(zip(*data["data"]))
            data_dict = {
                col: torch.tensor(vals)
                for col, vals in zip(data["columns"], transposed_data)
            }
            video_vae.register_buffer("std_of_means", data_dict["std-of-means"])
            video_vae.register_buffer(
                "mean_of_means",
                data_dict.get(
                    "mean-of-means", torch.zeros_like(data_dict["std-of-means"])
                ),
            )

        return video_vae

    @staticmethod
    def from_config(config):
        assert (
            config["_class_name"] == "VideoAutoencoder"
        ), "config must have _class_name=VideoAutoencoder"
        if isinstance(config["dims"], list):
            config["dims"] = tuple(config["dims"])

        assert config["dims"] in [2, 3, (2, 1)], "dims must be 2, 3 or (2, 1)"

        double_z = config.get("double_z", True)
        latent_log_var = config.get(
            "latent_log_var", "per_channel" if double_z else "none"
        )
        use_quant_conv = config.get("use_quant_conv", True)

        if use_quant_conv and latent_log_var == "uniform":
            raise ValueError("uniform latent_log_var requires use_quant_conv=False")

        encoder = Encoder(
            dims=config["dims"],
            in_channels=config.get("in_channels", 3),
            out_channels=config["latent_channels"],
            block_out_channels=config["block_out_channels"],
            patch_size=config.get("patch_size", 1),
            latent_log_var=latent_log_var,
            norm_layer=config.get("norm_layer", "group_norm"),
            patch_size_t=config.get("patch_size_t", config.get("patch_size", 1)),
            add_channel_padding=config.get("add_channel_padding", False),
        )

        decoder = Decoder(
            dims=config["dims"],
            in_channels=config["latent_channels"],
            out_channels=config.get("out_channels", 3),
            block_out_channels=config["block_out_channels"],
            patch_size=config.get("patch_size", 1),
            norm_layer=config.get("norm_layer", "group_norm"),
            patch_size_t=config.get("patch_size_t", config.get("patch_size", 1)),
            add_channel_padding=config.get("add_channel_padding", False),
        )

        dims = config["dims"]
        return VideoAutoencoder(
            encoder=encoder,
            decoder=decoder,
            latent_channels=config["latent_channels"],
            dims=dims,
            use_quant_conv=use_quant_conv,
        )

    @property
    def config(self):
        return SimpleNamespace(
            _class_name="VideoAutoencoder",
            dims=self.dims,
            in_channels=self.encoder.conv_in.in_channels
            // (self.encoder.patch_size_t * self.encoder.patch_size**2),
            out_channels=self.decoder.conv_out.out_channels
            // (self.decoder.patch_size_t * self.decoder.patch_size**2),
            latent_channels=self.decoder.conv_in.in_channels,
            block_out_channels=[
                self.encoder.down_blocks[i].res_blocks[-1].conv1.out_channels
                for i in range(len(self.encoder.down_blocks))
            ],
            scaling_factor=1.0,
            norm_layer=self.encoder.norm_layer,
            patch_size=self.encoder.patch_size,
            latent_log_var=self.encoder.latent_log_var,
            use_quant_conv=self.use_quant_conv,
            patch_size_t=self.encoder.patch_size_t,
            add_channel_padding=self.encoder.add_channel_padding,
        )

    @property
    def is_video_supported(self):
        """
        Check if the model supports video inputs of shape (B, C, F, H, W). Otherwise, the model only supports 2D images.
        """
        return self.dims != 2

    @property
    def downscale_factor(self):
        return self.encoder.downsample_factor

    def to_json_string(self) -> str:
        import json

        return json.dumps(self.config.__dict__)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        model_keys = set(name for name, _ in self.named_parameters())

        key_mapping = {
            ".resnets.": ".res_blocks.",
            "downsamplers.0": "downsample",
            "upsamplers.0": "upsample",
        }

        converted_state_dict = {}
        for key, value in state_dict.items():
            for k, v in key_mapping.items():
                key = key.replace(k, v)

            if "norm" in key and key not in model_keys:
                logger.info(
                    f"Removing key {key} from state_dict as it is not present in the model"
                )
                continue

            converted_state_dict[key] = value

        super().load_state_dict(converted_state_dict, strict=strict)

    def last_layer(self):
        if hasattr(self.decoder, "conv_out"):
            if isinstance(self.decoder.conv_out, nn.Sequential):
                last_layer = self.decoder.conv_out[-1]
            else:
                last_layer = self.decoder.conv_out
        else:
            last_layer = self.decoder.layers[-1]
        return last_layer


class Encoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        patch_size (`int`, *optional*, defaults to 1):
            The patch size to use. Should be a power of 2.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        latent_log_var (`str`, *optional*, defaults to `per_channel`):
            The number of channels for the log variance. Can be either `per_channel`, `uniform`, or `none`.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]] = 3,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        patch_size: Union[int, Tuple[int]] = 1,
        norm_layer: str = "group_norm",  # group_norm, pixel_norm
        latent_log_var: str = "per_channel",
        patch_size_t: Optional[int] = None,
        add_channel_padding: Optional[bool] = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t if patch_size_t is not None else patch_size
        self.add_channel_padding = add_channel_padding
        self.layers_per_block = layers_per_block
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        if add_channel_padding:
            in_channels = in_channels * self.patch_size**3
        else:
            in_channels = in_channels * self.patch_size_t * self.patch_size**2
        self.in_channels = in_channels
        output_channel = block_out_channels[0]

        self.conv_in = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownEncoderBlock3D(
                dims=dims,
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block,
                add_downsample=not is_final_block and 2**i >= patch_size,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_groups=norm_num_groups,
                norm_layer=norm_layer,
            )
            self.down_blocks.append(down_block)

        self.mid_block = LTXMidBlock3D(
            dims=dims,
            in_channels=block_out_channels[-1],
            num_layers=self.layers_per_block,
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
        )

        # out
        if norm_layer == "group_norm":
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[-1],
                num_groups=norm_num_groups,
                eps=1e-6,
            )
        elif norm_layer == "pixel_norm":
            self.conv_norm_out = PixelNorm()
        self.conv_act = nn.SiLU()

        conv_out_channels = out_channels
        if latent_log_var == "per_channel":
            conv_out_channels *= 2
        elif latent_log_var == "uniform":
            conv_out_channels += 1
        elif latent_log_var != "none":
            raise ValueError(f"Invalid latent_log_var: {latent_log_var}")
        self.conv_out = make_conv_nd(
            dims, block_out_channels[-1], conv_out_channels, 3, padding=1
        )

        self.gradient_checkpointing = False

    @property
    def downscale_factor(self):
        return (
            2
            ** len(
                [
                    block
                    for block in self.down_blocks
                    if isinstance(block.downsample, Downsample3D)
                ]
            )
            * self.patch_size
        )

    def forward(
        self, sample: torch.FloatTensor, return_features=False
    ) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""

        downsample_in_time = sample.shape[2] != 1

        # patchify
        patch_size_t = self.patch_size_t if downsample_in_time else 1
        sample = patchify(
            sample,
            patch_size_hw=self.patch_size,
            patch_size_t=patch_size_t,
            add_channel_padding=self.add_channel_padding,
        )

        sample = self.conv_in(sample)

        checkpoint_fn = (
            partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else lambda x: x
        )

        if return_features:
            features = []
        for down_block in self.down_blocks:
            sample = checkpoint_fn(down_block)(
                sample, downsample_in_time=downsample_in_time
            )
            if return_features:
                features.append(sample)

        sample = checkpoint_fn(self.mid_block)(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.latent_log_var == "uniform":
            last_channel = sample[:, -1:, ...]
            num_dims = sample.dim()

            if num_dims == 4:
                # For shape (B, C, H, W)
                repeated_last_channel = last_channel.repeat(
                    1, sample.shape[1] - 2, 1, 1
                )
                sample = torch.cat([sample, repeated_last_channel], dim=1)
            elif num_dims == 5:
                # For shape (B, C, F, H, W)
                repeated_last_channel = last_channel.repeat(
                    1, sample.shape[1] - 2, 1, 1, 1
                )
                sample = torch.cat([sample, repeated_last_channel], dim=1)
            else:
                raise ValueError(f"Invalid input shape: {sample.shape}")

        if return_features:
            features.append(sample[:, : self.latent_channels, ...])
            return sample, features
        return sample


class Decoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        patch_size (`int`, *optional*, defaults to 1):
            The patch size to use. Should be a power of 2.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
    """

    def __init__(
        self,
        dims,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        patch_size: int = 1,
        norm_layer: str = "group_norm",
        patch_size_t: Optional[int] = None,
        add_channel_padding: Optional[bool] = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t if patch_size_t is not None else patch_size
        self.add_channel_padding = add_channel_padding
        self.layers_per_block = layers_per_block
        if add_channel_padding:
            out_channels = out_channels * self.patch_size**3
        else:
            out_channels = out_channels * self.patch_size_t * self.patch_size**2
        self.out_channels = out_channels

        self.conv_in = make_conv_nd(
            dims,
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        self.mid_block = LTXMidBlock3D(
            dims=dims,
            in_channels=block_out_channels[-1],
            num_layers=self.layers_per_block,
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
        )

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = UpDecoderBlock3D(
                dims=dims,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block
                and 2 ** (len(block_out_channels) - i - 1) > patch_size,
                resnet_eps=1e-6,
                resnet_groups=norm_num_groups,
                norm_layer=norm_layer,
            )
            self.up_blocks.append(up_block)

        if norm_layer == "group_norm":
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
            )
        elif norm_layer == "pixel_norm":
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims, block_out_channels[0], out_channels, 3, padding=1
        )

        self.gradient_checkpointing = False

    def forward(self, sample: torch.FloatTensor, target_shape) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""
        assert target_shape is not None, "target_shape must be provided"
        upsample_in_time = sample.shape[2] < target_shape[2]

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        checkpoint_fn = (
            partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else lambda x: x
        )

        sample = checkpoint_fn(self.mid_block)(sample)
        sample = sample.to(upscale_dtype)

        for up_block in self.up_blocks:
            sample = checkpoint_fn(up_block)(sample, upsample_in_time=upsample_in_time)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # un-patchify
        patch_size_t = self.patch_size_t if upsample_in_time else 1
        sample = unpatchify(
            sample,
            patch_size_hw=self.patch_size,
            patch_size_t=patch_size_t,
            add_channel_padding=self.add_channel_padding,
        )

        return sample


class DownEncoderBlock3D(nn.Module):
    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        norm_layer: str = "group_norm",
    ):
        super().__init__()
        res_blocks = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            res_blocks.append(
                ResnetBlock3D(
                    dims=dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    norm_layer=norm_layer,
                )
            )

        self.res_blocks = nn.ModuleList(res_blocks)

        if add_downsample:
            self.downsample = Downsample3D(
                dims,
                out_channels,
                out_channels=out_channels,
                padding=downsample_padding,
            )
        else:
            self.downsample = Identity()

    def forward(
        self, hidden_states: torch.FloatTensor, downsample_in_time
    ) -> torch.FloatTensor:
        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states)

        hidden_states = self.downsample(
            hidden_states, downsample_in_time=downsample_in_time
        )

        return hidden_states


class LTXMidBlock3D(nn.Module):
    """
    A 3D LTX mid-block [`LTXMidBlock3D`] with multiple residual blocks.

    Args:
        in_channels (`int`): The number of input channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.

    Returns:
        `torch.FloatTensor`: The output of the last residual block, which is a tensor of shape `(batch_size,
        in_channels, height, width)`.

    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        norm_layer: str = "group_norm",
    ):
        super().__init__()
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )

        self.res_blocks = nn.ModuleList(
            [
                ResnetBlock3D(
                    dims=dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    norm_layer=norm_layer,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states)

        return hidden_states


class UpDecoderBlock3D(nn.Module):
    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        add_upsample: bool = True,
        norm_layer: str = "group_norm",
    ):
        super().__init__()
        res_blocks = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            res_blocks.append(
                ResnetBlock3D(
                    dims=dims,
                    in_channels=input_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    norm_layer=norm_layer,
                )
            )

        self.res_blocks = nn.ModuleList(res_blocks)

        if add_upsample:
            self.upsample = Upsample3D(
                dims=dims, channels=out_channels, out_channels=out_channels
            )
        else:
            self.upsample = Identity()

        self.resolution_idx = resolution_idx

    def forward(
        self, hidden_states: torch.FloatTensor, upsample_in_time=True
    ) -> torch.FloatTensor:
        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states)

        hidden_states = self.upsample(hidden_states, upsample_in_time=upsample_in_time)

        return hidden_states


class ResnetBlock3D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        norm_layer: str = "group_norm",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        if norm_layer == "group_norm":
            self.norm1 = torch.nn.GroupNorm(
                num_groups=groups, num_channels=in_channels, eps=eps, affine=True
            )
        elif norm_layer == "pixel_norm":
            self.norm1 = PixelNorm()

        self.non_linearity = nn.SiLU()

        self.conv1 = make_conv_nd(
            dims, in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if norm_layer == "group_norm":
            self.norm2 = torch.nn.GroupNorm(
                num_groups=groups, num_channels=out_channels, eps=eps, affine=True
            )
        elif norm_layer == "pixel_norm":
            self.norm2 = PixelNorm()

        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = make_conv_nd(
            dims, out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.conv_shortcut = (
            make_linear_nd(
                dims=dims, in_channels=in_channels, out_channels=out_channels
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        input_tensor: torch.FloatTensor,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)

        hidden_states = self.non_linearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)

        hidden_states = self.non_linearity(hidden_states)

        hidden_states = self.dropout(hidden_states)

        hidden_states = self.conv2(hidden_states)

        input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


class Downsample3D(nn.Module):
    def __init__(
        self,
        dims,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        stride: int = 2
        self.padding = padding
        self.in_channels = in_channels
        self.dims = dims
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x, downsample_in_time=True):
        conv = self.conv
        if self.padding == 0:
            if self.dims == 2:
                padding = (0, 1, 0, 1)
            else:
                padding = (0, 1, 0, 1, 0, 1 if downsample_in_time else 0)

            x = functional.pad(x, padding, mode="constant", value=0)

            if self.dims == (2, 1) and not downsample_in_time:
                return conv(x, skip_time_conv=True)

        return conv(x)


class Upsample3D(nn.Module):
    """
    An upsampling layer for 3D tensors of shape (B, C, D, H, W).

    :param channels: channels in the inputs and outputs.
    """

    def __init__(self, dims, channels, out_channels=None):
        super().__init__()
        self.dims = dims
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = make_conv_nd(
            dims, channels, out_channels, kernel_size=3, padding=1, bias=True
        )

    def forward(self, x, upsample_in_time):
        if self.dims == 2:
            x = functional.interpolate(
                x, (x.shape[2] * 2, x.shape[3] * 2), mode="nearest"
            )
        else:
            time_scale_factor = 2 if upsample_in_time else 1
            # print("before:", x.shape)
            b, c, d, h, w = x.shape
            x = rearrange(x, "b c d h w -> (b d) c h w")
            # height and width interpolate
            x = functional.interpolate(
                x, (x.shape[2] * 2, x.shape[3] * 2), mode="nearest"
            )
            _, _, h, w = x.shape

            if not upsample_in_time and self.dims == (2, 1):
                x = rearrange(x, "(b d) c h w -> b c d h w ", b=b, h=h, w=w)
                return self.conv(x, skip_time_conv=True)

            # Second ** upsampling ** which is essentially treated as a 1D convolution across the 'd' dimension
            x = rearrange(x, "(b d) c h w -> (b h w) c 1 d", b=b)

            # (b h w) c 1 d
            new_d = x.shape[-1] * time_scale_factor
            x = functional.interpolate(x, (1, new_d), mode="nearest")
            # (b h w) c 1 new_d
            x = rearrange(
                x, "(b h w) c 1 new_d  -> b c new_d h w", b=b, h=h, w=w, new_d=new_d
            )
            # b c d h w

            # x = functional.interpolate(
            #     x, (x.shape[2] * time_scale_factor, x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            # )
            # print("after:", x.shape)

        return self.conv(x)


def patchify(x, patch_size_hw, patch_size_t=1, add_channel_padding=False):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(
            x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size_hw, r=patch_size_hw
        )
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    if (
        (x.dim() == 5)
        and (patch_size_hw > patch_size_t)
        and (patch_size_t > 1 or add_channel_padding)
    ):
        channels_to_pad = x.shape[1] * (patch_size_hw // patch_size_t) - x.shape[1]
        padding_zeros = torch.zeros(
            x.shape[0],
            channels_to_pad,
            x.shape[2],
            x.shape[3],
            x.shape[4],
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([padding_zeros, x], dim=1)

    return x


def unpatchify(x, patch_size_hw, patch_size_t=1, add_channel_padding=False):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    if (
        (x.dim() == 5)
        and (patch_size_hw > patch_size_t)
        and (patch_size_t > 1 or add_channel_padding)
    ):
        channels_to_keep = int(x.shape[1] * (patch_size_t / patch_size_hw))
        x = x[:, :channels_to_keep, :, :, :]

    if x.dim() == 4:
        x = rearrange(
            x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size_hw, r=patch_size_hw
        )
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )

    return x


def vae_encode(
    media_items: Tensor,
    vae: AutoencoderKL,
    split_size: int = 1,
    vae_per_channel_normalize=False,
) -> Tensor:
    """
    Encodes media items (images or videos) into latent representations using a specified VAE model.
    The function supports processing batches of images or video frames and can handle the processing
    in smaller sub-batches if needed.

    Args:
        media_items (Tensor): A torch Tensor containing the media items to encode. The expected
            shape is (batch_size, channels, height, width) for images or (batch_size, channels,
            frames, height, width) for videos.
        vae (AutoencoderKL): An instance of the `AutoencoderKL` class from the `diffusers` library,
            pre-configured and loaded with the appropriate model weights.
        split_size (int, optional): The number of sub-batches to split the input batch into for encoding.
            If set to more than 1, the input media items are processed in smaller batches according to
            this value. Defaults to 1, which processes all items in a single batch.

    Returns:
        Tensor: A torch Tensor of the encoded latent representations. The shape of the tensor is adjusted
            to match the input shape, scaled by the model's configuration.

    Examples:
        >>> import torch
        >>> from diffusers import AutoencoderKL
        >>> vae = AutoencoderKL.from_pretrained('your-model-name')
        >>> images = torch.rand(10, 3, 8 256, 256)  # Example tensor with 10 videos of 8 frames.
        >>> latents = vae_encode(images, vae)
        >>> print(latents.shape)  # Output shape will depend on the model's latent configuration.

    Note:
        In case of a video, the function encodes the media item frame-by frame.
    """
    is_video_shaped = media_items.dim() == 5
    batch_size, channels = media_items.shape[0:2]

    if channels != 3:
        raise ValueError(f"Expects tensors with 3 channels, got {channels}.")

    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        media_items = rearrange(media_items, "b c n h w -> (b n) c h w")
    if split_size > 1:
        if len(media_items) % split_size != 0:
            raise ValueError(
                "Error: The batch size must be divisible by 'train.vae_bs_split"
            )
        encode_bs = len(media_items) // split_size
        # latents = [vae.encode(image_batch).latent_dist.sample() for image_batch in media_items.split(encode_bs)]
        latents = []
        if media_items.device.type == "xla":
            xm.mark_step()
        for image_batch in media_items.split(encode_bs):
            latents.append(vae.encode(image_batch).latent_dist.sample())
            if media_items.device.type == "xla":
                xm.mark_step()
        latents = torch.cat(latents, dim=0)
    else:
        latents = vae.encode(media_items).latent_dist.sample()

    latents = normalize_latents(latents, vae, vae_per_channel_normalize)
    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        latents = rearrange(latents, "(b n) c h w -> b c n h w", b=batch_size)
    return latents


def vae_decode(
    latents: Tensor,
    vae: AutoencoderKL,
    is_video: bool = True,
    split_size: int = 1,
    vae_per_channel_normalize=False,
) -> Tensor:
    is_video_shaped = latents.dim() == 5
    batch_size = latents.shape[0]

    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        latents = rearrange(latents, "b c n h w -> (b n) c h w")
    if split_size > 1:
        if len(latents) % split_size != 0:
            raise ValueError(
                "Error: The batch size must be divisible by 'train.vae_bs_split"
            )
        encode_bs = len(latents) // split_size
        image_batch = [
            _run_decoder(latent_batch, vae, is_video, vae_per_channel_normalize)
            for latent_batch in latents.split(encode_bs)
        ]
        images = torch.cat(image_batch, dim=0)
    else:
        images = _run_decoder(latents, vae, is_video, vae_per_channel_normalize)

    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        images = rearrange(images, "(b n) c h w -> b c n h w", b=batch_size)
    return images


def _run_decoder(
    latents: Tensor, vae: AutoencoderKL, is_video: bool, vae_per_channel_normalize=False
) -> Tensor:
    if isinstance(vae, (VideoAutoencoder, CausalVideoAutoencoder)):
        *_, fl, hl, wl = latents.shape
        temporal_scale, spatial_scale, _ = get_vae_size_scale_factor(vae)
        latents = latents.to(vae.dtype)
        image = vae.decode(
            un_normalize_latents(latents, vae, vae_per_channel_normalize),
            return_dict=False,
            target_shape=(
                1,
                3,
                fl * temporal_scale if is_video else 1,
                hl * spatial_scale,
                wl * spatial_scale,
            ),
        )[0]
    else:
        image = vae.decode(
            un_normalize_latents(latents, vae, vae_per_channel_normalize),
            return_dict=False,
        )[0]
    return image


def get_vae_size_scale_factor(vae: AutoencoderKL) -> float:
    if isinstance(vae, CausalVideoAutoencoder):
        spatial = vae.spatial_downscale_factor
        temporal = vae.temporal_downscale_factor
    else:
        down_blocks = len(
            [
                block
                for block in vae.encoder.down_blocks
                if isinstance(block.downsample, Downsample3D)
            ]
        )
        spatial = vae.config.patch_size * 2**down_blocks
        temporal = (
            vae.config.patch_size_t * 2**down_blocks
            if isinstance(vae, VideoAutoencoder)
            else 1
        )

    return (temporal, spatial, spatial)


def normalize_latents(
    latents: Tensor, vae: AutoencoderKL, vae_per_channel_normalize: bool = False
) -> Tensor:
    return (
        (latents - vae.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1))
        / vae.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
        if vae_per_channel_normalize
        else latents * vae.config.scaling_factor
    )


def un_normalize_latents(
    latents: Tensor, vae: AutoencoderKL, vae_per_channel_normalize: bool = False
) -> Tensor:
    return (
        latents * vae.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
        + vae.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
        if vae_per_channel_normalize
        else latents / vae.config.scaling_factor
    )