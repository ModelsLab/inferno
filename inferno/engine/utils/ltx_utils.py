from enum import Enum, auto
import torch
import torch.nn as nn
from kernels.ltx_video.q8_kernels.modules.rms_norm import RMSNorm as QRMSNorm
from diffusers.models.normalization import RMSNorm
from kernels.ltx_video.q8_kernels.modules.activations import GELU as QGELU
from diffusers.models.activations import GELU
from kernels.ltx_video.q8_kernels.modules.linear import Q8Linear
from ..model.attention.attention_ltx import LTXVideoQ8AttentionProcessor
import argparse
from diffusers import LTXVideoTransformer3DModel
from kernels.ltx_video.q8_kernels.functional.quantizer import quantize
from kernels.ltx_video.q8_kernels.functional.fast_hadamard import hadamard_transform

MODULES_TO_NOT_CONVERT = ["proj_in", "time_embed", "caption_projection", "proj_out"]

try:
    from torch._dynamo import allow_in_graph as maybe_allow_in_graph
except (ImportError, ModuleNotFoundError):

    def maybe_allow_in_graph(cls):
        return cls
    
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

def replace_linear(model, current_key_name=None, replaced=False):
    for name, child in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(child, nn.Linear) and name not in MODULES_TO_NOT_CONVERT:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in MODULES_TO_NOT_CONVERT
            ):
                new_linear = Q8Linear(
                    child.in_features, child.out_features, bias=child.bias is not None, device=child.weight.device
                )
                setattr(model, name, new_linear)
                replaced = True
        else:
            replace_linear(model=child, current_key_name=current_key_name, replaced=replaced)

        current_key_name.pop(-1)

    return model, replaced


def get_parent_module_and_attr(root, dotted_name: str):
    """
    Splits 'a.b.c' into:
    - parent module = root.a.b
    - attr_name = 'c'
    """
    parts = dotted_name.split(".")
    *parent_parts, attr_name = parts
    parent_module = root
    for p in parent_parts:
        parent_module = getattr(parent_module, p)
    return parent_module, attr_name


def replace_rms_norm(model):
    modules_to_replace = []
    for dotted_name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            modules_to_replace.append((dotted_name, module))

    replaced = False
    for dotted_name, module in modules_to_replace:
        parent, attr_name = get_parent_module_and_attr(model, dotted_name)
        new_norm = QRMSNorm(
            dim=module.dim,
            elementwise_affine=module.elementwise_affine,
        )
        setattr(parent, attr_name, new_norm)
        replaced = True

    return model, replaced


def replace_gelu(model, replaced=False):
    for name, child in model.named_children():
        if isinstance(child, GELU):
            new_gelu = QGELU(
                dim_in=child.proj.in_features,
                dim_out=child.proj.out_features,
                approximate=child.approximate,
                bias=child.proj.bias is not None,
            )
            setattr(model, name, new_gelu)
            replaced = True
        else:
            replace_gelu(model=child, replaced=replaced)

    return model, replaced


def set_attn_processors(model, processor):
    def fn_recursive_attn_processor(name, module: torch.nn.Module, processor):
        if hasattr(module, "set_processor"):
            module.set_processor(processor)
        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

    for name, module in model.named_children():
        fn_recursive_attn_processor(name, module, processor)


def attn_processors(model) -> dict:
    processors = {}

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: dict):
        if hasattr(module, "get_processor"):
            processors[f"{name}.processor"] = module.get_processor()

        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

        return processors

    for name, module in model.named_children():
        fn_recursive_add_processors(name, module, processors)

    return processors


def check_transformer_replaced_correctly(model):
    for block in model.transformer_blocks:
        assert isinstance(block.attn1.to_q, Q8Linear), f"{type(block.attn1.to_q)=} not linear."
        assert isinstance(block.attn2.to_q, Q8Linear), f"{type(block.attn2.to_q)=} not linear."
        assert block.attn1.to_q.weight.dtype == torch.int8, f"{block.attn1.to_q.weight.dtype=}."
        assert block.attn2.to_q.weight.dtype == torch.int8, f"{name=} {block.attn2.to_q.weight.dtype=}."

    for name, module in model.named_modules():
        if "norm" in name and "norm_out" not in name:
            assert isinstance(module, QRMSNorm), f"{name=}, {type(module)=}"

    for block in model.transformer_blocks:
        assert isinstance(block.ff.net[0], QGELU), f"{type(block.ff.net[0])=}"
        if getattr(block.ff.net[0], "proj", None) is not None:
            assert block.ff.net[0].proj.weight.dtype == torch.int8, f"{block.ff.net[0].proj.weight.dtype=}."

    set_attn_processors(model, LTXVideoQ8AttentionProcessor())
    all_attn_processors = attn_processors(model)
    for k, v in all_attn_processors.items():
        assert isinstance(v, LTXVideoQ8AttentionProcessor), f"{name} is not of type LTXVideoQ8AttentionProcessor."

"""
References:
https://github.com/KONAKONA666/q8_kernels/blob/9cee3f3d4ca5ec8ab463179be32c8001e31f8f33/q8_kernels/utils/convert_weights.py
"""
def convert_state_dict(orig_state_dict):
    prefix = "transformer_blocks"
    transformer_block_keys = []
    non_transformer_block_keys = []
    for k in orig_state_dict:
        if prefix in k:
            transformer_block_keys.append(k)
        else:
            non_transformer_block_keys.append(k)
    attn_keys = []
    ffn_keys = []
    scale_shift_keys = []
    for k in transformer_block_keys:
        if "attn" in k:
            attn_keys.append(k)
    for k in transformer_block_keys:
        if "ff" in k:
            ffn_keys.append(k)
    for k in transformer_block_keys:
        if "scale_shift_table" in k:
            scale_shift_keys.append(k)

    assert len(attn_keys + ffn_keys + scale_shift_keys) == len(transformer_block_keys), "error"

    new_state_dict = {}
    for k in attn_keys:
        new_key = k
        if "norm" in k and "weight" in k:
            new_state_dict[new_key] = orig_state_dict[k].float()
        elif "bias" in k:
            new_state_dict[new_key] = orig_state_dict[k].float()
        elif "weight" in k:
            w_quant, w_scales = quantize(hadamard_transform(orig_state_dict[k].cuda().to(torch.bfloat16)))
            assert w_quant.dtype == torch.int8, k
            new_state_dict[new_key] = w_quant
            new_state_dict[new_key.replace("weight", "scales")] = w_scales

    for k in ffn_keys:
        new_key = k

        if "bias" in k:
            new_state_dict[new_key] = orig_state_dict[k].float()
        elif "weight" in k:
            w_quant, w_scales = quantize(hadamard_transform(orig_state_dict[k].cuda().to(torch.bfloat16)))
            assert w_quant.dtype == torch.int8, k
            new_state_dict[new_key] = w_quant
            new_state_dict[new_key.replace("weight", "scales")] = w_scales

    for k in scale_shift_keys:
        new_state_dict[k] = orig_state_dict[k]

    for k in non_transformer_block_keys:
        new_state_dict[k] = orig_state_dict[k]

    return new_state_dict

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

ASPECT_RATIO_1024_BIN = {
    "0.25": [512.0, 2048.0],
    "0.28": [512.0, 1856.0],
    "0.32": [576.0, 1792.0],
    "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0],
    "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0],
    "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0],
    "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0],
    "0.6": [768.0, 1280.0],
    "0.68": [832.0, 1216.0],
    "0.72": [832.0, 1152.0],
    "0.78": [896.0, 1152.0],
    "0.82": [896.0, 1088.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.21": [1088.0, 896.0],
    "1.29": [1152.0, 896.0],
    "1.38": [1152.0, 832.0],
    "1.46": [1216.0, 832.0],
    "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0],
    "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0],
    "3.0": [1728.0, 576.0],
    "4.0": [2048.0, 512.0],
}

ASPECT_RATIO_512_BIN = {
    "0.25": [256.0, 1024.0],
    "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0],
    "0.33": [288.0, 864.0],
    "0.35": [288.0, 832.0],
    "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0],
    "0.48": [352.0, 736.0],
    "0.5": [352.0, 704.0],
    "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0],
    "0.6": [384.0, 640.0],
    "0.68": [416.0, 608.0],
    "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0],
    "0.82": [448.0, 544.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0],
    "1.38": [576.0, 416.0],
    "1.46": [608.0, 416.0],
    "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.09": [736.0, 352.0],
    "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0],
    "3.0": [864.0, 288.0],
    "4.0": [1024.0, 256.0],
}

@torch.no_grad()
def main(args):
    transformer = LTXVideoTransformer3DModel.from_pretrained(args.input_path, subfolder="transformer").to("cuda")
    new_state_dict = convert_state_dict(transformer.state_dict())
    transformer = replace_gelu(transformer)[0]
    transformer = replace_linear(transformer)[0]
    transformer = replace_rms_norm(transformer)[0]

    m, u = transformer.load_state_dict(new_state_dict, strict=True)
    for name, module in transformer.named_modules():
        if any(n in name for n in MODULES_TO_NOT_CONVERT):
            if hasattr(module, "weight"):
                assert module.weight.dtype == torch.float32
            elif hasattr(module, "linear"):
                assert module.linear.weight.dtype == torch.float32
        elif getattr(module, "weight", None) is not None:
            print(f"Non FP32 {name=} {module.weight.dtype=}")
            if "to_" in name:
                assert module.weight.dtype != torch.float32, f"{name=}, {module.weight.dtype=}"

    transformer.save_pretrained(args.output_path)
    print(f"Model saved in {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    main(args)