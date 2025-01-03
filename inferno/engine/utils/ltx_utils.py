import torch
import torch.nn as nn
from q8_kernels.modules.rms_norm import RMSNorm as QRMSNorm
from diffusers.models.normalization import RMSNorm
from q8_kernels.modules.activations import GELU as QGELU
from diffusers.models.activations import GELU
from q8_kernels.modules.linear import Q8Linear
from model.attention.attention_ltx import LTXVideoQ8AttentionProcessor
import argparse
from diffusers import LTXVideoTransformer3DModel
from q8_kernels.functional.quantizer import quantize
from q8_kernels.functional.fast_hadamard import hadamard_transform

MODULES_TO_NOT_CONVERT = ["proj_in", "time_embed", "caption_projection", "proj_out"]

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