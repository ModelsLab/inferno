from diffusers import LTXPipeline, LTXVideoTransformer3DModel
from huggingface_hub import hf_hub_download
import argparse
import os
from q8_ltx import check_transformer_replaced_correctly, replace_gelu, replace_linear, replace_rms_norm
import safetensors.torch
from q8_kernels.graph.graph import make_dynamic_graphed_callable
import torch
import gc
from diffusers.utils import export_to_video
import numpy as np
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import random
from enum import Enum, auto

class SkipLayerStrategy(Enum):
    Attention = auto()
    Residual = auto()

def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    # Remove non-letters and convert to lowercase
    clean_text = "".join(char.lower() for char in text if char.isalpha() or char.isspace())
    # Split into words
    words = clean_text.split()
    
    # Build result string keeping track of length
    result = []
    current_length = 0
    
    for word in words:
        # Add word length plus 1 for underscore (except for first word)
        new_length = current_length + len(word)
        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break
            
    return "-".join(result)

def load_image_to_tensor_with_resize_and_crop(image_path, target_height=512, target_width=768):
    image = Image.open(image_path).convert("RGB")
    input_width, input_height = image.size
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = input_width / input_height
    
    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(input_height * aspect_ratio_target)
        new_height = input_height
        x_start = (input_width - new_width) // 2
        y_start = 0
    else:
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)
        x_start = 0
        y_start = (input_height - new_height) // 2

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    image = image.resize((target_width, target_height))
    frame_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()
    frame_tensor = (frame_tensor / 127.5) - 1.0
    # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
    return frame_tensor.unsqueeze(0).unsqueeze(2)

def calculate_padding(source_height: int, source_width: int, target_height: int, target_width: int) -> tuple[int, int, int, int]:
    # Calculate total padding needed
    pad_height = target_height - source_height
    pad_width = target_width - source_width
    
    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top  # Handles odd padding
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left  # Handles odd padding
    
    return (pad_left, pad_right, pad_top, pad_bottom)

def load_text_encoding_pipeline():
    return LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video", transformer=None, vae=None, torch_dtype=torch.bfloat16
    ).to("cuda")

def encode_prompt(pipe, prompt, negative_prompt, max_sequence_length=128):
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = pipe.encode_prompt(
        prompt=prompt, negative_prompt=negative_prompt, max_sequence_length=max_sequence_length
    )
    return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

def load_q8_transformer(args):
    with torch.device("meta"):
        transformer_config = LTXVideoTransformer3DModel.load_config("Lightricks/LTX-Video", subfolder="transformer")
        transformer = LTXVideoTransformer3DModel.from_config(transformer_config)

    transformer = replace_gelu(transformer)[0]
    transformer = replace_linear(transformer)[0]
    transformer = replace_rms_norm(transformer)[0]

    if os.path.isfile(f"{args.q8_transformer_path}/diffusion_pytorch_model.safetensors"):
        state_dict = safetensors.torch.load_file(f"{args.q8_transformer_path}/diffusion_pytorch_model.safetensors")
    else:
        state_dict = safetensors.torch.load_file(
            hf_hub_download(args.q8_transformer_path, "diffusion_pytorch_model.safetensors")
        )
    transformer.load_state_dict(state_dict, strict=True, assign=True)
    check_transformer_replaced_correctly(transformer)
    return transformer

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

@torch.no_grad()
def main(args):
    if args.seed is not None:
        seed_everything(args.seed)

    text_encoding_pipeline = load_text_encoding_pipeline()
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = encode_prompt(
        pipe=text_encoding_pipeline,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        max_sequence_length=args.max_sequence_length,
    )
    del text_encoding_pipeline
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Handle input image if provided
    media_items = None
    if args.input_image_path:
        media_items_prepad = load_image_to_tensor_with_resize_and_crop(
            args.input_image_path, args.height, args.width
        )
        
        # Calculate padding to make dimensions divisible by 32
        height_padded = ((args.height - 1) // 32 + 1) * 32
        width_padded = ((args.width - 1) // 32 + 1) * 32
        padding = calculate_padding(args.height, args.width, height_padded, width_padded)
        
        media_items = F.pad(
            media_items_prepad, padding, mode="constant", value=-1
        )

    if args.q8_transformer_path:
        transformer = load_q8_transformer(args)
        pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", transformer=None, text_encoder=None)
        pipe.transformer = transformer

        pipe.transformer = pipe.transformer.to(torch.bfloat16)
        for b in pipe.transformer.transformer_blocks:
            b.to(dtype=torch.float)

        for n, m in pipe.transformer.transformer_blocks.named_parameters():
            if "scale_shift_table" in n:
                m.data = m.data.to(torch.bfloat16)

        pipe.transformer.forward = make_dynamic_graphed_callable(pipe.transformer.forward)
        pipe.vae = pipe.vae.to(torch.bfloat16)
    else:
        pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", text_encoder=None, torch_dtype=torch.bfloat16)

    pipe = pipe.to("cuda")

    # Set up STG parameters if enabled
    if args.stg_scale > 0:
        skip_block_list = [int(x.strip()) for x in args.stg_skip_layers.split(",")]
        skip_layer_strategy = (
            SkipLayerStrategy.Attention
            if args.stg_mode.lower() == "stg_a"
            else SkipLayerStrategy.Residual
        )
    else:
        skip_block_list = []
        skip_layer_strategy = None

    # Generate video
    video = pipe(
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        max_sequence_length=args.max_sequence_length,
        guidance_scale=args.guidance_scale,
        skip_layer_strategy=skip_layer_strategy,
        skip_block_list=skip_block_list,
        stg_scale=args.stg_scale,
        stg_rescale=args.stg_rescale,
        media_items=media_items,
        generator=torch.manual_seed(args.seed if args.seed is not None else 2025),
    ).frames[0]

    print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB.")

    # Handle output path and save video
    if args.out_path is None:
        output_dir = Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_filename = "img_to_vid" if args.input_image_path else "text_to_vid"
        filename_from_prompt = convert_prompt_to_filename(args.prompt, max_len=30)
        base_filename = f"{base_filename}_{filename_from_prompt}_{args.num_frames}x{args.height}x{args.width}"
        base_filename += "_q8" if args.q8_transformer_path is not None else ""
        args.out_path = str(output_dir / f"{base_filename}.mp4")

    export_to_video(video, args.out_path, fps=args.fps)

    # Save conditioning image if used
    if args.input_image_path:
        reference_image = ((media_items_prepad[0, :, 0].permute(1, 2, 0).cpu().data.numpy() + 1.0) / 2.0 * 255)
        ref_path = str(Path(args.out_path).with_name(Path(args.out_path).stem + "_condition.png"))
        Image.fromarray(reference_image.astype(np.uint8)).save(ref_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q8_transformer_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="worst quality, inconsistent motion, blurry, jittery, distorted")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--resolution", type=str, default="480x704")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--input_image_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--stg_scale", type=float, default=1.0)
    parser.add_argument("--stg_rescale", type=float, default=0.7)
    parser.add_argument("--stg_mode", type=str, default="stg_a")
    parser.add_argument("--stg_skip_layers", type=str, default="19")
    
    args = parser.parse_args()
    
    # Parse resolution into height and width
    width, height = args.resolution.split("x")[::-1]
    args.height = int(height)
    args.width = int(width)
    
    main(args)