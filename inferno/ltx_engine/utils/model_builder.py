import torch
import json
import safetensors.torch
from transformers import (
    CLIPProcessor,
    CLIPModel,
    T5EncoderModel,
    T5Tokenizer
)
from model.autoencoder import CausalVideoAutoencoder
from model.transformer import Transformer3DModel
from model.patchifier import Patchifier
from scheduler.scheduler import RectifiedFlowScheduler

def get_device():
    """Get the appropriate device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vae(vae_dir, device):
    """
    Load and initialize the VAE model.
    
    Args:
        vae_dir (Path): Directory containing VAE model files
        device (torch.device): Device to load the model on
    
    Returns:
        CausalVideoAutoencoder: Initialized VAE model
    """
    vae_ckpt_path = vae_dir / "vae_diffusion_pytorch_model.safetensors"
    vae_config_path = vae_dir / "config.json"
    
    with open(vae_config_path, "r") as f:
        vae_config = json.load(f)
        
    vae = CausalVideoAutoencoder.from_config(vae_config)
    vae_state_dict = safetensors.torch.load_file(vae_ckpt_path)
    vae.load_state_dict(vae_state_dict)
    
    return vae.to(device=device, dtype=torch.bfloat16)

def load_transformer(transformer_dir, device):
    """
    Load and initialize the transformer model for LTX.
    
    Args:
        unet_dir (Path): Directory containing Transformer3DModel model files
        device (torch.device): Device to load the model on
    
    Returns:
        Transformer3DModel: Initialized transformer model
    """
    transformer_ckpt_path = transformer_dir / "unet_diffusion_pytorch_model.safetensors"  # LTX has apprently named their safetensors as unet
    transformer_config_path = transformer_dir / "config.json"
    
    transformer_config = Transformer3DModel.load_config(transformer_config_path)
    transformer = Transformer3DModel.from_config(transformer_config)
    transformer_state_dict = safetensors.torch.load_file(transformer_ckpt_path)
    transformer.load_state_dict(transformer_state_dict, strict=True)
    
    return transformer.to(device=device, dtype=torch.bfloat16)


def load_scheduler(scheduler_dir):
    """
    Load and initialize the scheduler.
    
    Args:
        scheduler_dir (Path): Directory containing scheduler config
    
    Returns:
        RectifiedFlowScheduler: Initialized scheduler
    """
    scheduler_config_path = scheduler_dir / "scheduler_config.json"
    scheduler_config = RectifiedFlowScheduler.load_config(scheduler_config_path)
    return RectifiedFlowScheduler.from_config(scheduler_config)

def load_clip_model(model_name, cache_dir=None, device=None):
    """
    Load CLIP model and processor.
    
    Args:
        model_name (str): Name of the CLIP model
        cache_dir (str, optional): Directory to cache the model
        device (torch.device, optional): Device to load the model on
    
    Returns:
        tuple: (CLIPModel, CLIPProcessor)
    """
    if device is None:
        device = get_device()
        
    clip_model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    
    return clip_model, clip_processor

def load_text_encoder(model_name, device=None):
    """
    Load text encoder model and tokenizer.
    
    Args:
        model_name (str): Name of the text encoder model
        device (torch.device, optional): Device to load the model on
    
    Returns:
        tuple: (T5EncoderModel, T5Tokenizer)
    """
    if device is None:
        device = get_device()
        
    text_encoder = T5EncoderModel.from_pretrained(
        model_name,
        subfolder="text_encoder"
    ).to(device)
    
    tokenizer = T5Tokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer"
    )
    
    return text_encoder, tokenizer

def get_patchifier(patch_size=1):
    """
    Initialize a symmetric patchifier.
    
    Args:
        patch_size (int): Size of the patches
    
    Returns:
        Patchifier: Initialized patchifier
    """
    return Patchifier(patch_size=patch_size)