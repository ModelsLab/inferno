import torch
from some_library import VAE, TextEncoder, SanaModel  # Replace with actual imports


def get_vae(vae_type, vae_pretrained, device):
    vae = VAE(vae_type).from_pretrained(vae_pretrained)
    return vae.to(device)


def get_tokenizer_and_text_encoder(name, device):
    tokenizer = TextEncoder.get_tokenizer(name)
    text_encoder = TextEncoder(name).to(device)
    return tokenizer, text_encoder


def build_model(model_name, **kwargs):
    return SanaModel(model_name, **kwargs)


def find_model(model_path):
    import torch
    return torch.load(model_path, map_location="cpu")

