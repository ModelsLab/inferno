import math
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn


def get_3d_sincos_pos_embed(embed_dim, grid, w, h, f):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = rearrange(grid, "c (f h w) -> c f h w", h=h, w=w)
    grid = rearrange(grid, "c f h w -> c h w f", h=h, w=w)
    grid = grid.reshape([3, 1, w, h, f])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = pos_embed.transpose(1, 0, 2, 3)
    return rearrange(pos_embed, "h w f c -> (f h w) c")


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 3 != 0:
        raise ValueError("embed_dim must be divisible by 3")

    # use half of dimensions to encode grid_h
    emb_f = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W*T, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W*T, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W*T, D/3)

    emb = np.concatenate([emb_h, emb_w, emb_f], axis=-1)  # (H*W*T, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos_shape = pos.shape

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    out = out.reshape([*pos_shape, -1])[0]

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (M, D)
    return emb

class SinusoidalPositionalEmbedding(nn.Module):
    """Apply positional information to a sequence of embeddings.
    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them
    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings
    """

    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        _, seq_length, _ = x.shape
        x = x + self.pe[:, :seq_length]
        return x