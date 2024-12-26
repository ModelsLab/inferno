from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import Tensor
from diffusers.configuration_utils import ConfigMixin
from einops import rearrange

from model.utils import append_dims


class BasePatchifier(ConfigMixin, ABC):
    """
    Abstract base class for patchifier implementations.
    Handles conversion between regular tensors and patched representations.
    """
    def __init__(self, patch_size: int):
        """
        Initialize the patchifier.
        
        Args:
            patch_size (int): Size of patches to use
        """
        self._patch_size = (1, patch_size, patch_size)

    @abstractmethod
    def patchify(
        self, latents: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Convert input tensor into patches.
        
        Args:
            latents (Tensor): Input tensor to be patchified
            
        Returns:
            Tuple[Tensor, Tensor]: Patchified representation
        """
        pass

    @abstractmethod
    def unpatchify(
        self,
        latents: Tensor,
        output_height: int,
        output_width: int,
        output_num_frames: int,
        out_channels: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Convert patched representation back to regular tensor.
        
        Args:
            latents (Tensor): Patched tensor to be unpatchified
            output_height (int): Desired output height
            output_width (int): Desired output width
            output_num_frames (int): Desired number of frames
            out_channels (int): Desired number of output channels
            
        Returns:
            Tuple[Tensor, Tensor]: Unpatchified tensor
        """
        pass

    @property
    def patch_size(self):
        """Get the patch size configuration."""
        return self._patch_size

    def get_grid(
        self, orig_num_frames, orig_height, orig_width, batch_size, scale_grid, device
    ):
        """
        Generate a grid of coordinates for the patches.
        
        Args:
            orig_num_frames (int): Original number of frames
            orig_height (int): Original height
            orig_width (int): Original width
            batch_size (int): Batch size
            scale_grid: Scaling factors for the grid
            device: Device to place tensors on
            
        Returns:
            Tensor: Coordinate grid for patches
        """
        f = orig_num_frames // self._patch_size[0]
        h = orig_height // self._patch_size[1]
        w = orig_width // self._patch_size[2]
        
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_f = torch.arange(f, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=0)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        if scale_grid is not None:
            for i in range(3):
                if isinstance(scale_grid[i], Tensor):
                    scale = append_dims(scale_grid[i], grid.ndim - 1)
                else:
                    scale = scale_grid[i]
                grid[:, i, ...] = grid[:, i, ...] * scale * self._patch_size[i]

        grid = rearrange(grid, "b c f h w -> b c (f h w)", b=batch_size)
        return grid


class Patchifier(BasePatchifier):
    """
    Implementation of symmetric patchification for tensors.
    Handles both patchification and unpatchification while maintaining symmetry.
    """
    
    def patchify(
        self,
        latents: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Convert input tensor into symmetric patches.
        
        Args:
            latents (Tensor): Input tensor to be patchified [b, c, f, h, w]
            
        Returns:
            Tuple[Tensor, Tensor]: Patchified representation [b, (f h w), (c p1 p2 p3)]
        """
        latents = rearrange(
            latents,
            "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
            p1=self._patch_size[0],
            p2=self._patch_size[1],
            p3=self._patch_size[2],
        )
        return latents

    def unpatchify(
        self,
        latents: Tensor,
        output_height: int,
        output_width: int,
        output_num_frames: int,
        out_channels: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Convert patched representation back to regular tensor with symmetric unpatchification.
        
        Args:
            latents (Tensor): Patched tensor [b, (f h w), (c p1 p2 p3)]
            output_height (int): Desired output height
            output_width (int): Desired output width 
            output_num_frames (int): Desired number of frames
            out_channels (int): Desired number of output channels
            
        Returns:
            Tuple[Tensor, Tensor]: Unpatchified tensor [b, c, f, h, w]
        """
        output_height = output_height // self._patch_size[1]
        output_width = output_width // self._patch_size[2]
        
        latents = rearrange(
            latents,
            "b (f h w) (c p q) -> b c f (h p) (w q)",
            f=output_num_frames,
            h=output_height,
            w=output_width,
            p=self._patch_size[1],
            q=self._patch_size[2],
        )
        return latents
    
def patchify(x, patch_size_hw, patch_size_t=1):
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

    return x

def unpatchify(x, patch_size_hw, patch_size_t=1):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

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