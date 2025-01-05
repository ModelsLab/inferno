import json
import os
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
from safetensors import safe_open
import torch
from torch import Tensor

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from utils.ltx_utils import append_dims, make_hashable_key, diffusers_and_inferno_config_mapping


def simple_diffusion_resolution_dependent_timestep_shift(
    samples: Tensor,
    timesteps: Tensor,
    n: int = 32 * 32,
) -> Tensor:
    """Applies a resolution-dependent timestep shift using the simple diffusion method."""
    if len(samples.shape) == 3:
        _, m, _ = samples.shape
    elif len(samples.shape) in [4, 5]:
        m = math.prod(samples.shape[2:])
    else:
        raise ValueError(
            "Samples must have shape (b, t, c), (b, c, h, w) or (b, c, f, h, w)"
        )
    snr = (timesteps / (1 - timesteps)) ** 2
    shift_snr = torch.log(snr) + 2 * math.log(m / n)
    shifted_timesteps = torch.sigmoid(0.5 * shift_snr)

    return shifted_timesteps


def time_shift(mu: float, sigma: float, t: Tensor):
    """Applies a time shift transformation."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_normal_shift(
    n_tokens: int,
    min_tokens: int = 1024,
    max_tokens: int = 4096,
    min_shift: float = 0.95,
    max_shift: float = 2.05,
) -> float:
    """Calculates the normal shift based on token count."""
    m = (max_shift - min_shift) / (max_tokens - min_tokens)
    b = min_shift - m * min_tokens
    return m * n_tokens + b


def sd3_resolution_dependent_timestep_shift(
    samples: Tensor, timesteps: Tensor
) -> Tensor:
    """
    Shifts timesteps based on resolution using the SD3 method.
    
    For more details see:
    - SD3 paper: https://arxiv.org/pdf/2403.03206
    - Flux implementation: https://github.com/black-forest-labs/flux
    """
    if len(samples.shape) == 3:
        _, m, _ = samples.shape
    elif len(samples.shape) in [4, 5]:
        m = math.prod(samples.shape[2:])
    else:
        raise ValueError(
            "Samples must have shape (b, t, c), (b, c, h, w) or (b, c, f, h, w)"
        )

    shift = get_normal_shift(m)
    return time_shift(shift, 1, timesteps)


class TimestepShifter(ABC):
    """Abstract base class for timestep shifters."""
    @abstractmethod
    def shift_timesteps(self, samples: Tensor, timesteps: Tensor) -> Tensor:
        pass


@dataclass
class RectifiedFlowOutput:
    """
    Output class for the scheduler's step function.
    
    Args:
        prev_sample: Previous timestep sample (x_{t-1})
        pred_original_sample: Optional predicted denoised sample (x_{0})
    """
    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


class RectifiedFlowScheduler(SchedulerMixin, ConfigMixin, TimestepShifter):
    """
    A standalone implementation of the Rectified Flow scheduler.
    """
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shifting: Optional[str] = None,
        base_resolution: int = 32**2,
    ):
        """
        Initialize the scheduler.

        Args:
            num_train_timesteps: Number of training timesteps
            shifting: Type of timestep shifting to use ("SD3" or "SimpleDiffusion")
            base_resolution: Base resolution for shifting calculations
        """
        super().__init__()
        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = self.sigmas = torch.linspace(
            1, 1 / num_train_timesteps, num_train_timesteps
        )
        self.delta_timesteps = self.timesteps - torch.cat(
            [self.timesteps[1:], torch.zeros_like(self.timesteps[-1:])]
        )
        self.shifting = shifting
        self.base_resolution = base_resolution

    def shift_timesteps(self, samples: Tensor, timesteps: Tensor) -> Tensor:
        """Apply timestep shifting based on the configured method."""
        if self.shifting == "SD3":
            return sd3_resolution_dependent_timestep_shift(samples, timesteps)
        elif self.shifting == "SimpleDiffusion":
            return simple_diffusion_resolution_dependent_timestep_shift(
                samples, timesteps, self.base_resolution
            )
        return timesteps

    def set_timesteps(
        self,
        num_inference_steps: int,
        samples: Tensor,
        device: Union[str, torch.device] = None,
    ):
        """
        Set up discrete timesteps for the diffusion chain.

        Args:
            num_inference_steps: Number of diffusion steps for generation
            samples: Batch of samples 
            device: Device to move timesteps to
        """
        num_inference_steps = min(self.num_train_timesteps, num_inference_steps)
        timesteps = torch.linspace(1, 1 / num_inference_steps, num_inference_steps).to(
            device
        )
        self.timesteps = self.shift_timesteps(samples, timesteps)
        self.delta_timesteps = self.timesteps - torch.cat(
            [self.timesteps[1:], torch.zeros_like(self.timesteps[-1:])]
        )
        self.num_inference_steps = num_inference_steps
        self.sigmas = self.timesteps

    @staticmethod
    def from_pretrained(pretrained_model_path: Union[str, os.PathLike]):
        pretrained_model_path = Path(pretrained_model_path)
        if pretrained_model_path.is_file():
            comfy_single_file_state_dict = {}
            with safe_open(pretrained_model_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                for k in f.keys():
                    comfy_single_file_state_dict[k] = f.get_tensor(k)
            configs = json.loads(metadata["config"])
            config = configs["scheduler"]
            del comfy_single_file_state_dict

        elif pretrained_model_path.is_dir():
            diffusers_noise_scheduler_config_path = (
                pretrained_model_path / "scheduler" / "scheduler_config.json"
            )

            with open(diffusers_noise_scheduler_config_path, "r") as f:
                scheduler_config = json.load(f)
            hashable_config = make_hashable_key(scheduler_config)
            if hashable_config in diffusers_and_inferno_config_mapping:
                config = diffusers_and_inferno_config_mapping[hashable_config]
        return RectifiedFlowScheduler.from_config(config)

    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Optional[int] = None
    ) -> torch.FloatTensor:
        """
        Scale the denoising model input. In this implementation, returns the sample unchanged.
        """
        return sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: torch.FloatTensor,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[RectifiedFlowOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE.

        Args:
            model_output: Direct output from learned diffusion model
            timestep: Current timestep in the diffusion chain
            sample: Current instance of sample being diffused
            eta: Weight of noise for added noise
            use_clipped_model_output: Whether to use clipped model output
            generator: Random number generator
            variance_noise: Optional pre-generated noise
            return_dict: Whether to return output as a dataclass

        Returns:
            Either RectifiedFlowOutput or a tuple containing the previous sample
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if timestep.ndim == 0:
            # Global timestep
            current_index = (self.timesteps - timestep).abs().argmin()
            dt = self.delta_timesteps.gather(0, current_index.unsqueeze(0))
        else:
            # Timestep per token
            assert timestep.ndim == 2
            current_index = (
                (self.timesteps[:, None, None] - timestep[None]).abs().argmin(dim=0)
            )
            dt = self.delta_timesteps[current_index]
            # Special treatment for zero timestep tokens
            dt = torch.where(timestep == 0.0, torch.zeros_like(dt), dt)[..., None]

        prev_sample = sample - dt * model_output

        if not return_dict:
            return (prev_sample,)

        return RectifiedFlowOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Add noise to samples at specified timesteps.

        Args:
            original_samples: Clean samples
            noise: Noise to add
            timesteps: Timesteps at which to add noise

        Returns:
            Noisy samples
        """
        sigmas = timesteps
        sigmas = append_dims(sigmas, original_samples.ndim)
        alphas = 1 - sigmas
        noisy_samples = alphas * original_samples + sigmas * noise
        return noisy_samples