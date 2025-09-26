"""
Qwen Sampler Wrapper - DiffSynth FlowMatch scheduler implementation for ComfyUI.

This module provides a custom sampler that implements DiffSynth's exact
FlowMatch scheduling logic, crucial for proper Qwen Image generation.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm import tqdm

from .qwen_wrapper_base import (
    QwenWrapperBase,
    QwenConditioningWrapper,
    DiffSynthSchedulerWrapper,
    QWEN_VAE_CHANNELS
)

logger = logging.getLogger(__name__)


class FlowMatchSampler:
    """
    Implementation of DiffSynth's FlowMatch scheduler.

    From DiffSynth pipeline line 57:
    FlowMatchScheduler(sigma_min=0, sigma_max=1, extra_one_step=True,
                      exponential_shift=True, exponential_shift_mu=0.8,
                      shift_terminal=0.02)
    """

    def __init__(self):
        self.sigma_min = 0.0
        self.sigma_max = 1.0
        self.extra_one_step = True
        self.exponential_shift = True
        self.exponential_shift_mu = 0.8
        self.shift_terminal = 0.02
        self.training = False  # Set to False for inference

    def set_timesteps(
        self,
        num_inference_steps: int,
        denoising_strength: float = 1.0,
        dynamic_shift_len: Optional[int] = None,
        exponential_shift_mu: Optional[float] = None,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Set timesteps with DiffSynth's exact scheduling logic.

        Based on DiffSynth's FlowMatchScheduler.set_timesteps method.
        """
        if exponential_shift_mu is not None:
            self.exponential_shift_mu = exponential_shift_mu

        # Calculate number of steps
        total_steps = num_inference_steps
        if self.extra_one_step:
            total_steps += 1

        if self.exponential_shift:
            # Apply exponential shift as in DiffSynth
            # This creates a non-linear timestep schedule
            steps = torch.linspace(0, 1, total_steps, device=device)

            # Apply exponential transformation
            mu = self.exponential_shift_mu
            steps = (torch.exp(mu * steps) - 1) / (torch.exp(torch.tensor(mu)) - 1)

            # Apply terminal shift
            steps = steps * (1 - self.shift_terminal) + self.shift_terminal
        else:
            # Linear schedule
            steps = torch.linspace(self.sigma_min, self.sigma_max, total_steps, device=device)

        # Apply denoising strength
        if denoising_strength < 1.0:
            # Start from a later timestep based on denoising strength
            start_idx = int((1.0 - denoising_strength) * len(steps))
            steps = steps[start_idx:]

        # Reverse for denoising process (go from noise to clean)
        self.timesteps = steps.flip(0)

        logger.debug(f"Set {len(self.timesteps)} timesteps with exponential_shift={self.exponential_shift}, mu={self.exponential_shift_mu}")
        return self.timesteps

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to samples at given timestep (for img2img).

        Based on FlowMatch forward diffusion process.
        """
        # Ensure timestep is in [0, 1] range
        if timestep.max() > 1:
            timestep = timestep / 1000.0

        # FlowMatch noise addition
        # x_t = (1 - t) * x_0 + t * noise
        alpha = 1.0 - timestep
        sigma = timestep

        if alpha.dim() == 0:
            alpha = alpha.unsqueeze(0)
            sigma = sigma.unsqueeze(0)

        # Reshape for broadcasting
        while alpha.dim() < original_samples.dim():
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)

        noisy_samples = alpha * original_samples + sigma * noise

        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict]:
        """
        Perform one step of the denoising process.

        Based on FlowMatch/Rectified Flow ODE solver.
        """
        # Get current and next timestep
        t = timestep
        if t.max() > 1:
            t = t / 1000.0

        # Find the index of current timestep
        timestep_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]

        # Get next timestep (if not last step)
        if timestep_idx < len(self.timesteps) - 1:
            t_next = self.timesteps[timestep_idx + 1]
            if t_next.max() > 1:
                t_next = t_next / 1000.0
        else:
            t_next = torch.tensor(0.0, device=t.device, dtype=t.dtype)

        # FlowMatch update rule (simplified Euler step)
        # dx/dt = v(x_t, t) where v is the velocity (model output)
        # x_{t-dt} = x_t - dt * v(x_t, t)
        dt = t - t_next  # Note: negative because we're going backwards

        # Reshape dt for broadcasting
        while dt.dim() < sample.dim():
            dt = dt.unsqueeze(-1)

        # Update sample
        prev_sample = sample - dt * model_output

        if return_dict:
            return {"prev_sample": prev_sample, "pred_original_sample": None}
        return prev_sample


class QwenFlowMatchSamplerNode(QwenWrapperBase):
    """
    ComfyUI node for DiffSynth's FlowMatch sampling.

    This replaces ComfyUI's standard KSampler with DiffSynth's exact
    FlowMatch implementation for proper Qwen Image generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "steps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 200
                }),
                "cfg": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "exponential_shift_mu": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "shift_terminal": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01
                }),
                "height": ("INT", {
                    "default": 1328,
                    "min": 256,
                    "max": 4096,
                    "step": 32
                }),
                "width": ("INT", {
                    "default": 1328,
                    "min": 256,
                    "max": 4096,
                    "step": 32
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "QwenWrapper/Sampling"
    DISPLAY_NAME = "Qwen FlowMatch Sampler (DiffSynth)"

    def sample(
        self,
        model,
        positive,
        negative,
        latent,
        seed,
        steps,
        cfg,
        denoise,
        exponential_shift_mu=0.8,
        shift_terminal=0.02,
        height=1328,
        width=1328
    ):
        """
        Perform sampling using DiffSynth's FlowMatch scheduler.
        """
        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Ensure resolution is compatible
        height, width = self.prepare_resolution(height, width)

        # Extract latent samples
        if isinstance(latent, dict):
            latent_samples = latent.get("samples")
            batch_size = latent_samples.shape[0]
        else:
            latent_samples = latent
            batch_size = latent_samples.shape[0]

        # Ensure 16-channel format
        latent_samples = self.ensure_16_channel(latent_samples)

        # Initialize scheduler
        scheduler = FlowMatchSampler()
        scheduler.exponential_shift_mu = exponential_shift_mu
        scheduler.shift_terminal = shift_terminal

        # Calculate dynamic shift based on resolution
        dynamic_shift_len = (height // 16) * (width // 16)

        # Set timesteps
        timesteps = scheduler.set_timesteps(
            num_inference_steps=steps,
            denoising_strength=denoise,
            dynamic_shift_len=dynamic_shift_len,
            exponential_shift_mu=exponential_shift_mu,
            device=latent_samples.device
        )

        # Generate or use existing noise
        if denoise == 1.0:
            # Full denoising - start from pure noise
            noise = torch.randn_like(latent_samples)
            latents = noise
        else:
            # Partial denoising - add noise to existing latent
            noise = torch.randn_like(latent_samples)
            # Add noise at the starting timestep
            latents = scheduler.add_noise(latent_samples, noise, timesteps[0])

        # Extract conditioning
        pos_cond = QwenConditioningWrapper.unpack_conditioning(positive)
        neg_cond = QwenConditioningWrapper.unpack_conditioning(negative)

        # Sampling loop
        logger.info(f"Starting FlowMatch sampling: {steps} steps, cfg={cfg}, denoise={denoise}")

        for i, timestep in enumerate(tqdm(timesteps, desc="FlowMatch Sampling")):
            # Expand timestep to batch dimension
            t = timestep.unsqueeze(0).expand(batch_size)

            # Pack the current latents for model input
            packed_latents = self.pack_latents(latents, height, width)

            # Prepare model inputs from conditioning
            model_kwargs = {
                "prompt_emb": pos_cond.get("prompt_emb"),
                "prompt_emb_mask": pos_cond.get("prompt_emb_mask"),
                "edit_latents": pos_cond.get("edit_latents"),
                "context_latents": pos_cond.get("context_latents"),
            }

            # Forward through model (positive)
            if hasattr(model, 'apply_model'):
                # Use wrapped model's apply_model
                noise_pred_pos = model.apply_model(
                    packed_latents,
                    t,
                    transformer_options={"height": height, "width": width},
                    **model_kwargs
                )
            else:
                # Fallback to standard forward
                noise_pred_pos = model(packed_latents, t, **model_kwargs)

            # Apply CFG if needed
            if cfg != 1.0 and neg_cond:
                # Negative prediction
                neg_kwargs = {
                    "prompt_emb": neg_cond.get("prompt_emb"),
                    "prompt_emb_mask": neg_cond.get("prompt_emb_mask"),
                    "edit_latents": neg_cond.get("edit_latents"),
                    "context_latents": neg_cond.get("context_latents"),
                }

                if hasattr(model, 'apply_model'):
                    noise_pred_neg = model.apply_model(
                        packed_latents,
                        t,
                        transformer_options={"height": height, "width": width},
                        **neg_kwargs
                    )
                else:
                    noise_pred_neg = model(packed_latents, t, **neg_kwargs)

                # CFG formula
                noise_pred = noise_pred_neg + cfg * (noise_pred_pos - noise_pred_neg)
            else:
                noise_pred = noise_pred_pos

            # Unpack the noise prediction
            noise_pred = self.unpack_latents(noise_pred, height, width)

            # Perform scheduler step
            latents = scheduler.step(noise_pred, timestep, latents)

        # Return result in ComfyUI format
        result = {"samples": latents}

        logger.info(f"Sampling complete. Final latent shape: {latents.shape}")
        return (result,)


class QwenSchedulerInfoNode:
    """
    Utility node to inspect and visualize FlowMatch scheduler settings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 200
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "exponential_shift_mu": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "shift_terminal": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01
                }),
                "height": ("INT", {
                    "default": 1328,
                    "min": 256,
                    "max": 4096,
                    "step": 32
                }),
                "width": ("INT", {
                    "default": 1328,
                    "min": 256,
                    "max": 4096,
                    "step": 32
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_scheduler_info"
    CATEGORY = "QwenWrapper/Utils"
    DISPLAY_NAME = "Qwen Scheduler Info (DiffSynth)"
    OUTPUT_NODE = True

    def get_scheduler_info(
        self,
        steps,
        denoise,
        exponential_shift_mu,
        shift_terminal,
        height,
        width
    ):
        """
        Generate information about the scheduler settings.
        """
        # Create scheduler
        scheduler = FlowMatchSampler()
        scheduler.exponential_shift_mu = exponential_shift_mu
        scheduler.shift_terminal = shift_terminal

        # Calculate dynamic shift
        dynamic_shift_len = (height // 16) * (width // 16)

        # Set timesteps
        timesteps = scheduler.set_timesteps(
            num_inference_steps=steps,
            denoising_strength=denoise,
            dynamic_shift_len=dynamic_shift_len,
            exponential_shift_mu=exponential_shift_mu,
            device="cpu"
        )

        # Create info string
        info_lines = [
            "FlowMatch Scheduler Settings:",
            f"  Steps: {steps}",
            f"  Denoise Strength: {denoise:.2f}",
            f"  Exponential Shift μ: {exponential_shift_mu:.2f}",
            f"  Terminal Shift: {shift_terminal:.3f}",
            f"  Resolution: {width}x{height}",
            f"  Dynamic Shift Length: {dynamic_shift_len}",
            f"  Extra One Step: {scheduler.extra_one_step}",
            "",
            f"Timesteps ({len(timesteps)} total):",
        ]

        # Add first and last few timesteps
        if len(timesteps) <= 10:
            for i, t in enumerate(timesteps):
                info_lines.append(f"  [{i:3d}]: {t.item():.6f}")
        else:
            # Show first 3, middle, and last 3
            for i in range(3):
                info_lines.append(f"  [{i:3d}]: {timesteps[i].item():.6f}")
            info_lines.append("  ...")
            mid = len(timesteps) // 2
            info_lines.append(f"  [{mid:3d}]: {timesteps[mid].item():.6f}")
            info_lines.append("  ...")
            for i in range(len(timesteps) - 3, len(timesteps)):
                info_lines.append(f"  [{i:3d}]: {timesteps[i].item():.6f}")

        info = "\n".join(info_lines)
        return (info,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "QwenFlowMatchSamplerNode": QwenFlowMatchSamplerNode,
    "QwenSchedulerInfoNode": QwenSchedulerInfoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenFlowMatchSamplerNode": "Qwen FlowMatch Sampler (Wrapper)",
    "QwenSchedulerInfoNode": "Qwen Scheduler Info (Wrapper)",
}
