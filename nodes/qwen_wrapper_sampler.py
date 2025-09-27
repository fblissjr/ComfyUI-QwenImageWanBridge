"""
Qwen Image Sampler Wrapper with FlowMatch scheduling.

This module implements the FlowMatch scheduler from DiffSynth with proper
dynamic shift calculation based on resolution and Qwen-specific settings.
"""

import torch
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, List
import comfy.samplers
import comfy.model_management as mm
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Set up verbose logging
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class QwenFlowMatchScheduler:
    """
    FlowMatch scheduler with DiffSynth-specific settings for Qwen.

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
        self.num_train_timesteps = 1000

    def set_timesteps(self, num_steps: int, denoising_strength: float = 1.0,
                      height: int = 1024, width: int = 1024) -> torch.Tensor:
        """
        Set timesteps with dynamic shift based on resolution.

        From DiffSynth line 399: dynamic_shift_len=(height // 16) * (width // 16)
        """
        # Calculate dynamic shift based on resolution
        dynamic_shift_len = (height // 16) * (width // 16)

        # Generate base timesteps
        if self.extra_one_step:
            steps = torch.linspace(0, 1, num_steps + 1)
        else:
            steps = torch.linspace(0, 1, num_steps)

        if self.exponential_shift:
            # Apply exponential transformation with mu
            shift_mu = self.exponential_shift_mu

            # Adjust shift based on resolution
            # Larger resolutions need adjusted shift values
            resolution_factor = np.sqrt(dynamic_shift_len / ((1024 // 16) ** 2))
            adjusted_mu = shift_mu * (1 + 0.1 * np.log2(resolution_factor))
            adjusted_mu = np.clip(adjusted_mu, 0.5, 0.95)

            # Apply exponential transformation
            steps = torch.exp(adjusted_mu * steps) - 1
            steps = steps / (torch.exp(torch.tensor(adjusted_mu)) - 1)

            # Apply terminal shift
            steps = steps * (1 - self.shift_terminal) + self.shift_terminal

        # Apply denoising strength
        if denoising_strength < 1.0:
            num_steps_to_use = max(1, int(num_steps * denoising_strength))
            steps = steps[-num_steps_to_use:]

        # Convert to timesteps in milliseconds for compatibility
        timesteps = steps * 1000

        return timesteps

    def add_noise(self, original: torch.Tensor, noise: torch.Tensor,
                  timestep: torch.Tensor) -> torch.Tensor:
        """Add noise for forward diffusion process."""
        # FlowMatch uses simple interpolation
        alpha = (timestep / 1000).view(-1, 1, 1, 1)
        noisy = (1 - alpha) * original + alpha * noise
        return noisy

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor,
                     timestep: torch.Tensor) -> torch.Tensor:
        """Calculate velocity for FlowMatch."""
        # Velocity is the difference between noise and sample
        return noise - sample

    def step(self, model_output: torch.Tensor, sample: torch.Tensor,
             timestep: torch.Tensor, prev_timestep: torch.Tensor) -> torch.Tensor:
        """Single denoising step."""
        # FlowMatch uses velocity prediction
        dt = (prev_timestep - timestep) / 1000
        dt = dt.view(-1, 1, 1, 1)

        # Update sample with velocity
        sample = sample + model_output * dt

        return sample


class QwenImageSamplerWrapper:
    """
    Custom sampler wrapper for Qwen Image with FlowMatch scheduling.

    This integrates DiffSynth's scheduling with ComfyUI's sampling system.
    """

    def __init__(self, model=None, scheduler=None):
        self.model = model
        self.scheduler = scheduler or QwenFlowMatchScheduler()
        self.device = mm.get_torch_device()

    def sample(
        self,
        latents: torch.Tensor,
        positive: List,
        negative: List,
        num_steps: int = 50,
        cfg_scale: float = 7.0,
        denoising_strength: float = 1.0,
        height: int = 1024,
        width: int = 1024,
        seed: int = None,
        callback: Optional[callable] = None
    ) -> torch.Tensor:
        """
        Perform sampling with FlowMatch scheduler.

        Args:
            latents: Initial latent tensor [B, C, H, W]
            positive: Positive conditioning from encoder
            negative: Negative conditioning (optional)
            num_steps: Number of denoising steps
            cfg_scale: Classifier-free guidance scale
            denoising_strength: How much to denoise (1.0 = full)
            height: Image height for dynamic shift
            width: Image width for dynamic shift
            seed: Random seed for noise
            callback: Progress callback

        Returns:
            Denoised latents
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Get timesteps with dynamic shift
        timesteps = self.scheduler.set_timesteps(
            num_steps, denoising_strength, height, width
        )

        # Initialize with noise if full denoising
        if denoising_strength >= 1.0:
            # Full generation from noise
            noise = torch.randn_like(latents)
            sample = noise
        else:
            # Partial denoising - add noise to input
            noise = torch.randn_like(latents)
            sample = self.scheduler.add_noise(latents, noise, timesteps[0])

        # Extract conditioning
        pos_cond = self.unpack_conditioning(positive)
        neg_cond = self.unpack_conditioning(negative) if negative else None

        # Sampling loop
        for i, timestep in enumerate(tqdm(timesteps, desc="Sampling")):
            # Prepare model inputs
            model_input = sample

            # Classifier-free guidance
            if cfg_scale > 1.0 and neg_cond is not None:
                # Duplicate inputs for CFG
                model_input = torch.cat([model_input, model_input])
                timestep_input = torch.cat([timestep.unsqueeze(0), timestep.unsqueeze(0)])

                # Combine positive and negative conditioning
                combined_cond = self.combine_conditioning(pos_cond, neg_cond)
            else:
                timestep_input = timestep.unsqueeze(0)
                combined_cond = pos_cond

            # Call model
            if self.model is not None:
                with torch.no_grad():
                    # Check if model has a forward method or is callable
                    if hasattr(self.model, 'forward'):
                        model_output = self.model.forward(
                            model_input,
                            timestep_input,
                            combined_cond,
                            height=height,
                            width=width
                        )
                    elif callable(self.model):
                        model_output = self.model(
                            model_input,
                            timestep_input,
                            combined_cond,
                            height=height,
                            width=width
                        )
                    else:
                        logger.error(f"Model is not callable: {type(self.model)}")
                        model_output = torch.randn_like(model_input)
            else:
                # Fallback for testing
                model_output = torch.randn_like(model_input)

            # Apply CFG
            if cfg_scale > 1.0 and neg_cond is not None:
                noise_pred_uncond, noise_pred_cond = model_output.chunk(2)
                model_output = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

            # Compute previous timestep
            if i < len(timesteps) - 1:
                prev_timestep = timesteps[i + 1]
            else:
                prev_timestep = torch.tensor(0.0)

            # Perform step
            sample = self.scheduler.step(model_output, sample, timestep, prev_timestep)

            # Progress callback
            if callback is not None:
                callback(i, num_steps, sample)

        return sample

    def unpack_conditioning(self, conditioning: List) -> Dict[str, torch.Tensor]:
        """Extract conditioning tensors from ComfyUI format."""
        if not conditioning or not isinstance(conditioning, list):
            return {}

        result = {}
        if len(conditioning) > 0 and isinstance(conditioning[0], list):
            if len(conditioning[0]) >= 2:
                result["prompt_emb"] = conditioning[0][0]

                cond_dict = conditioning[0][1]
                if isinstance(cond_dict, dict):
                    result["prompt_emb_mask"] = cond_dict.get("attention_mask")
                    result["edit_latents"] = cond_dict.get("edit_latents")
                    result["context_latents"] = cond_dict.get("context_latents")
                    result["pooled_output"] = cond_dict.get("pooled_output")

        return result

    def combine_conditioning(self, pos_cond: Dict, neg_cond: Dict) -> Dict:
        """Combine positive and negative conditioning for CFG."""
        combined = {}

        # Stack embeddings
        if "prompt_emb" in pos_cond and "prompt_emb" in neg_cond:
            combined["prompt_emb"] = torch.cat([neg_cond["prompt_emb"], pos_cond["prompt_emb"]])

        if "prompt_emb_mask" in pos_cond and "prompt_emb_mask" in neg_cond:
            if pos_cond["prompt_emb_mask"] is not None and neg_cond["prompt_emb_mask"] is not None:
                combined["prompt_emb_mask"] = torch.cat([neg_cond["prompt_emb_mask"], pos_cond["prompt_emb_mask"]])
            elif pos_cond["prompt_emb_mask"] is not None:
                # If only positive has mask, duplicate for negative
                combined["prompt_emb_mask"] = torch.cat([pos_cond["prompt_emb_mask"], pos_cond["prompt_emb_mask"]])

        # Copy other fields from positive only
        for key in ["edit_latents", "context_latents", "pooled_output"]:
            if key in pos_cond:
                combined[key] = pos_cond[key]

        return combined


class QwenImageSamplerNode:
    """
    ComfyUI node for Qwen Image sampling with FlowMatch scheduler.

    This node provides a custom sampler specifically tuned for Qwen Image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "tooltip": "Number of denoising steps"
                }),
                "cfg": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": "Classifier-free guidance scale"
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Denoising strength (1.0 = full generation)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**32 - 1,
                    "tooltip": "Random seed"
                }),
            },
            "optional": {
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 32,
                    "tooltip": "Target height for dynamic shift"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 32,
                    "tooltip": "Target width for dynamic shift"
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "sample"
    CATEGORY = "QwenImage/Sampling"
    TITLE = "Qwen Image Sampler (FlowMatch)"
    DESCRIPTION = "Custom sampler with FlowMatch scheduling for Qwen Image"

    def sample(self, model, positive, negative, latent, steps, cfg, denoise,
               seed, height=1024, width=1024):
        """Perform sampling with FlowMatch scheduler."""

        # Extract latent tensor
        latent_image = latent["samples"]

        # Get actual dimensions from latent if not specified
        if height == 1024 and width == 1024:
            # Use latent dimensions (scaled up by 8x)
            height = latent_image.shape[2] * 8
            width = latent_image.shape[3] * 8

        # Ensure dimensions are divisible by 32
        height = (height // 32) * 32
        width = (width // 32) * 32

        # Extract the actual model if it's wrapped
        if hasattr(model, 'get_model_object'):
            actual_model = model.get_model_object()
        elif hasattr(model, 'diffusion_model'):
            actual_model = model.diffusion_model
        elif hasattr(model, 'model'):
            actual_model = model.model
        else:
            actual_model = model

        # Create sampler with the actual model
        sampler = QwenImageSamplerWrapper(actual_model)

        # Perform sampling
        samples = sampler.sample(
            latents=latent_image,
            positive=positive,
            negative=negative,
            num_steps=steps,
            cfg_scale=cfg,
            denoising_strength=denoise,
            height=height,
            width=width,
            seed=seed
        )

        # Return in ComfyUI format
        return ({"samples": samples},)


class QwenSchedulerNode:
    """
    Standalone FlowMatch scheduler node for testing and visualization.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponential_shift": ("BOOLEAN", {"default": True}),
                "shift_mu": ("FLOAT", {"default": 0.8, "min": 0.5, "max": 0.95, "step": 0.05}),
                "shift_terminal": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.1, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = "QwenImage/Scheduling"
    TITLE = "Qwen FlowMatch Scheduler"
    DESCRIPTION = "Generate timesteps with FlowMatch scheduling for Qwen"

    def get_sigmas(self, steps, height, width, denoise, exponential_shift,
                   shift_mu, shift_terminal):
        """Generate timesteps/sigmas."""

        scheduler = QwenFlowMatchScheduler()
        scheduler.exponential_shift = exponential_shift
        scheduler.exponential_shift_mu = shift_mu
        scheduler.shift_terminal = shift_terminal

        timesteps = scheduler.set_timesteps(steps, denoise, height, width)

        # Convert to sigmas for ComfyUI compatibility
        # FlowMatch uses timesteps directly, but ComfyUI expects sigmas
        sigmas = timesteps / 1000  # Normalize to [0, 1]

        logger.info(f"Generated {len(sigmas)} timesteps with dynamic shift for {height}x{width}")
        logger.info(f"Timestep range: {sigmas.min():.3f} to {sigmas.max():.3f}")

        return (sigmas,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenImageSamplerNode": QwenImageSamplerNode,
    # QwenSchedulerNode removed - sampler has built-in FlowMatch scheduling
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageSamplerNode": "Qwen Image Sampler (FlowMatch)",
    # QwenSchedulerNode removed - not needed for workflow
}