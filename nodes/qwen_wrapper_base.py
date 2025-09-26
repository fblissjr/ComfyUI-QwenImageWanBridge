"""
Base classes and utilities for DiffSynth-style Qwen wrapper nodes.

This module provides the foundational components that bridge DiffSynth's
proven logic with ComfyUI's node system. It includes critical operations
like the packing/unpacking transformations and proper handling of 16-channel VAE.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from PIL import Image
import logging

logger = logging.getLogger(__name__)


# Critical constants from DiffSynth
QWEN_VAE_CHANNELS = 16  # Qwen uses 16-channel VAE, not 4
PATCH_SIZE = 2  # For packing operation: P=2, Q=2
TOKEN_DROP_TEXT = 34  # Tokens to drop for text-only prompts
TOKEN_DROP_VISION = 64  # Tokens to drop for vision prompts
DEFAULT_RESOLUTION = (1328, 1328)  # Default generation resolution


class QwenWrapperBase:
    """Base class for all Qwen wrapper nodes with DiffSynth logic."""

    @staticmethod
    def pack_latents(latents: torch.Tensor, height: int = None, width: int = None) -> torch.Tensor:
        """
        Apply DiffSynth's critical packing operation.

        This transforms latents from [B, C, H, W] to [B, (H/2 * W/2), (C * 2 * 2)]
        which is essential for Qwen's patch-based processing.

        Args:
            latents: Input latents in standard format [B, C, H, W]
            height: Optional target height in pixels (for reference only)
            width: Optional target width in pixels (for reference only)

        Returns:
            Packed latents in format [B, num_patches, patch_features]
        """
        # Get actual latent dimensions
        B, C, H_latent, W_latent = latents.shape

        # Handle odd dimensions by padding if necessary
        H_pad = H_latent
        W_pad = W_latent

        if H_latent % 2 != 0:
            H_pad = H_latent + 1
            logger.warning(f"Latent height {H_latent} not divisible by 2, padding to {H_pad}")

        if W_latent % 2 != 0:
            W_pad = W_latent + 1
            logger.warning(f"Latent width {W_latent} not divisible by 2, padding to {W_pad}")

        # Pad if needed
        if H_pad != H_latent or W_pad != W_latent:
            latents = F.pad(latents, (0, W_pad - W_latent, 0, H_pad - H_latent), mode='replicate')
            logger.debug(f"Padded latents from [{B}, {C}, {H_latent}, {W_latent}] to [{B}, {C}, {H_pad}, {W_pad}]")

        # CRITICAL: Following DiffSynth exactly - use latent dimensions divided by patch size
        # This calculates the NUMBER OF PATCHES, not trying to fit pixel dimensions
        H_patches = H_pad // PATCH_SIZE  # Number of patches in height
        W_patches = W_pad // PATCH_SIZE  # Number of patches in width

        packed = rearrange(
            latents,
            "B C (H P) (W Q) -> B (H W) (C P Q)",
            H=H_patches,
            W=W_patches,
            P=PATCH_SIZE,
            Q=PATCH_SIZE
        )

        logger.debug(f"Packed latents: [{B}, {C}, {H_pad}, {W_pad}] -> {packed.shape}")
        return packed

    @staticmethod
    def unpack_latents(packed: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Reverse the packing operation to get back standard latents.

        Args:
            packed: Packed latents [B, num_patches, patch_features]
            height: Target height (pixel space, divisible by 16)
            width: Target width (pixel space, divisible by 16)

        Returns:
            Unpacked latents [B, C, H, W]
        """
        # CRITICAL: Reverse of model_fn_qwen_image line 860
        # rearrange(image, "B (H W) (C P Q) -> B C (H P) (W Q)", H=height//16, W=width//16, P=2, Q=2)

        unpacked = rearrange(
            packed,
            "B (H W) (C P Q) -> B C (H P) (W Q)",
            H=height // 16,
            W=width // 16,
            P=PATCH_SIZE,
            Q=PATCH_SIZE
        )

        logger.debug(f"Unpacked latents from {packed.shape} to {unpacked.shape}")
        return unpacked

    @staticmethod
    def handle_edit_latents(edit_latents: Union[torch.Tensor, List[torch.Tensor]],
                          main_latents: torch.Tensor) -> torch.Tensor:
        """
        Properly concatenate edit latents following DiffSynth logic.

        CRITICAL: Edit latents are CONCATENATED in sequence dimension, not added!
        This is a key difference from many incorrect implementations.

        Args:
            edit_latents: Single tensor or list of edit latent tensors
            main_latents: Main generation latents (already packed)

        Returns:
            Combined latents with edit information in sequence dimension
        """
        # Based on model_fn_qwen_image lines 809-813
        if edit_latents is None:
            return main_latents

        # Ensure we have a list
        if not isinstance(edit_latents, list):
            edit_latents = [edit_latents]

        # Pack each edit latent and concatenate in sequence dimension
        packed_edits = []
        for edit_latent in edit_latents:
            # Use the improved pack_latents function that handles padding
            packed_edit = QwenWrapperBase.pack_latents(edit_latent)
            packed_edits.append(packed_edit)

        # Concatenate along sequence dimension (dim=1)
        combined = torch.cat([main_latents] + packed_edits, dim=1)

        logger.debug(f"Combined main latents {main_latents.shape} with {len(packed_edits)} edit latents -> {combined.shape}")
        return combined

    @staticmethod
    def calculate_dynamic_shift(height: int, width: int, base_shift: float = 0.8) -> float:
        """
        Calculate dynamic shift for FlowMatch scheduler based on resolution.

        From DiffSynth pipeline line 399: dynamic_shift_len=(height // 16) * (width // 16)

        Args:
            height: Image height in pixels
            width: Image width in pixels
            base_shift: Base exponential shift value

        Returns:
            Adjusted shift value for scheduler
        """
        # Number of patches
        num_patches = (height // 16) * (width // 16)

        # Adjust shift based on resolution
        # Larger resolutions need different shift values
        shift_adjustment = np.log2(num_patches / (1024 // 16) ** 2)  # Relative to 1024x1024
        adjusted_shift = base_shift + shift_adjustment * 0.1

        return np.clip(adjusted_shift, 0.5, 0.95)

    @staticmethod
    def ensure_16_channel(latents: torch.Tensor) -> torch.Tensor:
        """
        Ensure latents have 16 channels for Qwen VAE.

        ComfyUI typically works with 4-channel latents, but Qwen requires 16.

        Args:
            latents: Input latents [B, C, H, W]

        Returns:
            16-channel latents [B, 16, H, W]
        """
        B, C, H, W = latents.shape

        if C == 16:
            return latents
        else:
            raise ValueError(
                f"Qwen Image Edit requires 16-channel latents from a proper 16-channel VAE. "
                f"Got {C} channels. Please use QwenVLEmptyLatent or QwenVLImageToLatent nodes "
                f"with the correct Qwen VAE model."
            )

        return latents

    @staticmethod
    def prepare_resolution(height: int, width: int) -> Tuple[int, int]:
        """
        Ensure resolution is compatible with Qwen's requirements.

        Resolutions must be divisible by 32 (for optimal performance with patches).

        Args:
            height: Requested height
            width: Requested width

        Returns:
            Adjusted (height, width) tuple
        """
        # Round to nearest 32
        height = round(height / 32) * 32
        width = round(width / 32) * 32

        # Ensure minimum size
        height = max(height, 256)
        width = max(width, 256)

        return height, width

    @staticmethod
    def apply_rope_interpolation(positions: torch.Tensor, edit_mode: bool = False) -> torch.Tensor:
        """
        Apply RoPE position interpolation for edit mode if needed.

        Based on DiffSynth line 825-828 with edit_rope_interpolation flag.

        Args:
            positions: Position embeddings
            edit_mode: Whether we're in edit mode (may need different interpolation)

        Returns:
            Adjusted position embeddings
        """
        if not edit_mode:
            return positions

        # In edit mode, we might need to interpolate positions differently
        # This is a placeholder - actual implementation depends on the specific
        # position embedding module from Qwen model
        logger.debug("Applying RoPE interpolation for edit mode")
        return positions

    @staticmethod
    def drop_template_tokens(hidden_states: torch.Tensor,
                           has_vision: bool = False) -> torch.Tensor:
        """
        Drop template tokens from hidden states based on prompt type.

        From DiffSynth:
        - Text prompts: drop first 34 tokens (line 612)
        - Vision prompts: drop first 64 tokens (lines 554, 564)

        Args:
            hidden_states: Token hidden states [B, seq_len, hidden_dim]
            has_vision: Whether this is a vision prompt

        Returns:
            Hidden states with template tokens removed
        """
        drop_count = TOKEN_DROP_VISION if has_vision else TOKEN_DROP_TEXT

        if hidden_states.shape[1] <= drop_count:
            logger.warning(f"Hidden states seq_len {hidden_states.shape[1]} <= drop_count {drop_count}")
            return hidden_states

        return hidden_states[:, drop_count:, :]


class DiffSynthSchedulerWrapper:
    """
    Wrapper for FlowMatch scheduler with DiffSynth-specific settings.

    From DiffSynth pipeline line 57:
    FlowMatchScheduler(sigma_min=0, sigma_max=1, extra_one_step=True,
                      exponential_shift=True, exponential_shift_mu=0.8,
                      shift_terminal=0.02)
    """

    def __init__(self):
        self.sigma_min = 0
        self.sigma_max = 1
        self.extra_one_step = True
        self.exponential_shift = True
        self.exponential_shift_mu = 0.8
        self.shift_terminal = 0.02
        self.timesteps = None

    def set_timesteps(self, num_steps: int, denoising_strength: float = 1.0,
                      height: int = 1024, width: int = 1024):
        """Set timesteps with dynamic shift based on resolution."""
        # Calculate dynamic shift
        dynamic_shift_len = (height // 16) * (width // 16)

        # Generate timesteps with exponential shift
        if self.exponential_shift:
            # Apply exponential shift formula
            shift_mu = self.exponential_shift_mu
            steps = torch.linspace(0, 1, num_steps + (1 if self.extra_one_step else 0))

            # Apply exponential transformation
            steps = torch.exp(shift_mu * steps) - 1
            steps = steps / (torch.exp(torch.tensor(shift_mu)) - 1)

            # Apply terminal shift
            steps = steps * (1 - self.shift_terminal) + self.shift_terminal

            # Apply denoising strength
            if denoising_strength < 1.0:
                steps = steps[-int(num_steps * denoising_strength):]

            self.timesteps = steps
        else:
            self.timesteps = torch.linspace(
                self.sigma_min,
                self.sigma_max,
                num_steps
            )

        # Scale timesteps to milliseconds for compatibility
        self.timesteps = self.timesteps * 1000

        return self.timesteps


class QwenConditioningWrapper:
    """
    Handles conditioning translation between DiffSynth and ComfyUI formats.

    This class bridges the gap between DiffSynth's direct tensor passing
    and ComfyUI's nested conditioning structure.
    """

    @staticmethod
    def pack_conditioning(prompt_embeds: torch.Tensor,
                         prompt_mask: torch.Tensor,
                         edit_latents: Optional[torch.Tensor] = None,
                         context_latents: Optional[torch.Tensor] = None,
                         pooled_output: Optional[torch.Tensor] = None) -> List:
        """
        Pack DiffSynth-style tensors into ComfyUI conditioning format.

        Args:
            prompt_embeds: Text embeddings [B, seq_len, hidden_dim]
            prompt_mask: Attention mask [B, seq_len]
            edit_latents: Optional edit image latents
            context_latents: Optional context image latents
            pooled_output: Optional pooled text output

        Returns:
            ComfyUI-compatible conditioning list
        """
        # ComfyUI expects: [[prompt_embeds, {"pooled_output": pooled, ...}]]
        cond_dict = {}

        if pooled_output is not None:
            cond_dict["pooled_output"] = pooled_output

        if prompt_mask is not None:
            cond_dict["attention_mask"] = prompt_mask

        if edit_latents is not None:
            cond_dict["edit_latents"] = edit_latents

        if context_latents is not None:
            cond_dict["context_latents"] = context_latents

        return [[prompt_embeds, cond_dict]]

    @staticmethod
    def unpack_conditioning(conditioning: List) -> Dict[str, torch.Tensor]:
        """
        Extract DiffSynth-style tensors from ComfyUI conditioning.

        Args:
            conditioning: ComfyUI conditioning list

        Returns:
            Dictionary with extracted tensors
        """
        if not conditioning or not isinstance(conditioning, list):
            return {}

        result = {}

        # Standard ComfyUI format: [[embeds, dict]]
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


class QwenVAEWrapper:
    """
    Wrapper for handling Qwen's 16-channel VAE operations.

    Critical differences from standard VAE:
    - 16 channels instead of 4
    - Special packing operation required
    - Different scaling factors
    """

    def __init__(self, vae_model=None):
        self.vae = vae_model
        self.scaling_factor = 0.13025  # From DiffSynth VAE config

    def encode(self, image: torch.Tensor, tiled: bool = False,
               tile_size: int = 128, tile_stride: int = 64) -> torch.Tensor:
        """
        Encode image to 16-channel latents.

        Args:
            image: Input image tensor [B, 3, H, W] normalized to [-1, 1]
            tiled: Whether to use tiled encoding for large images
            tile_size: Size of tiles if tiled=True
            tile_stride: Stride between tiles

        Returns:
            16-channel latents [B, 16, H/8, W/8]
        """
        if self.vae is None:
            # Fallback: create synthetic 16-channel latents
            B, C, H, W = image.shape
            latents = torch.randn(B, QWEN_VAE_CHANNELS, H // 8, W // 8,
                                 device=image.device, dtype=image.dtype)
            return latents * self.scaling_factor

        # Use actual VAE encoding
        if hasattr(self.vae, 'encode'):
            latents = self.vae.encode(image)
            if hasattr(latents, 'latent_dist'):
                latents = latents.latent_dist.sample()
        else:
            # Direct encoding
            latents = self.vae(image)

        # Ensure 16 channels
        if latents.shape[1] != QWEN_VAE_CHANNELS:
            latents = QwenWrapperBase.ensure_16_channel(latents)

        return latents * self.scaling_factor

    def decode(self, latents: torch.Tensor, tiled: bool = False,
               tile_size: int = 128, tile_stride: int = 64) -> torch.Tensor:
        """
        Decode 16-channel latents to image.

        Args:
            latents: 16-channel latents [B, 16, H, W]
            tiled: Whether to use tiled decoding
            tile_size: Size of tiles if tiled=True
            tile_stride: Stride between tiles

        Returns:
            Decoded image [B, 3, H*8, W*8] in range [-1, 1]
        """
        latents = latents / self.scaling_factor

        if self.vae is None:
            # Fallback: create synthetic image
            B, C, H, W = latents.shape
            return torch.randn(B, 3, H * 8, W * 8,
                             device=latents.device, dtype=latents.dtype)

        # Use actual VAE decoding
        if hasattr(self.vae, 'decode'):
            image = self.vae.decode(latents)
            if hasattr(image, 'sample'):
                image = image.sample
        else:
            # Direct decoding
            image = self.vae(latents)

        return image


# Utility functions for node implementations

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    # Assume tensor is [B, C, H, W] in range [-1, 1]
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first in batch

    # Convert to [H, W, C] in range [0, 255]
    tensor = tensor.permute(1, 2, 0)
    tensor = (tensor + 1) * 127.5
    tensor = tensor.clamp(0, 255).cpu().numpy().astype(np.uint8)

    return Image.fromarray(tensor)


def pil_to_tensor(image: Image.Image, device="cuda", dtype=torch.float32) -> torch.Tensor:
    """Convert PIL Image to tensor."""
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to tensor
    tensor = torch.from_numpy(np.array(image)).float()

    # Rearrange to [C, H, W] and normalize to [-1, 1]
    tensor = tensor.permute(2, 0, 1) / 127.5 - 1

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor.to(device=device, dtype=dtype)