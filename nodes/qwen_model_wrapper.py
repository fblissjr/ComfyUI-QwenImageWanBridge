"""
Wrapper for Qwen Image model to handle dimension mismatches gracefully
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def ensure_even_dimensions(latent: torch.Tensor) -> torch.Tensor:
    """Ensure latent dimensions are even (divisible by 2) for patch processing."""
    shape = latent.shape

    # Handle both 4D and 5D tensors
    if len(shape) == 5:  # Wan21 format [B, C, T, H, W]
        b, c, t, h, w = shape
        new_h = h if h % 2 == 0 else h + 1
        new_w = w if w % 2 == 0 else w + 1

        if new_h != h or new_w != w:
            # Pad to nearest even dimension
            pad_h = new_h - h
            pad_w = new_w - w
            latent = F.pad(latent, (0, pad_w, 0, pad_h, 0, 0, 0, 0, 0, 0), mode='constant', value=0)
            logger.info(f"Padded latent from [{b}, {c}, {t}, {h}, {w}] to [{b}, {c}, {t}, {new_h}, {new_w}]")

    elif len(shape) == 4:  # Standard format [B, C, H, W]
        b, c, h, w = shape
        new_h = h if h % 2 == 0 else h + 1
        new_w = w if w % 2 == 0 else w + 1

        if new_h != h or new_w != w:
            # Pad to nearest even dimension
            pad_h = new_h - h
            pad_w = new_w - w
            latent = F.pad(latent, (0, pad_w, 0, pad_h, 0, 0, 0, 0), mode='constant', value=0)
            logger.info(f"Padded latent from [{b}, {c}, {h}, {w}] to [{b}, {c}, {new_h}, {new_w}]")

    return latent


def match_latent_dimensions(ref_latents: List[torch.Tensor], target_shape: Tuple[int, ...],
                           debug: bool = False) -> List[torch.Tensor]:
    """
    Resize reference latents to match target dimensions.

    Args:
        ref_latents: List of reference latent tensors
        target_shape: Target shape to match (from main latent)
        debug: Enable debug logging

    Returns:
        List of resized reference latents
    """
    if not ref_latents:
        return ref_latents

    # Determine target H and W from target shape
    if len(target_shape) == 5:  # [B, C, T, H, W]
        target_h, target_w = target_shape[3], target_shape[4]
    elif len(target_shape) == 4:  # [B, C, H, W]
        target_h, target_w = target_shape[2], target_shape[3]
    else:
        logger.warning(f"Unexpected target shape: {target_shape}")
        return ref_latents

    # Ensure target dimensions are even
    target_h = target_h if target_h % 2 == 0 else target_h + 1
    target_w = target_w if target_w % 2 == 0 else target_w + 1

    resized_latents = []

    for i, latent in enumerate(ref_latents):
        original_shape = latent.shape

        # Get current dimensions
        if len(original_shape) == 5:  # Wan21 format
            b, c, t, h, w = original_shape
            needs_resize = (h != target_h or w != target_w)

            if needs_resize:
                # Resize the spatial dimensions
                # First, reshape to combine batch and time for interpolation
                latent_reshaped = latent.view(b * t, c, h, w)

                # Interpolate to target size
                latent_resized = F.interpolate(
                    latent_reshaped,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )

                # Reshape back to 5D
                latent_resized = latent_resized.view(b, c, t, target_h, target_w)

                if debug:
                    logger.info(f"Resized ref latent {i+1} from {original_shape} to {latent_resized.shape}")

                resized_latents.append(latent_resized)
            else:
                # Already correct size, just ensure even dimensions
                resized_latents.append(ensure_even_dimensions(latent))

        elif len(original_shape) == 4:  # Standard format
            b, c, h, w = original_shape
            needs_resize = (h != target_h or w != target_w)

            if needs_resize:
                # Interpolate to target size
                latent_resized = F.interpolate(
                    latent,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )

                if debug:
                    logger.info(f"Resized ref latent {i+1} from {original_shape} to {latent_resized.shape}")

                resized_latents.append(latent_resized)
            else:
                # Already correct size, just ensure even dimensions
                resized_latents.append(ensure_even_dimensions(latent))
        else:
            logger.warning(f"Unexpected latent shape: {original_shape}")
            resized_latents.append(latent)

    return resized_latents


def wrap_reference_latents(conditioning, debug_mode: bool = False):
    """
    Wrap reference latents in conditioning to handle dimension mismatches.

    This function modifies the conditioning in-place to ensure all reference
    latents can be processed by the model regardless of dimension mismatches.
    """
    if not conditioning or not isinstance(conditioning, list):
        return conditioning

    # Look for reference_latents in the conditioning
    for cond_item in conditioning:
        if isinstance(cond_item, list) and len(cond_item) >= 2:
            cond_dict = cond_item[1]

            if isinstance(cond_dict, dict) and "reference_latents" in cond_dict:
                ref_latents = cond_dict["reference_latents"]

                if ref_latents and isinstance(ref_latents, list):
                    # Ensure all latents have even dimensions
                    processed_latents = []
                    for i, latent in enumerate(ref_latents):
                        processed = ensure_even_dimensions(latent)
                        if debug_mode and not torch.equal(processed, latent):
                            logger.info(f"Adjusted reference latent {i+1} to have even dimensions")
                        processed_latents.append(processed)

                    cond_dict["reference_latents"] = processed_latents

                    if debug_mode:
                        shapes = [str(lat.shape) for lat in processed_latents]
                        logger.info(f"Processed {len(processed_latents)} reference latents: {shapes}")

    return conditioning


class QwenModelPatcher:
    """
    Patches for Qwen model to handle dimension mismatches gracefully.
    """

    @staticmethod
    def patch_extra_conds():
        """
        Patch the extra_conds method to handle dimension mismatches.
        """
        try:
            import comfy.model_base

            # Store original method
            if not hasattr(comfy.model_base.QwenImage, '_original_extra_conds'):
                comfy.model_base.QwenImage._original_extra_conds = comfy.model_base.QwenImage.extra_conds

            def patched_extra_conds(self, **kwargs):
                # Call original method
                out = self._original_extra_conds(**kwargs)

                # Check if we have ref_latents that need adjustment
                if 'ref_latents' in out:
                    ref_latents_cond = out['ref_latents']
                    # The ref_latents are already wrapped in CONDList by the original method
                    # We need to access the actual latents inside
                    if hasattr(ref_latents_cond, 'cond_or_uncond'):
                        # This is a CONDList, extract the latents
                        latents = ref_latents_cond.cond_or_uncond
                        if latents and isinstance(latents, list):
                            # Ensure even dimensions for all latents
                            processed = [ensure_even_dimensions(lat) for lat in latents]
                            ref_latents_cond.cond_or_uncond = processed

                return out

            # Apply patch
            comfy.model_base.QwenImage.extra_conds = patched_extra_conds
            logger.info("Applied QwenImage dimension handling patch")
            return True

        except Exception as e:
            logger.error(f"Failed to apply QwenImage patch: {e}")
            return False

    @staticmethod
    def remove_patches():
        """
        Remove applied patches and restore original methods.
        """
        try:
            import comfy.model_base

            if hasattr(comfy.model_base.QwenImage, '_original_extra_conds'):
                comfy.model_base.QwenImage.extra_conds = comfy.model_base.QwenImage._original_extra_conds
                delattr(comfy.model_base.QwenImage, '_original_extra_conds')
                logger.info("Removed QwenImage dimension handling patch")

        except Exception as e:
            logger.error(f"Failed to remove patches: {e}")


# Auto-apply patches on import
def initialize_patches():
    """Initialize dimension handling patches."""
    QwenModelPatcher.patch_extra_conds()

# Apply patches when module is imported
initialize_patches()