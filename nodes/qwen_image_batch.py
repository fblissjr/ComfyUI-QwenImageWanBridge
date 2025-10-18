"""
QwenImageBatch - Aspect-ratio preserving image batching for Qwen
Solves two issues with standard batch nodes:
1. Skips None/empty inputs (no black images)
2. Preserves aspect ratios (no cropping)
3. Applies v2.6.1 scaling modes for consistent dimensions
"""

import torch
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

try:
    import comfy.utils
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    logger.warning("ComfyUI utilities not available")


class QwenImageBatch:
    """
    Batch multiple images while preserving aspect ratios.

    Unlike standard batch nodes:
    - Skips empty inputs (no black images added)
    - Preserves original aspect ratios (no cropping)
    - Applies consistent scaling to ensure compatible dimensions
    - Returns batch ready for QwenVLTextEncoder
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE", {
                    "tooltip": "First image (required)"
                }),
            },
            "optional": {
                "image_2": ("IMAGE", {
                    "tooltip": "Second image (optional, auto-detected)"
                }),
                "image_3": ("IMAGE", {
                    "tooltip": "Third image (optional)"
                }),
                "image_4": ("IMAGE", {
                    "tooltip": "Fourth image (optional - may cause VRAM issues)"
                }),
                "image_5": ("IMAGE", {
                    "tooltip": "Fifth image (optional - may cause VRAM issues)"
                }),
                "image_6": ("IMAGE", {
                    "tooltip": "Sixth image (optional)"
                }),
                "image_7": ("IMAGE", {
                    "tooltip": "Seventh image (optional)"
                }),
                "image_8": ("IMAGE", {
                    "tooltip": "Eighth image (optional)"
                }),
                "image_9": ("IMAGE", {
                    "tooltip": "Ninth image (optional)"
                }),
                "image_10": ("IMAGE", {
                    "tooltip": "Tenth image (optional)"
                }),
                "scaling_mode": ([
                    "preserve_resolution",
                    "max_dimension_1024",
                    "area_1024",
                    "no_scaling"
                ], {
                    "default": "preserve_resolution",
                    "tooltip": (
                        "Resolution scaling mode (base calculation):\n\n"
                        "preserve_resolution (recommended):\n"
                        "  - Keeps original size with 32px alignment\n"
                        "  - No zoom-out, best quality\n\n"
                        "max_dimension_1024 (for 4K/large images):\n"
                        "  - Scales largest side to 1024px\n"
                        "  - Reduces VRAM usage\n\n"
                        "area_1024 (legacy):\n"
                        "  - Scales to ~1024x1024 area\n\n"
                        "no_scaling:\n"
                        "  - Pass images through as-is\n\n"
                        "Note: See batch_strategy for multi-image handling"
                    )
                }),
                "batch_strategy": ([
                    "max_dimensions",
                    "first_image"
                ], {
                    "default": "max_dimensions",
                    "tooltip": (
                        "How to handle different aspect ratios:\n\n"
                        "max_dimensions (recommended):\n"
                        "  - Calculates ideal size per image\n"
                        "  - Scales all to maximum width/height\n"
                        "  - Minimal aspect distortion\n"
                        "  - Best for mixed aspect ratios\n\n"
                        "first_image:\n"
                        "  - Uses first image dimensions as target\n"
                        "  - All images scaled to match first\n"
                        "  - May distort aspect ratios more\n"
                        "  - Good when hero image should dominate"
                    )
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show batching details in console"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "count", "info")
    FUNCTION = "batch_images"
    CATEGORY = "QwenImage/Utilities"
    TITLE = "Qwen Image Batch"
    DESCRIPTION = "Batch images (auto-detects up to 10), preserving aspect ratios with v2.6.1 scaling. No inputcount needed!"

    def calculate_dimensions(self, w: int, h: int, mode: str) -> Tuple[int, int]:
        """
        Calculate target dimensions based on scaling mode.
        Matches v2.6.1 scaling logic from qwen_vl_encoder.py
        """
        import math

        if mode == "no_scaling":
            # Just ensure 32px alignment
            return (round(w / 32) * 32, round(h / 32) * 32)

        aspect_ratio = w / h

        if mode == "preserve_resolution":
            # Preserve original dimensions with 32px alignment
            width = round(w / 32) * 32
            height = round(h / 32) * 32

        elif mode == "max_dimension_1024":
            # Scale so largest dimension equals 1024px
            scale = 1024 / max(w, h)
            width = round(w * scale / 32) * 32
            height = round(h * scale / 32) * 32

        else:  # area_1024
            # Area-based scaling to ~1024x1024
            target_area = 1024 * 1024
            width = math.sqrt(target_area * aspect_ratio)
            height = width / aspect_ratio
            width = round(width / 32) * 32
            height = round(height / 32) * 32

        return (max(32, int(width)), max(32, int(height)))

    def batch_images(self, image_1, scaling_mode: str = "preserve_resolution",
                    batch_strategy: str = "max_dimensions",
                    debug_mode: bool = False, **kwargs) -> Tuple[torch.Tensor, int, str]:
        """
        Batch images with aspect ratio preservation and scaling.
        Auto-detects connected images (up to 10).
        """
        if not COMFY_AVAILABLE:
            raise RuntimeError("ComfyUI not available")

        # Collect non-None images (auto-detect up to 10)
        images = [image_1]  # First image is required
        image_info = []

        h, w = image_1.shape[1], image_1.shape[2]
        image_info.append(f"Image 1: {w}x{h}")
        if debug_mode:
            logger.info(f"[QwenImageBatch] Image 1: {w}x{h} ({image_1.shape})")

        # Auto-detect remaining images (2-10)
        for i in range(2, 11):
            img = kwargs.get(f"image_{i}", None)
            if img is not None:
                images.append(img)
                h, w = img.shape[1], img.shape[2]
                image_info.append(f"Image {i}: {w}x{h}")

                if debug_mode:
                    logger.info(f"[QwenImageBatch] Image {i}: {w}x{h} ({img.shape})")

        if len(images) == 0:
            raise ValueError("At least one image must be provided")

        # First pass: Calculate target dimensions for each image
        target_dimensions = []
        for img in images:
            h, w = img.shape[1], img.shape[2]
            img_target_w, img_target_h = self.calculate_dimensions(w, h, scaling_mode)
            target_dimensions.append((img_target_w, img_target_h))

        # Determine final target dimensions based on batch_strategy
        if batch_strategy == "first_image":
            # Use first image's target dimensions for all
            final_w, final_h = target_dimensions[0]
            strategy_info = f"first_image (all scaled to {final_w}x{final_h})"
        else:  # max_dimensions (default)
            # Find maximum dimensions across all images
            final_w = max(dim[0] for dim in target_dimensions)
            final_h = max(dim[1] for dim in target_dimensions)
            strategy_info = f"max_dimensions (all scaled to {final_w}x{final_h})"

        if debug_mode:
            logger.info(f"[QwenImageBatch] Batch strategy: {strategy_info} (scaling_mode: {scaling_mode})")

        # Second pass: Scale images to unified dimensions
        scaled_images = []
        for i, img in enumerate(images):
            h, w = img.shape[1], img.shape[2]
            orig_target_w, orig_target_h = target_dimensions[i]
            orig_aspect = w / h
            final_aspect = final_w / final_h

            # Convert to CHW for upscale
            img_chw = img.movedim(-1, 1)  # HWC to CHW

            # All images scaled to final dimensions (first_image or max_dimensions)
            target_w, target_h = final_w, final_h

            scaled = comfy.utils.common_upscale(
                img_chw,
                target_w,
                target_h,
                "bicubic",
                "disabled"
            )

            # Convert back to HWC
            scaled_hwc = scaled.movedim(1, -1)  # CHW to HWC
            scaled_images.append(scaled_hwc)

            if debug_mode:
                scale_factor = target_w / w
                aspect_diff = abs(orig_aspect - final_aspect)

                # Determine aspect change description
                if aspect_diff < 0.01:  # Less than 1% difference
                    aspect_status = "preserved"
                elif (orig_target_w, orig_target_h) == (target_w, target_h):
                    aspect_status = "preserved"
                else:
                    aspect_status = f"adjusted (AR: {orig_aspect:.2f} → {final_aspect:.2f}, diff: {aspect_diff:.2f})"

                logger.info(
                    f"[QwenImageBatch]   Image {i+1}: {w}x{h} -> {target_w}x{target_h} "
                    f"({scale_factor:.2f}x, aspect {aspect_status})"
                )

        # Concatenate along batch dimension
        batched = torch.cat(scaled_images, dim=0)

        # Mark images as pre-scaled so encoder knows to skip scaling
        # This prevents double-scaling when batch node → encoder
        batched.qwen_pre_scaled = True
        batched.qwen_scaling_mode = scaling_mode
        batched.qwen_batch_strategy = batch_strategy

        # Build info string
        info_lines = [
            f"Batched {len(images)} images",
            f"Scaling mode: {scaling_mode}",
            f"Batch strategy: {batch_strategy}",
            ""
        ] + image_info + [
            "",
            f"Output shape: {batched.shape}",
            f"Strategy: {strategy_info}"
        ]

        if len(images) >= 4:
            info_lines.append("\nWARNING: 4+ images may cause VRAM issues")

        # Warn about aspect ratio adjustments
        if len(set(target_dimensions)) > 1:
            unique_aspects = len(set(w/h for w, h in target_dimensions))
            if unique_aspects > 1:
                info_lines.append(f"\nNOTE: {unique_aspects} different aspect ratios detected - images adjusted for batching")

        info_str = "\n".join(info_lines)

        if debug_mode:
            logger.info(f"[QwenImageBatch] Final batch shape: {batched.shape}")
            logger.info(f"[QwenImageBatch] Info:\n{info_str}")

        return (batched, len(images), info_str)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenImageBatch": QwenImageBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageBatch": "Qwen Image Batch",
}
