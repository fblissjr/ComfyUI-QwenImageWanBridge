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
                "vae_max_dimension": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 3584,
                    "step": 64,
                    "tooltip": (
                        "VAE encoder max dimension (pixel-level detail).\n\n"
                        "Recommended values:\n"
                        "  • 1024 - Safe for 8GB VRAM\n"
                        "  • 2048 - Recommended (12GB+ VRAM)\n"
                        "  • 3584 - Model maximum (24GB+ VRAM)\n\n"
                        "Applied to EACH image before batching.\n"
                        "Always preserves aspect ratio with 32px alignment."
                    )
                }),
                "batch_alignment": ([
                    "match_smallest",
                    "match_first",
                    "match_largest"
                ], {
                    "default": "match_smallest",
                    "tooltip": (
                        "How to align multiple images with different sizes:\n\n"
                        "match_smallest (VRAM Safe - Recommended):\n"
                        "  • All images scaled DOWN to smallest\n"
                        "  • Example: 2048×2048 + 1024×1024 → both 1024×1024\n"
                        "  ✓ Lowest VRAM usage\n"
                        "  ✓ No quality loss on small images\n"
                        "  ⚠ May lose detail from large images\n\n"
                        "match_first (Predictable):\n"
                        "  • All images match first image size\n"
                        "  • Example: First=1536×2048, others scaled to match\n"
                        "  ✓ Consistent output size\n"
                        "  ⚠ May upscale or downscale other images\n\n"
                        "match_largest (Quality - High VRAM):\n"
                        "  • All images scaled UP to largest\n"
                        "  • Example: 1024×1024 + 2048×2048 → both 2048×2048\n"
                        "  ⚠ WARNING: Can cause out-of-memory errors!\n"
                        "  ⚠ Upscaling small images reduces quality\n"
                        "  ✓ Preserves max detail from largest image"
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

    def calculate_vae_dimensions(self, w: int, h: int, max_dimension: int) -> Tuple[int, int]:
        """
        Calculate VAE dimensions with 32px alignment.

        Args:
            w: Original width
            h: Original height
            max_dimension: Maximum dimension size (0 = unlimited)

        Returns:
            (width, height) with 32px alignment
        """
        # Step 1: Cap to max_dimension if needed
        if max_dimension > 0 and max(w, h) > max_dimension:
            scale = max_dimension / max(w, h)
            w = int(w * scale)
            h = int(h * scale)

        # Step 2: Apply 32px alignment (VAE requirement)
        w = round(w / 32) * 32
        h = round(h / 32) * 32

        return (max(32, int(w)), max(32, int(h)))

    def calculate_vision_dimensions(self, w: int, h: int) -> Tuple[int, int]:
        """
        Calculate vision encoder dimensions using area-based scaling.

        Vision encoder trained at 384×384. Scale to that target area while
        preserving aspect ratio, then align to 28px.

        Args:
            w: Original width
            h: Original height

        Returns:
            (width, height) scaled to ~384×384 area with 28px alignment
        """
        target_area = 384 * 384  # Model's trained resolution
        aspect_ratio = w / h

        # Calculate dimensions from target area and aspect ratio
        vision_w = int((target_area * aspect_ratio) ** 0.5)
        vision_h = int(vision_w / aspect_ratio)

        # Apply 28px alignment
        vision_w = round(vision_w / 28) * 28
        vision_h = round(vision_h / 28) * 28

        return (max(28, vision_w), max(28, vision_h))

    def batch_images(self, image_1,
                    vae_max_dimension: int = 2048,
                    batch_alignment: str = "match_smallest",
                    debug_mode: bool = False, **kwargs) -> Tuple[torch.Tensor, int, str]:
        """
        Batch images with aspect ratio preservation and separate VAE/vision sizing.
        Auto-detects connected images (up to 10).

        Vision encoder hardcoded to 384px (model's trained resolution).
        Higher values cause object duplication and scaling artifacts.

        Args:
            vae_max_dimension: Maximum dimension for VAE encoder (default: 2048)
            batch_alignment: How to align different-sized images (default: match_smallest)
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

        # First pass: Calculate VAE and vision dimensions for each image
        vae_dimensions = []
        vision_dimensions = []
        for img in images:
            h, w = img.shape[1], img.shape[2]
            vae_w, vae_h = self.calculate_vae_dimensions(w, h, vae_max_dimension)
            vision_w, vision_h = self.calculate_vision_dimensions(w, h)
            vae_dimensions.append((vae_w, vae_h))
            vision_dimensions.append((vision_w, vision_h))

        # Determine final target dimensions based on batch_alignment
        if batch_alignment == "match_first":
            # Use first image's dimensions for all
            final_vae_w, final_vae_h = vae_dimensions[0]
            final_vision_w, final_vision_h = vision_dimensions[0]
            strategy_info = f"match_first (VAE: {final_vae_w}x{final_vae_h}, Vision: {final_vision_w}x{final_vision_h})"
        elif batch_alignment == "match_largest":
            # Take maximum of EACH dimension separately to preserve aspect ranges
            final_vae_w = max(w for w, h in vae_dimensions)
            final_vae_h = max(h for w, h in vae_dimensions)
            final_vision_w = max(w for w, h in vision_dimensions)
            final_vision_h = max(h for w, h in vision_dimensions)
            strategy_info = f"match_largest (VAE: {final_vae_w}x{final_vae_h}, Vision: {final_vision_w}x{final_vision_h})"
            if debug_mode:
                logger.info(f"[QwenImageBatch] WARNING: match_largest may cause OOM on large images!")
        else:  # match_smallest (default)
            # Find smallest dimensions (by total pixels)
            smallest_vae_idx = min(range(len(vae_dimensions)), key=lambda i: vae_dimensions[i][0] * vae_dimensions[i][1])
            smallest_vision_idx = min(range(len(vision_dimensions)), key=lambda i: vision_dimensions[i][0] * vision_dimensions[i][1])
            final_vae_w, final_vae_h = vae_dimensions[smallest_vae_idx]
            final_vision_w, final_vision_h = vision_dimensions[smallest_vision_idx]
            strategy_info = f"match_smallest (VAE: {final_vae_w}x{final_vae_h}, Vision: {final_vision_w}x{final_vision_h})"

        # Recalculate vision dimensions from final VAE dimensions to ensure matching aspect ratio
        # This prevents aspect ratio mismatches when batch_alignment creates "chimera" dimensions
        final_vision_w, final_vision_h = self.calculate_vision_dimensions(
            final_vae_w, final_vae_h
        )

        if debug_mode:
            logger.info(f"[QwenImageBatch] Batch alignment: {strategy_info}")
            logger.info(f"[QwenImageBatch] Recalculated vision dims from VAE aspect: {final_vision_w}x{final_vision_h}")

        # Second pass: Scale images to unified VAE dimensions
        # (Note: Vision dimensions stored in metadata, encoder will handle vision resize)
        scaled_images = []
        for i, img in enumerate(images):
            h, w = img.shape[1], img.shape[2]
            orig_vae_w, orig_vae_h = vae_dimensions[i]
            orig_aspect = w / h
            final_aspect = final_vae_w / final_vae_h

            # Convert to CHW for upscale
            img_chw = img.movedim(-1, 1)  # HWC to CHW

            # Scale to final VAE dimensions
            scaled = comfy.utils.common_upscale(
                img_chw,
                final_vae_w,
                final_vae_h,
                "bicubic",
                "disabled"
            )

            # Convert back to HWC
            scaled_hwc = scaled.movedim(1, -1)  # CHW to HWC
            scaled_images.append(scaled_hwc)

            if debug_mode:
                scale_factor = final_vae_w / w
                aspect_diff = abs(orig_aspect - final_aspect)

                # Determine aspect change description
                if aspect_diff < 0.01:  # Less than 1% difference
                    aspect_status = "preserved"
                elif (orig_vae_w, orig_vae_h) == (final_vae_w, final_vae_h):
                    aspect_status = "preserved"
                else:
                    aspect_status = f"adjusted (AR: {orig_aspect:.2f} → {final_aspect:.2f}, diff: {aspect_diff:.2f})"

                logger.info(
                    f"[QwenImageBatch]   Image {i+1}: {w}x{h} -> VAE: {final_vae_w}x{final_vae_h}, "
                    f"Vision: {final_vision_w}x{final_vision_h} ({scale_factor:.2f}x, aspect {aspect_status})"
                )

        # Concatenate along batch dimension
        batched = torch.cat(scaled_images, dim=0)

        # Attach metadata so encoder knows to skip scaling
        # This prevents double-scaling when batch node → encoder
        batched.qwen_pre_scaled = True
        batched.qwen_vae_dimensions = (final_vae_w, final_vae_h)
        batched.qwen_vision_dimensions = (final_vision_w, final_vision_h)
        batched.qwen_batch_alignment = batch_alignment

        # Build info string
        info_lines = [
            f"Batched {len(images)} images",
            f"VAE max: {vae_max_dimension}px, Vision: 384×384 area (hardcoded)",
            f"Batch alignment: {batch_alignment}",
            ""
        ] + image_info + [
            "",
            f"Output shape: {batched.shape}",
            f"VAE dimensions: {final_vae_w}x{final_vae_h}",
            f"Vision dimensions: {final_vision_w}x{final_vision_h}"
        ]

        if len(images) >= 4:
            info_lines.append("\nWARNING: 4+ images may cause VRAM issues")

        if batch_alignment == "match_largest":
            info_lines.append("\nWARNING: match_largest can cause OOM on large images!")

        # Warn about aspect ratio adjustments
        if len(set(vae_dimensions)) > 1:
            unique_aspects = len(set(w/h for w, h in vae_dimensions))
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
