"""
Qwen VL Encoder Advanced - Power user features for weighted images and variable resolution
Maintains backward compatibility by being a separate node
"""

import torch
import logging
import math
from typing import Optional, List, Tuple, Any, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try imports
try:
    import comfy
    import folder_paths
    from comfy import model_management
    from nodes import MAX_RESOLUTION
    import comfy.utils
    import node_helpers
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    logger.warning("ComfyUI modules not available")

from .qwen_vl_encoder import QwenVLTextEncoder
from .qwen_model_wrapper import wrap_reference_latents, ensure_even_dimensions


class QwenVLTextEncoderAdvanced(QwenVLTextEncoder):
    """
    Advanced encoder with power user features:
    - Per-image resolution control
    - Weighted image importance
    - Memory optimization modes
    - Custom resolution targets
    """

    @classmethod
    def INPUT_TYPES(cls):
        base_types = super().INPUT_TYPES()

        # Add advanced options to optional section
        base_types["optional"].update({
            "scaling_mode": (["preserve_resolution", "max_dimension_1024", "area_1024"], {
                "default": "preserve_resolution",
                "tooltip": "Resolution scaling mode:\n\npreserve_resolution (recommended):\n  ✓ No zoom-out, subjects stay full-size\n  ✓ Best quality, minimal cropping (32px alignment)\n  ✗ May use more VRAM with very large images\n  Example: 1477×2056 → 1472×2048 (1.00x)\n\nmax_dimension_1024 (for 4K/large images):\n  ✓ Reduces VRAM usage on large images\n  ✓ Balanced quality vs performance\n  ✗ Some zoom-out on large images\n  Example: 3840×2160 → 1024×576 (0.27x)\n\narea_1024 (legacy):\n  ✓ Consistent ~1MP output size\n  ✗ Aggressive zoom-out on large images\n  ✗ Upscales small images unnecessarily\n  Example: 1477×2056 → 864×1216 (0.58x)"
            }),
            "resolution_mode": (["balanced", "hero_first", "hero_last", "progressive", "custom", "memory_optimized"], {
                "default": "balanced",
                "tooltip": "Resolution strategy for multi-image processing (applies weight to scaling_mode base resolution)"
            }),
            "vision_target_area": ("INT", {
                "default": 147456,  # 384*384
                "min": 65536,       # 256*256
                "max": 1048576,     # 1024*1024
                "step": 4096,
                "tooltip": "Target pixel area for vision encoder (default: 384²=147456)"
            }),
            "vae_target_area": ("INT", {
                "default": 1048576,  # 1024*1024
                "min": 262144,       # 512*512
                "max": 4194304,      # 2048*2048
                "step": 16384,
                "tooltip": "Target pixel area for VAE encoder (default: 1024²=1048576)"
            }),
            "hero_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.5,
                "max": 2.0,
                "step": 0.1,
                "tooltip": "Resolution multiplier for hero image (hero modes only)"
            }),
            "reference_weight": ("FLOAT", {
                "default": 0.5,
                "min": 0.1,
                "max": 1.0,
                "step": 0.1,
                "tooltip": "Resolution multiplier for reference images"
            }),
            "max_memory_mb": ("INT", {
                "default": 0,
                "min": 0,
                "max": 16384,
                "step": 512,
                "tooltip": "Max memory for images in MB (0=unlimited)"
            }),
            "image_weights": ("STRING", {
                "default": "",
                "tooltip": "Comma-separated weights per image (e.g., '1.0,0.5,0.3')"
            }),
            "label_format": (["Picture", "Image"], {
                "default": "Picture",
                "tooltip": "Label format for multi-image: 'Picture X:' (DiffSynth) or 'Image X:' (Docs)"
            }),
            "auto_label": ("BOOLEAN", {
                "default": True,
                "tooltip": "Automatically add labels for multiple images (e.g., 'Picture 1:')"
            }),
            "verbose_log": ("BOOLEAN", {
                "default": False,
                "tooltip": "Enable verbose console logging of model forward passes"
            }),
        })

        return base_types

    TITLE = "Qwen2.5-VL Text Encoder (Advanced)"
    DESCRIPTION = "Power user encoder with weighted images and variable resolution"

    def calculate_weighted_dimensions(self, target_area: int, aspect_ratio: float, weight: float) -> tuple:
        """Calculate dimensions with weight applied to target area."""
        weighted_area = target_area * weight
        width = math.sqrt(weighted_area * aspect_ratio)
        height = width / aspect_ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return max(32, int(width)), max(32, int(height))

    def get_resolution_weights(self, num_images: int, mode: str, hero_weight: float,
                              reference_weight: float, image_weights_str: str) -> List[float]:
        """Calculate per-image resolution weights based on mode."""

        # Parse custom weights if provided
        if image_weights_str:
            try:
                weights = [float(w.strip()) for w in image_weights_str.split(',')]
                # Pad or truncate to match image count
                if len(weights) < num_images:
                    weights += [reference_weight] * (num_images - len(weights))
                return weights[:num_images]
            except:
                logger.warning(f"Invalid image_weights format, using mode: {mode}")

        # Mode-based weights
        if mode == "balanced":
            return [1.0] * num_images

        elif mode == "hero_first":
            weights = [hero_weight] + [reference_weight] * (num_images - 1)
            return weights

        elif mode == "hero_last":
            weights = [reference_weight] * (num_images - 1) + [hero_weight]
            return weights

        elif mode == "progressive":
            # Gradually decrease weight
            if num_images <= 1:
                return [1.0]
            step = (hero_weight - reference_weight) / (num_images - 1)
            weights = [hero_weight - (i * step) for i in range(num_images)]
            return weights

        elif mode == "memory_optimized":
            # Will be handled separately with memory calculation
            return [1.0] * num_images

        else:  # custom or fallback
            return [1.0] * num_images

    def optimize_for_memory(self, images: List, max_memory_mb: int,
                           base_vision_area: int, base_vae_area: int) -> Tuple[List[float], List[float]]:
        """Calculate weights to fit within memory budget."""

        # Estimate memory per pixel (rough approximation)
        bytes_per_pixel_vision = 4  # float32
        bytes_per_pixel_vae = 16    # latent space

        # Calculate base memory requirement
        total_pixels = len(images) * (base_vision_area + base_vae_area)
        base_memory_mb = (total_pixels * bytes_per_pixel_vision) / (1024 * 1024)

        if max_memory_mb == 0 or base_memory_mb <= max_memory_mb:
            # No optimization needed
            return [1.0] * len(images), [1.0] * len(images)

        # Calculate reduction factor
        reduction = max_memory_mb / base_memory_mb

        # Apply different reduction strategies
        vision_weights = []
        vae_weights = []

        for i in range(len(images)):
            if i == 0:
                # Keep first image at higher quality
                vision_weights.append(min(1.0, reduction * 1.5))
                vae_weights.append(min(1.0, reduction * 1.2))
            else:
                # Reduce reference images more aggressively
                vision_weights.append(reduction * 0.8)
                vae_weights.append(reduction * 0.6)

        return vision_weights, vae_weights

    def encode(self, clip, text: str, mode: str = "text_to_image",
              edit_image: Optional[torch.Tensor] = None,
              vae=None, inpaint_mask: Optional[torch.Tensor] = None,
              system_prompt: str = "", scaling_mode: str = "preserve_resolution",
              debug_mode: bool = False,
              resolution_mode: str = "balanced",
              vision_target_area: int = 147456,  # 384*384
              vae_target_area: int = 1048576,    # 1024*1024
              hero_weight: float = 1.0,
              reference_weight: float = 0.5,
              max_memory_mb: int = 0,
              image_weights: str = "",
              label_format: str = "Picture",
              auto_label: bool = True,
              verbose_log: bool = False) -> Tuple[Any]:
        """
        Advanced encode with per-image resolution control.
        """

        import comfy.utils

        # Control verbose debug output based on verbose_log parameter
        try:
            from . import debug_patch
            debug_patch.set_debug_verbose(verbose_log)
            if verbose_log:
                logger.info("[Advanced Encoder] Verbose console logging enabled - model forward passes will be traced")
            elif debug_mode:
                logger.info("[Advanced Encoder] Debug mode enabled - UI output active, console logging disabled")
        except Exception as e:
            if debug_mode:
                logger.debug(f"Could not control verbose logging: {e}")

        vision_images = []
        ref_latents = []
        debug_info = []

        # Process images if provided
        if mode == "image_edit" and edit_image is not None:
            if debug_mode:
                logger.info(f"[Advanced Encoder] Input shape: {edit_image.shape}")
                logger.info(f"[Advanced Encoder] Resolution mode: {resolution_mode}")
                debug_info.append(f"Input shape: {edit_image.shape}")
                debug_info.append(f"Resolution mode: {resolution_mode}")

            # Convert batch tensor to list of images
            images = [edit_image[i:i+1, :, :, :3] for i in range(edit_image.shape[0])]
            num_images = len(images)

            # Get resolution weights for each image
            weights = self.get_resolution_weights(
                num_images, resolution_mode, hero_weight, reference_weight, image_weights
            )

            # Handle memory optimization
            if resolution_mode == "memory_optimized" and max_memory_mb > 0:
                vision_weights, vae_weights = self.optimize_for_memory(
                    images, max_memory_mb, vision_target_area, vae_target_area
                )
            else:
                vision_weights = weights
                vae_weights = weights

            debug_info.append(f"Processing {num_images} images with weights: {[f'{w:.2f}' for w in vision_weights]}")

            # Process each image with its weight
            for i, img in enumerate(images):
                h, w = img.shape[1], img.shape[2]
                aspect_ratio = w / h

                # Calculate weighted dimensions for vision encoder (always use area mode for vision)
                vision_w, vision_h = self.calculate_weighted_dimensions(
                    vision_target_area, aspect_ratio, vision_weights[i]
                )

                img_chw = img.movedim(-1, 1)
                vision_img = comfy.utils.common_upscale(img_chw, vision_w, vision_h, "bicubic", "disabled")
                vision_images.append(vision_img.movedim(1, -1))

                if debug_mode:
                    logger.info(f"[Advanced Encoder] Image {i+1}: {w}x{h} -> vision: {vision_w}x{vision_h} (weight: {vision_weights[i]:.2f})")

                debug_info.append(f"Image {i+1}: {w}x{h} -> vision: {vision_w}x{vision_h} (weight: {vision_weights[i]:.2f})")

                # Calculate weighted dimensions for VAE encoder if provided - respects scaling_mode
                if vae is not None:
                    # Apply scaling_mode to base dimensions, then apply weight
                    if scaling_mode == "preserve_resolution":
                        # Start with original dimensions, apply weight
                        base_w = round(w / 32) * 32
                        base_h = round(h / 32) * 32
                        # Apply weight by scaling the base dimensions
                        vae_w = round((base_w * vae_weights[i]) / 32) * 32
                        vae_h = round((base_h * vae_weights[i]) / 32) * 32
                        if debug_mode:
                            logger.info(f"[Advanced Encoder]        VAE (preserve+weight): {w}x{h} -> {vae_w}x{vae_h} (weight: {vae_weights[i]:.2f})")
                        debug_info.append(f"    VAE scaling: preserve_resolution with weight {vae_weights[i]:.2f} ({vae_w}x{vae_h})")
                    elif scaling_mode == "max_dimension_1024":
                        # Scale to max dimension 1024, then apply weight
                        scale = 1024 / max(w, h)
                        base_w = round(w * scale / 32) * 32
                        base_h = round(h * scale / 32) * 32
                        vae_w = round((base_w * vae_weights[i]) / 32) * 32
                        vae_h = round((base_h * vae_weights[i]) / 32) * 32
                        if debug_mode:
                            logger.info(f"[Advanced Encoder]        VAE (max_dim+weight): {w}x{h} -> {vae_w}x{vae_h} (weight: {vae_weights[i]:.2f})")
                        debug_info.append(f"    VAE scaling: max_dimension_1024 with weight {vae_weights[i]:.2f} ({vae_w}x{vae_h})")
                    else:  # area_1024
                        # Use weighted dimensions calculation (original behavior)
                        vae_w, vae_h = self.calculate_weighted_dimensions(
                            vae_target_area, aspect_ratio, vae_weights[i]
                        )
                        if debug_mode:
                            logger.info(f"[Advanced Encoder]        VAE (area+weight): {w}x{h} -> {vae_w}x{vae_h} (weight: {vae_weights[i]:.2f})")
                        debug_info.append(f"    VAE scaling: area_1024 with weight {vae_weights[i]:.2f} ({vae_w}x{vae_h})")

                    vae_img = comfy.utils.common_upscale(img_chw, vae_w, vae_h, "bicubic", "disabled")
                    vae_img_hwc = vae_img.movedim(1, -1)
                    ref_latent = vae.encode(vae_img_hwc[:, :, :, :3])

                    # Wan21 latent format expects 5D tensors (batch, channels, time, height, width)
                    # If VAE returns 4D, we need to add time dimension
                    if len(ref_latent.shape) == 4:
                        ref_latent = ref_latent.unsqueeze(2)  # Add time dimension: [B, C, H, W] -> [B, C, 1, H, W]
                        if debug_mode:
                            logger.info(f"[Advanced Encoder] Added time dimension for Wan21 format: {ref_latent.shape}")

                    # Ensure dimensions are even for patch processing
                    ref_latent = ensure_even_dimensions(ref_latent)

                    ref_latents.append(ref_latent)

                    if debug_mode:
                        logger.info(f"[Advanced Encoder]        Latent shape: {ref_latent.shape}")

        # Call parent encode with processed images
        # We'll pass the debug_info we've collected
        parent_result = super().encode(
            clip=clip,
            text=text,
            mode=mode,
            edit_image=None,  # We handle images ourselves
            vae=None,  # We handle VAE ourselves
            system_prompt=system_prompt,
            debug_mode=False,  # We'll build our own debug output
        )

        # Get the conditioning from parent
        if isinstance(parent_result, tuple):
            conditioning = parent_result[0]
        else:
            conditioning = parent_result

        # Initialize variables for debug output
        vision_tokens = ""
        formatted_text = text
        drop_idx = 0

        # Now we need to manually handle the vision tokens and formatting
        # since we bypassed the parent's image processing
        if vision_images:
            # Format with vision tokens
            num_images = len(vision_images)

            if num_images == 1:
                vision_tokens = "<|vision_start|><|image_pad|><|vision_end|>"
            else:
                if auto_label:
                    # Use configurable label format
                    vision_tokens = "".join([
                        f"{label_format} {i+1}: <|vision_start|><|image_pad|><|vision_end|>"
                        for i in range(num_images)
                    ])
                else:
                    # No labels, just raw vision tokens
                    vision_tokens = "".join([
                        "<|vision_start|><|image_pad|><|vision_end|>"
                        for i in range(num_images)
                    ])

            # Format text with processor if available
            if self.processor and system_prompt:
                formatted_text = self.processor.format_template(text, system_prompt, vision_tokens)
                drop_idx = self.processor.get_drop_index(mode)
            else:
                formatted_text = f"{vision_tokens}{text}" if mode != "text_to_image" else text
                drop_idx = 0

            # Tokenize with vision support
            tokens = clip.tokenize(formatted_text, images=vision_images)
            conditioning = clip.encode_from_tokens_scheduled(tokens)

            # Apply token dropping
            if drop_idx > 0:
                for i, cond in enumerate(conditioning):
                    if len(cond) >= 1 and isinstance(cond[0], torch.Tensor):
                        if len(cond[0].shape) >= 2 and cond[0].shape[1] > drop_idx:
                            cond[0] = cond[0][:, drop_idx:, ...]

        # Add reference latents if available
        if ref_latents and COMFY_AVAILABLE:
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": ref_latents},
                append=True
            )

            # Apply dimension wrapper to handle mismatches
            conditioning = wrap_reference_latents(conditioning, debug_mode)

        # Build advanced debug output
        if debug_mode:
            debug_output = "=== QWEN ADVANCED ENCODER DEBUG ===\n\n"

            # Add input information
            debug_output += "=== INPUT ===\n"
            debug_output += f"Mode: {mode}\n"
            debug_output += f"User Prompt: {text}\n"
            debug_output += f"User Prompt Length: {len(text)} characters\n"

            if system_prompt:
                debug_output += f"\n=== SYSTEM PROMPT ===\n"
                debug_output += f"{system_prompt}\n"
                debug_output += f"System Prompt Length: {len(system_prompt)} characters\n"

            # Add image processing info
            if vision_images:
                debug_output += f"\n=== IMAGE PROCESSING ===\n"
                debug_output += "\n".join(debug_info)

                # Add vision token information
                debug_output += f"\n\n=== VISION TOKENS ===\n"
                if auto_label:
                    debug_output += f"Auto-labeling: ENABLED ({label_format} X:)\n"
                else:
                    debug_output += f"Auto-labeling: DISABLED\n"
                debug_output += f"Vision Tokens: {vision_tokens}\n"
                debug_output += f"Vision Token Count: {vision_tokens.count('<|vision_start|>')} images\n"

                # Add formatted text that's actually being encoded
                debug_output += f"\n=== FORMATTED TEXT BEING ENCODED ===\n"
                debug_output += f"{formatted_text}\n"
                debug_output += f"Total Length: {len(formatted_text)} characters\n"

                # Add token dropping info
                if drop_idx > 0:
                    debug_output += f"\n=== TOKEN DROPPING ===\n"
                    debug_output += f"Drop Index: {drop_idx} (first {drop_idx} tokens will be dropped after encoding)\n"
                    debug_output += f"Mode: {mode} -> drop_idx={drop_idx}\n"

            debug_output += f"\n=== ADVANCED SETTINGS ===\n"
            debug_output += f"Scaling Mode: {scaling_mode}\n"
            debug_output += f"Resolution Mode: {resolution_mode}\n"
            debug_output += f"Vision Target: {vision_target_area} ({int(math.sqrt(vision_target_area))}²)\n"
            debug_output += f"VAE Target: {vae_target_area} ({int(math.sqrt(vae_target_area))}²)\n"

            if resolution_mode in ["hero_first", "hero_last"]:
                debug_output += f"Hero Weight: {hero_weight}\n"
                debug_output += f"Reference Weight: {reference_weight}\n"

            if max_memory_mb > 0:
                debug_output += f"Memory Limit: {max_memory_mb}MB\n"

            if image_weights:
                debug_output += f"Custom Weights: {image_weights}\n"

            debug_output += f"Label Format: {label_format} X:\n"
            debug_output += f"Auto Label: {auto_label}\n"

            # Calculate memory usage
            if vision_images:
                total_pixels_vision = sum(vision_images[i].shape[1] * vision_images[i].shape[2]
                                         for i in range(len(vision_images)))
                total_pixels_vae = sum(ref_latents[i].shape[2] * ref_latents[i].shape[3] * 64
                                      for i in range(len(ref_latents))) if ref_latents else 0

                estimated_memory_mb = ((total_pixels_vision * 4) + (total_pixels_vae * 16)) / (1024 * 1024)
                debug_output += f"\n=== MEMORY USAGE ===\n"
                debug_output += f"Estimated: {estimated_memory_mb:.1f}MB\n"
                debug_output += f"Vision Pixels: {total_pixels_vision:,}\n"
                debug_output += f"VAE Pixels: {total_pixels_vae:,}\n"

                # Add reference latent info
                if ref_latents:
                    debug_output += f"\n=== REFERENCE LATENTS ===\n"
                    debug_output += f"Number of latents: {len(ref_latents)}\n"
                    for i, latent in enumerate(ref_latents):
                        debug_output += f"Latent {i+1} shape: {latent.shape}\n"
            else:
                # Text-to-image mode
                debug_output += f"\n=== TEXT-TO-IMAGE MODE ===\n"
                debug_output += f"No images provided - using text-only generation\n"
                debug_output += f"Text: {text}\n"
                if system_prompt:
                    debug_output += f"System Prompt: {system_prompt}\n"
        else:
            debug_output = "Enable debug_mode for detailed output"

        return (conditioning, debug_output)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenVLTextEncoderAdvanced": QwenVLTextEncoderAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLTextEncoderAdvanced": "Qwen2.5-VL Text Encoder (Advanced)",
}
