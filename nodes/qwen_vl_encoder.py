"""
Qwen2.5-VL CLIP Wrapper for ComfyUI
Uses ComfyUI's internal Qwen loader but with DiffSynth-Studio templates
Includes all fixes from DiffSynth-Studio and DiffSynth-Engine
Templates loaded from nodes/templates/*.md files
"""

import os
import torch
import logging
from typing import Optional, Dict, Any, Tuple, Union, List
import folder_paths

logger = logging.getLogger(__name__)

# Import our custom processor for proper token dropping
try:
    from .qwen_processor_v2 import QwenProcessorV2
    PROCESSOR_AVAILABLE = True
except ImportError:
    PROCESSOR_AVAILABLE = False
    logger.warning("QwenProcessorV2 not available - will use simplified approach")

# Validation and smart labeling removed for simplicity

# Try to import ComfyUI's utilities
try:
    import comfy.sd
    import comfy.model_management as mm
    import node_helpers
    COMFY_AVAILABLE = True
except ImportError:
    logger.warning("ComfyUI utilities not available")
    COMFY_AVAILABLE = False

# Apply RoPE position embedding fix from DiffSynth-Studio
def apply_rope_fix():
    """Monkey patch to fix batch processing with different image sizes"""
    try:
        import comfy.ldm.qwen_image.model as qwen_model

        # Check if the model has QwenEmbedRope (it might not exist in native models)
        if not hasattr(qwen_model, 'QwenEmbedRope'):
            logger.debug("QwenEmbedRope not found, skipping RoPE fix (expected for native models)")
            return

        original_expand = qwen_model.QwenEmbedRope._expand_pos_freqs_if_needed

        def fixed_expand_pos_freqs(self, video_fhw, txt_seq_lens):
            # Apply fix from DiffSynth-Studio commit 8fcfa1d
            if isinstance(video_fhw, list):
                # Take max dimensions across batch instead of just first element
                video_fhw = tuple(max([i[j] for i in video_fhw]) for j in range(3))

            # Call original method with fixed video_fhw
            return original_expand(self, video_fhw, txt_seq_lens)

        qwen_model.QwenEmbedRope._expand_pos_freqs_if_needed = fixed_expand_pos_freqs
        logger.info("Applied RoPE position embedding fix for batch processing")

    except Exception as e:
        logger.warning(f"Could not apply RoPE fix: {e}")

# Apply fix on module load
apply_rope_fix()

class QwenVLCLIPLoader:
    """
    Load Qwen2.5-VL using ComfyUI's internal CLIP loader
    This ensures compatibility with the diffusion pipeline
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get models from text_encoders folder
        models = folder_paths.get_filename_list("text_encoders")
        # Filter for Qwen models
        qwen_models = [m for m in models if "qwen" in m.lower()]
        if not qwen_models:
            qwen_models = ["qwen_2.5_vl_7b.safetensors"]

        return {
            "required": {
                "model_name": (qwen_models, {
                    "tooltip": "Qwen2.5-VL model from 'ComfyUI/models/text_encoders'"
                }),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "QwenImage/Loaders"
    TITLE = "Qwen2.5-VL CLIP Loader"
    DESCRIPTION = "Load Qwen2.5-VL as CLIP for ComfyUI compatibility"

    def load_clip(self, model_name: str) -> Tuple[Any]:
        """Load Qwen2.5-VL using ComfyUI's CLIP loader"""

        if not COMFY_AVAILABLE:
            raise RuntimeError("ComfyUI not available")

        # Get full path
        model_path = folder_paths.get_full_path("text_encoders", model_name)
        logger.info(f"Loading Qwen2.5-VL from: {model_path}")

        # Load using ComfyUI's CLIP loader with qwen_image type
        clip = comfy.sd.load_clip(
            ckpt_paths=[model_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.QWEN_IMAGE
        )

        logger.info("Successfully loaded Qwen2.5-VL as CLIP")
        return (clip,)


class QwenVLTextEncoder:
    """
    Text encoder for Qwen2.5-VL - DiffSynth/Diffusers compatible
    Handles single or multiple images with proper 32-pixel alignment
    """

    def __init__(self):
        """Initialize with processor for proper token dropping"""
        self.processor = None
        if PROCESSOR_AVAILABLE:
            try:
                self.processor = QwenProcessorV2()
                logger.info("[QwenVLTextEncoder] Initialized with QwenProcessorV2 for proper token dropping")
            except Exception as e:
                logger.warning(f"[QwenVLTextEncoder] Failed to initialize processor: {e}")
                self.processor = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Your prompt"
                }),
                "mode": (["text_to_image", "image_edit", "multi_image_edit", "inpainting"], {
                    "default": "image_edit",
                    "tooltip": "text_to_image: Generate from scratch | image_edit: Single image modify | multi_image_edit: Multiple reference images (DiffSynth pattern) | inpainting: Mask-based editing. Overridden by template_output if connected."
                }),
            },
            "optional": {
                "template_output": ("QWEN_TEMPLATE", {
                    "tooltip": "Template from Template Builder - overrides text/system_prompt/mode when connected"
                }),
                "edit_image": ("IMAGE", {
                    "tooltip": "Single image or batch. For multiple images, use Image Batch node first."
                }),
                "vae": ("VAE", {
                    "tooltip": "Required for image editing - encodes reference latents"
                }),
                "inpaint_mask": ("MASK", {
                    "tooltip": "Inpainting mask for selective editing (use with inpainting mode)"
                }),
                "system_prompt": ("STRING", {
                    "tooltip": "System prompt (overridden by template_output if connected)",
                    "multiline": True,
                    "default": ""
                }),
                "vae_max_dimension": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 3584,
                    "step": 64,
                    "tooltip": (
                        "⚠️ SINGLE-IMAGE MODE ONLY\n"
                        "Ignored when using ImageBatch node.\n\n"
                        "VAE encoder max dimension.\n"
                        "Use ImageBatch node for multi-image workflows."
                    )
                }),
                "vision_max_dimension": ("INT", {
                    "default": 768,
                    "min": 384,
                    "max": 3584,
                    "step": 384,
                    "tooltip": (
                        "⚠️ SINGLE-IMAGE MODE ONLY\n"
                        "Ignored when using ImageBatch node.\n\n"
                        "Uses 384px multiples (model trained resolution).\n"
                        "Valid: 384, 768, 1152, 1536...\n\n"
                        "Recommended:\n"
                        "  • 384 - Model default\n"
                        "  • 768 - 2x (recommended)\n"
                        "  • 1152+ - Experimental"
                    )
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show processing details in console"
                }),
                "auto_label": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically add 'Picture X:' labels for multiple images (DiffSynth standard)"
                }),
                "verbose_log": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable verbose console logging of model forward passes"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "debug_output")
    FUNCTION = "encode"
    CATEGORY = "QwenImage/Encoding"
    TITLE = "Qwen2.5-VL Text Encoder"
    DESCRIPTION = "DiffSynth/Diffusers-compatible encoder with 32-pixel alignment"

    @staticmethod
    def calculate_vae_dimensions(w: int, h: int, max_dimension: int) -> tuple:
        """Calculate VAE dimensions with 32px alignment.

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

        return max(32, int(w)), max(32, int(h))

    @staticmethod
    def calculate_vision_dimensions(w: int, h: int, max_dimension: int) -> tuple:
        """Calculate vision encoder dimensions as multiples of trained 384px resolution.

        Vision encoder trained at 384×384, so we scale to nearest multiple (384, 768, 1152...)
        while preserving aspect ratio. This keeps position embeddings in clean relationship.

        Args:
            w: Original width
            h: Original height
            max_dimension: Maximum dimension size (0 = unlimited)

        Returns:
            (width, height) as multiples of 384px, preserving aspect ratio
        """
        # Step 1: Cap to max_dimension if needed
        if max_dimension > 0 and max(w, h) > max_dimension:
            scale = max_dimension / max(w, h)
            w = int(w * scale)
            h = int(h * scale)

        # Step 2: Round to nearest 384px multiple, preserving aspect ratio
        w = round(w / 384) * 384
        h = round(h / 384) * 384

        return max(384, int(w)), max(384, int(h))



    def encode(self, clip, text: str, mode: str = "text_to_image",
              template_output: Optional[Dict[str, Any]] = None,
              edit_image: Optional[torch.Tensor] = None,
              vae=None, inpaint_mask: Optional[torch.Tensor] = None,
              system_prompt: str = "",
              vae_max_dimension: int = 2048,
              vision_max_dimension: int = 768,
              debug_mode: bool = False, auto_label: bool = True,
              verbose_log: bool = False) -> Tuple[Any]:

        """Encode text and images for Qwen Image generation.
        Follows DiffSynth/Diffusers implementation with 32-pixel alignment."""

        import math
        import comfy.utils

        # Template output overrides individual params
        if template_output:
            text = template_output.get("prompt", text)
            system_prompt = template_output.get("system_prompt", system_prompt)
            mode = template_output.get("mode", mode)
            if debug_mode:
                logger.info(f"[Encoder] Using template_output: mode={mode}, template={template_output.get('template_name')}")

        vision_images = []
        ref_latents = []
        debug_info = []  # Collect debug information for UI output

        # Control verbose debug output based on verbose_log parameter
        try:
            from . import debug_patch
            debug_patch.set_debug_verbose(verbose_log)
            if verbose_log:
                logger.info("[Encoder] Verbose console logging enabled - model forward passes will be traced")
            elif debug_mode:
                logger.info("[Encoder] Debug mode enabled - UI output active, console logging disabled")
        except Exception as e:
            if debug_mode:
                logger.debug(f"Could not control verbose logging: {e}")

        # Validate inpainting mode requirements
        if mode == "inpainting":
            if edit_image is None:
                raise ValueError("Inpainting mode requires edit_image to be provided")
            if inpaint_mask is None:
                raise ValueError("Inpainting mode requires inpaint_mask to be provided")
            if debug_mode:
                logger.info("[Encoder] Inpainting mode: mask and image validation passed")
                debug_info.append("Mode: inpainting (mask-based editing)")

        # Process images if provided
        if mode in ["image_edit", "multi_image_edit", "inpainting"] and edit_image is not None:
            # Check if images are pre-scaled from QwenImageBatch
            is_pre_scaled = getattr(edit_image, 'qwen_pre_scaled', False)
            if is_pre_scaled:
                batch_vae_dims = getattr(edit_image, 'qwen_vae_dimensions', None)
                batch_vision_dims = getattr(edit_image, 'qwen_vision_dimensions', None)
                batch_alignment = getattr(edit_image, 'qwen_batch_alignment', 'unknown')
                if debug_mode:
                    logger.info(f"[Encoder] Images pre-scaled by QwenImageBatch")
                    logger.info(f"[Encoder]   VAE target: {batch_vae_dims}, Vision target: {batch_vision_dims}")
                    logger.info(f"[Encoder]   Batch alignment: {batch_alignment}")
                debug_info.append(f"Pre-scaled by QwenImageBatch (alignment: {batch_alignment})")
                debug_info.append(f"Target dimensions - VAE: {batch_vae_dims}, Vision: {batch_vision_dims}")

            if debug_mode:
                logger.info(f"[Encoder] Input shape: {edit_image.shape}")
                debug_info.append(f"Input shape: {edit_image.shape}")

            # Convert batch tensor to list of images
            images = [edit_image[i:i+1, :, :, :3] for i in range(edit_image.shape[0])]
            debug_info.append(f"Processing {len(images)} images")

            # Determine target dimensions
            if is_pre_scaled:
                # Trust QwenImageBatch metadata completely
                vision_target_w, vision_target_h = batch_vision_dims
                vae_target_w, vae_target_h = batch_vae_dims if vae is not None else (None, None)

                if debug_mode:
                    logger.info(f"[Encoder] Using ImageBatch targets - Vision: {vision_target_w}x{vision_target_h}, VAE: {vae_target_w}x{vae_target_h}")
                debug_info.append(f"Resize mode: Using ImageBatch targets")
            else:
                # Single-image mode - calculate using encoder parameters
                # Get first image dimensions to calculate targets
                h, w = images[0].shape[1], images[0].shape[2]
                vision_target_w, vision_target_h = self.calculate_vision_dimensions(w, h, vision_max_dimension)
                vae_target_w, vae_target_h = self.calculate_vae_dimensions(w, h, vae_max_dimension) if vae is not None else (None, None)

                if debug_mode:
                    logger.info(f"[Encoder] Single-image mode - Vision max: {vision_max_dimension}px, VAE max: {vae_max_dimension}px")
                    logger.info(f"[Encoder] Calculated targets - Vision: {vision_target_w}x{vision_target_h}, VAE: {vae_target_w}x{vae_target_h}")
                debug_info.append(f"Resize mode: Single-image (Vision max: {vision_max_dimension}px, VAE max: {vae_max_dimension}px)")

            # Process each image to target dimensions
            for i, img in enumerate(images):
                # Get original dimensions
                h, w = img.shape[1], img.shape[2]
                img_chw = img.movedim(-1, 1)  # HWC to CHW for upscale

                # Vision encoder: Resize to target dimensions
                vision_img = comfy.utils.common_upscale(img_chw, vision_target_w, vision_target_h, "bicubic", "disabled")
                vision_images.append(vision_img.movedim(1, -1))  # CHW to HWC

                if debug_mode:
                    logger.info(f"[Encoder] Image {i+1}: {w}x{h} -> vision: {vision_target_w}x{vision_target_h}")
                debug_info.append(f"Image {i+1}: {w}x{h} -> vision: {vision_target_w}x{vision_target_h}")

                # VAE encoder: Resize to target dimensions
                if vae is not None:
                    vae_img = comfy.utils.common_upscale(img_chw, vae_target_w, vae_target_h, "bicubic", "disabled")
                    vae_img_hwc = vae_img.movedim(1, -1)  # CHW to HWC

                    if debug_mode:
                        logger.info(f"[Encoder]        VAE: {w}x{h} -> {vae_target_w}x{vae_target_h}")
                    debug_info.append(f"    VAE: {vae_target_w}x{vae_target_h}")

                    ref_latent = vae.encode(vae_img_hwc[:, :, :, :3])
                    ref_latents.append(ref_latent)

                    if debug_mode:
                        logger.info(f"[Encoder]        Latent shape: {ref_latent.shape}")

        # Simple processing: Template Builder provides system prompt, we handle technical bits
        num_images = len(vision_images) if vision_images else 0

        # Validation removed for simplicity - users can check their own prompts

        # Build vision tokens based on mode and image count
        # DiffSynth uses two different template structures:
        # 1. image_edit (single): vision tokens BEFORE user prompt
        # 2. multi_image_edit: vision tokens WITH labels INSIDE user prompt
        if mode == "text_to_image":
            vision_tokens = ""
            debug_info.append(f"Mode: text_to_image, no vision tokens")
        elif mode == "multi_image_edit":
            # Multi-image mode: labels + vision tokens go INSIDE the prompt
            # Template: <system>{system_prompt}</system><user>{vision_tokens}{text_prompt}</user>
            if num_images == 0:
                vision_tokens = ""
                debug_info.append(f"Multi-image mode: No images provided")
            else:
                # Always use labels for multi_image_edit (DiffSynth standard)
                label_format = "Picture"
                vision_tokens = "".join([
                    f"{label_format} {i+1}: <|vision_start|><|image_pad|><|vision_end|>"
                    for i in range(num_images)
                ])
                debug_info.append(f"Multi-image mode: Added {label_format} 1-{num_images} labels")
                if debug_mode:
                    logger.info(f"[Encoder] Multi-image mode: {num_images} images with Picture labels")
        else:
            # image_edit or inpainting: vision tokens placement depends on image count
            if num_images == 0:
                vision_tokens = ""
                debug_info.append(f"No images provided")
            elif num_images == 1:
                # Single image: no label needed
                vision_tokens = "<|vision_start|><|image_pad|><|vision_end|>"
                debug_info.append(f"Single image mode (no Picture label)")
            else:
                # Multiple images in image_edit mode: optionally add labels
                if auto_label:
                    # DiffSynth standard: "Picture X:" labels
                    label_format = "Picture"
                    vision_tokens = "".join([
                        f"{label_format} {i+1}: <|vision_start|><|image_pad|><|vision_end|>"
                        for i in range(num_images)
                    ])
                    debug_info.append(f"Multi-image mode: Added {label_format} 1-{num_images} labels (auto_label=True)")
                else:
                    # No labels, just concatenate vision tokens
                    vision_tokens = "".join([
                        "<|vision_start|><|image_pad|><|vision_end|>"
                        for i in range(num_images)
                    ])
                    debug_info.append(f"Multi-image mode: No labels, raw vision tokens (auto_label=False)")

                if debug_mode:
                    logger.info(f"[Encoder] Auto-label: {auto_label}, Images: {num_images}")

        # Initialize formatted_text and drop_idx for debug output
        formatted_text = ""
        drop_idx = 0

        # Format the text with system prompt if we have a processor and system prompt
        # DiffSynth ALWAYS uses drop_idx based on mode, regardless of system prompt
        if self.processor and system_prompt:
            # Use processor to format with system prompt wrapper
            formatted_text = self.processor.format_template(text, system_prompt, vision_tokens)
            # DiffSynth always drops tokens for each mode when using their templates
            drop_idx = self.processor.get_drop_index(mode)
        else:
            # No system prompt - still apply DiffSynth drop indices if using standard format
            formatted_text = f"{vision_tokens}{text}" if mode != "text_to_image" else text
            # DiffSynth behavior: they always have templates, so always drop
            # But we allow no template, so only drop if we're in a vision mode
            drop_idx = 0  # Only drop when we have the full template

        if debug_mode:
            logger.info(f"[Encoder] Mode: {mode}, Images: {num_images}, Drop index: {drop_idx}")
            if self.processor and system_prompt:
                logger.info(f"[Encoder] Using processor with system prompt formatting and token dropping")
            elif system_prompt:
                logger.info(f"[Encoder] WARNING: System prompt provided but processor not available - no token dropping!")
            if num_images > 0:
                for i, img in enumerate(vision_images):
                    h, w = img.shape[1], img.shape[2]
                    logger.info(f"[Encoder]   Picture {i+1}: {w}x{h}")
            logger.info(f"[Encoder] Template (first 150 chars): {formatted_text[:150]}...")

        # Tokenize with vision support
        tokens = clip.tokenize(formatted_text, images=vision_images if vision_images else [])

        # Encode and then drop embeddings (like DiffSynth does)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        # Apply token dropping AFTER encoding if we formatted with system prompt
        # This matches DiffSynth's approach: encode first, then drop
        if drop_idx > 0:
            # Access the actual conditioning data
            # ComfyUI conditioning format: [[embeddings, dict]]
            for i, cond in enumerate(conditioning):
                if len(cond) >= 1 and isinstance(cond[0], torch.Tensor):
                    original_shape = cond[0].shape
                    # Drop the first drop_idx embeddings from sequence dimension
                    # Shape is typically [batch, sequence, hidden_dim]
                    if len(cond[0].shape) >= 2 and cond[0].shape[1] > drop_idx:
                        cond[0] = cond[0][:, drop_idx:, ...]
                        if debug_mode:
                            logger.info(f"[Encoder] Dropped first {drop_idx} embeddings: {original_shape} -> {cond[0].shape}")
                    elif debug_mode:
                        logger.info(f"[Encoder] Warning: Not enough tokens to drop. Shape: {original_shape}, drop_idx: {drop_idx}")

        # Add reference latents to conditioning if available
        if ref_latents and COMFY_AVAILABLE:
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": ref_latents},
                append=True
            )

            # Don't manipulate dimensions - pass through as-is

            if debug_mode:
                logger.info(f"[Encoder] Added {len(ref_latents)} reference latents to conditioning")
                debug_info.append(f"Reference latents: {len(ref_latents)} with auto-dimension handling")
                # Add dimension check warning
                if ref_latents:
                    latent_shapes = [str(lat.shape) for lat in ref_latents]
                    unique_shapes = set(latent_shapes)
                    if len(unique_shapes) > 1:
                        debug_info.append(f"\nWARNING: Reference latents have different shapes: {unique_shapes}")
                        debug_info.append("This may cause generation issues. Ensure all images have similar dimensions.")

        # Add inpaint mask to conditioning if provided
        if inpaint_mask is not None and COMFY_AVAILABLE:
            # Resize mask to match VAE input dimensions if we have reference latents
            if ref_latents and len(ref_latents) > 0:
                # Get VAE dimensions from first image processing
                # ref_latents[0] shape is [B, C, H/8, W/8] so we need to match the pre-encoded dimensions
                # We need to get vae_w, vae_h from the image processing loop
                # For now, resize mask to match first reference latent dimensions * 8 (latent->pixel space)
                first_latent = ref_latents[0]
                latent_h, latent_w = first_latent.shape[-2], first_latent.shape[-1]
                target_h, target_w = latent_h * 8, latent_w * 8

                # Resize mask to match VAE input dimensions
                mask_batch = inpaint_mask.unsqueeze(1)  # Add channel dimension
                resized_mask = comfy.utils.common_upscale(mask_batch, target_w, target_h, "bicubic", "disabled")
                resized_mask = resized_mask.squeeze(1)  # Remove channel dimension

                if debug_mode:
                    logger.info(f"[Encoder] Resized mask from {inpaint_mask.shape} to {resized_mask.shape} to match VAE dimensions {target_w}x{target_h}")
                    debug_info.append(f"Mask resized: {inpaint_mask.shape} → {resized_mask.shape}")

                inpaint_mask = resized_mask

            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"inpaint_mask": inpaint_mask},
                append=True
            )

            if debug_mode:
                logger.info(f"[Encoder] Added inpaint mask to conditioning: {inpaint_mask.shape}")
                debug_info.append(f"Inpaint mask: {inpaint_mask.shape}")

        # Build debug output string
        if debug_mode and debug_info:
            debug_output = "=== QWEN ENCODER DEBUG ===\n\n"
            debug_output += "\n".join(debug_info)

            # Show vision tokens (truncate if very long)
            vision_display = vision_tokens[:200] + "..." if len(vision_tokens) > 200 else vision_tokens

            debug_output += f"\n\n=== VISION TOKENS ===\n{vision_display}"

            # Show FULL formatted text without truncation for debugging
            debug_output += f"\n\n=== FULL PROMPT BEING ENCODED ===\n{formatted_text}"

            # Also show just the user prompt part for clarity
            if vision_tokens and text:
                debug_output += f"\n\n=== USER PROMPT (without vision tokens) ===\n{text}"

            debug_output += f"\n\n=== SETTINGS ===\nMode: {mode}\nImages: {num_images}\nDrop Index: {drop_idx}"
            debug_output += f"\nSystem Prompt: {'Yes' if system_prompt else 'No'}"

            # Show character counts for reference
            debug_output += f"\n\n=== CHARACTER COUNTS ===\nTotal prompt length: {len(formatted_text)} chars"
            debug_output += f"\nVision tokens: {len(vision_tokens)} chars"
            debug_output += f"\nUser prompt: {len(text)} chars"
            if system_prompt:
                debug_output += f"\nSystem prompt: {len(system_prompt)} chars"
        else:
            debug_output = "Enable debug_mode to see detailed output"

        return (conditioning, debug_output)


class QwenLowresFixNode:
    """
    Makes your image BETTER with two-stage refinement.
    Stage 1: Generate at current size
    Stage 2: Upscale and polish details

    Connect AFTER your first KSampler for quality boost!
    """

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except:
            samplers = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral"]
            schedulers = ["normal", "karras", "exponential", "simple", "ddim_uniform"]

        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,                }),
                "upscale_factor": ("FLOAT", {
                    "default": 1.5,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Upscale by 1.5x recommended",
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process"
    CATEGORY = "QwenImage/Refinement"
    TITLE = "Qwen Lowres Fix"
    DESCRIPTION = "Two-stage refinement for higher quality (DiffSynth-Studio method)"

    def process(self, model, positive, negative, latent, vae, seed, steps,
                cfg, sampler_name, scheduler, denoise, upscale_factor):
        import comfy.samplers
        import comfy.utils

        sampler = comfy.samplers.KSampler()

        # Stage 1: Full generation at current resolution
        stage1_samples = sampler.sample(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=1.0
        )[0]

        # Decode to image space
        decoded = vae.decode(stage1_samples["samples"])

        # Upscale the image
        h, w = decoded.shape[1], decoded.shape[2]
        new_h = int(h * upscale_factor)
        new_w = int(w * upscale_factor)

        decoded_chw = decoded.movedim(-1, 1)
        upscaled = comfy.utils.common_upscale(
            decoded_chw, new_w, new_h, "bicubic", "disabled"
        )
        upscaled_hwc = upscaled.movedim(1, -1)

        # Encode back to latent space
        stage2_latent = vae.encode(upscaled_hwc[:, :, :, :3])

        # Stage 2: Refinement with partial denoise
        refined_samples = sampler.sample(
            model=model,
            seed=seed + 1,
            steps=max(steps // 2, 10),
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image={"samples": stage2_latent},
            denoise=denoise
        )[0]

        return (refined_samples,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenVLCLIPLoader": QwenVLCLIPLoader,
    "QwenVLTextEncoder": QwenVLTextEncoder,
    "QwenLowresFixNode": QwenLowresFixNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLCLIPLoader": "Qwen2.5-VL CLIP Loader",
    "QwenVLTextEncoder": "Qwen2.5-VL Text Encoder",
    "QwenLowresFixNode": "Qwen Lowres Fix (Two-Stage)",
}
