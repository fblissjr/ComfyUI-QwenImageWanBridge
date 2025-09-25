"""
Qwen2.5-VL CLIP Wrapper for ComfyUI
Uses ComfyUI's internal Qwen loader but with DiffSynth-Studio templates
Includes all fixes from DiffSynth-Studio and DiffSynth-Engine
"""

import os
import torch
import logging
from typing import Optional, Dict, Any, Tuple, Union, List
import folder_paths

# Simplified encoder - no longer needs complex processors

logger = logging.getLogger(__name__)

# Import our custom processor for proper token dropping
try:
    from .qwen_processor_v2 import QwenProcessorV2
    PROCESSOR_AVAILABLE = True
except ImportError:
    PROCESSOR_AVAILABLE = False
    logger.warning("QwenProcessorV2 not available - will use simplified approach")

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
                    "default": "A beautiful landscape",
                    "tooltip": "Your prompt"
                }),
                "mode": (["text_to_image", "image_edit"], {
                    "default": "image_edit",
                    "tooltip": "text_to_image: Generate from scratch | image_edit: Modify existing image"
                }),
            },
            "optional": {
                "edit_image": ("IMAGE", {
                    "tooltip": "Single image or batch. For multiple images, use Image Batch node first."
                }),
                "vae": ("VAE", {
                    "tooltip": "Required for image editing - encodes reference latents"
                }),
                "system_prompt": ("STRING", {
                    "tooltip": "System prompt override from Template Builder (optional)",
                    "multiline": True,
                    "default": ""
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show processing details in console"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "QwenImage/Encoding"
    TITLE = "Qwen2.5-VL Text Encoder"
    DESCRIPTION = "DiffSynth/Diffusers-compatible encoder with 32-pixel alignment"

    @staticmethod
    def calculate_dimensions(target_area: int, aspect_ratio: float) -> tuple:
        """Calculate dimensions matching target area while preserving aspect ratio.
        Uses 32-pixel alignment for VAE compatibility (DiffSynth/Diffusers standard)."""
        import math
        width = math.sqrt(target_area * aspect_ratio)
        height = width / aspect_ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return max(32, int(width)), max(32, int(height))



    def encode(self, clip, text: str, mode: str = "text_to_image",
              edit_image: Optional[torch.Tensor] = None,
              vae=None, system_prompt: str = "", debug_mode: bool = False) -> Tuple[Any]:

        """Encode text and images for Qwen Image generation.
        Follows DiffSynth/Diffusers implementation with 32-pixel alignment."""

        import math
        import comfy.utils

        vision_images = []
        ref_latents = []

        # Process images if provided
        if mode == "image_edit" and edit_image is not None:
            if debug_mode:
                logger.info(f"[Encoder] Input shape: {edit_image.shape}")

            # Convert batch tensor to list of images
            images = [edit_image[i:i+1, :, :, :3] for i in range(edit_image.shape[0])]

            # Process each image with DiffSynth/Diffusers standard sizing
            for i, img in enumerate(images):
                # Get original dimensions
                h, w = img.shape[1], img.shape[2]
                aspect_ratio = w / h

                # Resize for vision encoder (384x384 target area)
                vision_w, vision_h = self.calculate_dimensions(384*384, aspect_ratio)
                img_chw = img.movedim(-1, 1)  # HWC to CHW for upscale
                vision_img = comfy.utils.common_upscale(img_chw, vision_w, vision_h, "bicubic", "disabled")
                vision_images.append(vision_img.movedim(1, -1))  # CHW to HWC

                if debug_mode:
                    logger.info(f"[Encoder] Image {i+1}: {w}x{h} -> vision: {vision_w}x{vision_h}")

                # Resize for VAE encoder (1024x1024 target area) if VAE provided
                if vae is not None:
                    vae_w, vae_h = self.calculate_dimensions(1024*1024, aspect_ratio)
                    vae_img = comfy.utils.common_upscale(img_chw, vae_w, vae_h, "bicubic", "disabled")
                    vae_img_hwc = vae_img.movedim(1, -1)
                    ref_latent = vae.encode(vae_img_hwc[:, :, :, :3])
                    ref_latents.append(ref_latent)

                    if debug_mode:
                        logger.info(f"[Encoder]        VAE: {vae_w}x{vae_h}, latent: {ref_latent.shape}")

        # Simple processing: Template Builder provides system prompt, we handle technical bits
        num_images = len(vision_images) if vision_images else 0

        # Build vision tokens for image editing
        if mode == "text_to_image":
            vision_tokens = ""
            formatted_text = text
        else:
            # Build vision tokens based on number of images
            if num_images == 0:
                vision_tokens = ""
            elif num_images == 1:
                vision_tokens = "<|vision_start|><|image_pad|><|vision_end|>"
            else:
                vision_tokens = "".join([
                    f"Picture {i+1}: <|vision_start|><|image_pad|><|vision_end|>"
                    for i in range(num_images)
                ])
            formatted_text = f"{vision_tokens}{text}"

        # Get drop index based on mode and whether system prompt exists
        # DiffSynth drops 34 for text_to_image, 64 for image_edit
        drop_idx = 0
        if system_prompt:
            drop_idx = 34 if mode == "text_to_image" else 64

        if debug_mode:
            logger.info(f"[Encoder] Mode: {mode}, Images: {num_images}, Drop index: {drop_idx}")
            if num_images > 0:
                for i, img in enumerate(vision_images):
                    h, w = img.shape[1], img.shape[2]
                    logger.info(f"[Encoder]   Picture {i+1}: {w}x{h}")
            logger.info(f"[Encoder] Template (first 150 chars): {formatted_text[:150]}...")

        # Tokenize with vision support
        tokens = clip.tokenize(formatted_text, images=vision_images if vision_images else [])

        # Encode and then drop embeddings (like DiffSynth does)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        # Apply token dropping AFTER encoding if we have a system prompt
        # This matches DiffSynth's approach: encode first, then drop
        if drop_idx > 0 and system_prompt:
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
            if debug_mode:
                logger.info(f"[Encoder] Added {len(ref_latents)} reference latents to conditioning")

        return (conditioning,)


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
