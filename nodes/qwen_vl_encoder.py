"""
Qwen2.5-VL CLIP Wrapper for ComfyUI
Uses ComfyUI's internal Qwen loader but with DiffSynth-Studio templates
"""

import os
import torch
import logging
from typing import Optional, Dict, Any, Tuple, Union
import folder_paths

logger = logging.getLogger(__name__)

# Try to import ComfyUI's utilities
try:
    import comfy.sd
    import comfy.model_management as mm
    COMFY_AVAILABLE = True
except ImportError:
    logger.warning("ComfyUI utilities not available")
    COMFY_AVAILABLE = False

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
    Text encoder for Qwen2.5-VL that works with the diffusion pipeline
    Uses ComfyUI's internal CLIP infrastructure for compatibility
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape"
                }),
                "mode": (["text_to_image", "image_edit"], {
                    "default": "text_to_image"
                }),
            },
            "optional": {
                "edit_image": ("IMAGE",),
                "vae": ("VAE",),  # Optional VAE for reference latents like official
                "debug_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable debug logging"
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "QwenImage/Encoding"
    TITLE = "Qwen2.5-VL Text Encoder"
    DESCRIPTION = "Encode text using ComfyUI's CLIP with DiffSynth-Studio templates"
    
    def encode(self, clip, text: str, mode: str = "text_to_image",
              edit_image: Optional[torch.Tensor] = None, vae=None, debug_mode: bool = True) -> Tuple[Any]:
        """
        Encode text using ComfyUI's CLIP but with optional DiffSynth templates
        """
        
        images = []
        ref_latent = None  # Initialize here so it's in scope later
        
        # Prepare image if in edit mode - following ComfyUI's TextEncodeQwenImageEdit pattern
        if mode == "image_edit" and edit_image is not None:
            # Process image tensor like the official node does
            import math
            import comfy.utils
            
            if debug_mode:
                logger.info(f"[DEBUG] Input image shape: {edit_image.shape}")
                logger.info(f"[DEBUG] Input image dtype: {edit_image.dtype}")
                logger.info(f"[DEBUG] Input image min/max: {edit_image.min():.4f}/{edit_image.max():.4f}")
            
            # ComfyUI IMAGE is [B, H, W, C], we need to process it
            samples = edit_image.movedim(-1, 1)  # [B, H, W, C] -> [B, C, H, W]
            
            if debug_mode:
                logger.info(f"[DEBUG] After movedim shape: {samples.shape}")
            
            # Scale to target resolution (1024x1024 total pixels like official)
            total = int(1024 * 1024)
            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)
            
            if debug_mode:
                logger.info(f"[DEBUG] Scaling from {samples.shape[3]}x{samples.shape[2]} to {width}x{height}")
                logger.info(f"[DEBUG] Scale factor: {scale_by:.4f}")
            
            # Resize using ComfyUI's common_upscale
            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            image = s.movedim(1, -1)  # [B, C, H, W] -> [B, H, W, C]
            
            if debug_mode:
                logger.info(f"[DEBUG] After resize shape: {image.shape}")
                logger.info(f"[DEBUG] After resize min/max: {image.min():.4f}/{image.max():.4f}")
            
            # Extract RGB channels (drop alpha if present)
            images = [image[:, :, :, :3]]
            
            if debug_mode:
                logger.info(f"[DEBUG] Final image shape for tokenizer: {images[0].shape}")
                logger.info(f"[DEBUG] Final image dtype: {images[0].dtype}")
                logger.info(f"[DEBUG] Final image min/max: {images[0].min():.4f}/{images[0].max():.4f}")
            
            # Add reference latents if VAE provided (like official node)
            if vae is not None:
                ref_latent = vae.encode(image[:, :, :, :3])
                if debug_mode:
                    logger.info(f"[DEBUG] Encoded reference latent shape: {ref_latent.shape}")
                    logger.info(f"[DEBUG] Reference latent min/max: {ref_latent.min():.4f}/{ref_latent.max():.4f}")
        
        # Important: ComfyUI ALREADY applies the correct DiffSynth template internally!
        # The qwen_image.py tokenizer uses the exact same template we want
        # We should NOT apply the template ourselves - that causes double templating
        
        # Just pass the text and images directly - ComfyUI handles the rest
        if debug_mode:
            logger.info(f"[DEBUG] Tokenizing with text: '{text[:50]}...' and {len(images)} images")
        
        if images:
            tokens = clip.tokenize(text, images=images)
        else:
            tokens = clip.tokenize(text)
        
        # Debug token info
        if debug_mode and isinstance(tokens, dict):
            for key in tokens:
                if isinstance(tokens[key], list) and len(tokens[key]) > 0:
                    logger.info(f"[DEBUG] Token key '{key}' has {len(tokens[key][0])} tokens")
        
        # Encode tokens using ComfyUI's method
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        # Add reference latents if we have them (like official node)
        if ref_latent is not None:
            import node_helpers
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]}, append=True)
            if debug_mode:
                logger.info(f"[DEBUG] Added reference latents to conditioning")
        
        # Debug conditioning
        if debug_mode and isinstance(conditioning, list) and len(conditioning) > 0:
            cond_tensor = conditioning[0][0]
            logger.info(f"[DEBUG] Conditioning shape: {cond_tensor.shape}")
            logger.info(f"[DEBUG] Conditioning dtype: {cond_tensor.dtype}")
            logger.info(f"[DEBUG] Conditioning min/max: {cond_tensor.min():.4f}/{cond_tensor.max():.4f}")
            logger.info(f"[DEBUG] Conditioning mean/std: {cond_tensor.mean():.4f}/{cond_tensor.std():.4f}")
            
            # Check if there's metadata
            if len(conditioning[0]) > 1:
                metadata = conditioning[0][1]
                logger.info(f"[DEBUG] Conditioning metadata keys: {metadata.keys() if isinstance(metadata, dict) else 'Not a dict'}")
                if isinstance(metadata, dict) and 'reference_latents' in metadata:
                    logger.info(f"[DEBUG] Has reference_latents: {len(metadata['reference_latents'])} items")
        
        if debug_mode:
            logger.info(f"[DEBUG] Encoded text in {mode} mode")
        
        return (conditioning,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenVLCLIPLoader": QwenVLCLIPLoader,
    "QwenVLTextEncoder": QwenVLTextEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLCLIPLoader": "Qwen2.5-VL CLIP Loader",
    "QwenVLTextEncoder": "Qwen2.5-VL Text Encoder",
}