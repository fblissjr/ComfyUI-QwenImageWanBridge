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


class QwenVLTextEncoderProper:
    """
    Text encoder that uses ComfyUI's CLIP but with DiffSynth-Studio templates
    This is the proper implementation that actually works with the diffusion pipeline
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
                "use_diffsynth_template": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use DiffSynth-Studio templates for consistency"
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
              edit_image: Optional[torch.Tensor] = None,
              use_diffsynth_template: bool = True) -> Tuple[Any]:
        """
        Encode text using ComfyUI's CLIP but with optional DiffSynth templates
        """
        
        images = []
        
        # Prepare image if in edit mode
        if mode == "image_edit" and edit_image is not None:
            # Convert tensor to PIL
            import numpy as np
            from PIL import Image
            
            if isinstance(edit_image, torch.Tensor):
                if edit_image.dim() == 4:
                    edit_image = edit_image[0]
                image_np = (edit_image.cpu().numpy() * 255).astype(np.uint8)
                if image_np.shape[0] == 3:  # CHW to HWC
                    image_np = np.transpose(image_np, (1, 2, 0))
                image_pil = Image.fromarray(image_np)
                images = [image_pil]
        
        # Choose template
        if use_diffsynth_template:
            if mode == "image_edit" and images:
                # DiffSynth-Studio template for image editing
                template = (
                    "<|im_start|>system\n"
                    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
                    "then explain how the user's text instruction should alter or modify the image. "
                    "Generate a new image that meets the user's requirements while maintaining consistency "
                    "with the original input where appropriate.<|im_end|>\n"
                    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            else:
                # DiffSynth-Studio template for text-to-image
                template = (
                    "<|im_start|>system\n"
                    "You are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\n{}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            
            prompt = template.format(text)
            
            # Override ComfyUI's template by tokenizing with our prompt
            if images:
                # For image edit, pass the template directly
                # ComfyUI will handle the vision tokens
                tokens = clip.tokenize(prompt, images=images)
            else:
                tokens = clip.tokenize(prompt)
        else:
            # Use ComfyUI's default templates
            if images:
                tokens = clip.tokenize(text, images=images)
            else:
                tokens = clip.tokenize(text)
        
        # Encode tokens using ComfyUI's method
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        logger.info(f"Encoded text in {mode} mode with DiffSynth={use_diffsynth_template}")
        
        return (conditioning,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenVLCLIPLoader": QwenVLCLIPLoader,
    "QwenVLTextEncoderProper": QwenVLTextEncoderProper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLCLIPLoader": "Qwen2.5-VL CLIP Loader",
    "QwenVLTextEncoderProper": "Qwen2.5-VL Text Encoder",
}