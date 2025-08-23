"""
Qwen2.5-VL Text Encoder with Vision Support
Properly handles multimodal encoding for both T2I and Image Edit modes
"""

import torch
import numpy as np
from PIL import Image
import logging
from typing import Optional, Dict, Any, Tuple, Union

logger = logging.getLogger(__name__)

class QwenVLTextEncoder:
    """
    Text encoder that properly handles Qwen2.5-VL's vision tokens
    and multimodal processing for both T2I and Image Edit modes
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_VL_MODEL",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape"
                }),
                "mode": (["text_to_image", "image_edit"], {
                    "default": "text_to_image"
                }),
            },
            "optional": {
                "edit_image": ("IMAGE",),  # For image edit mode
                "apply_template": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply Qwen's system prompt template"
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("conditioning", "vision_features")
    FUNCTION = "encode"
    CATEGORY = "QwenImage/Encoding"
    DESCRIPTION = "Encode text with optional vision tokens for Qwen Image generation"
    
    def encode(self, qwen_model, text: str, mode: str = "text_to_image",
              edit_image: Optional[torch.Tensor] = None, 
              apply_template: bool = True) -> Tuple[Any, Dict]:
        """
        Encode text and optionally image for Qwen model
        
        Args:
            qwen_model: QwenVLModelWrapper from QwenVLLoader
            text: Input prompt text
            mode: Either "text_to_image" or "image_edit"
            edit_image: Optional image tensor for edit mode [B, H, W, C]
            apply_template: Whether to apply Qwen's system prompts
            
        Returns:
            Tuple of (conditioning, vision_features)
        """
        
        # Apply template if requested
        if apply_template:
            if mode == "text_to_image":
                # T2I template from DiffSynth
                template = (
                    "<|im_start|>system\n"
                    "Describe the image by detailing the color, shape, size, texture, "
                    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                    "<|im_start|>user\n{}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                text = template.format(text)
                drop_start = 34  # Tokens to drop from template
                
            elif mode == "image_edit" and edit_image is not None:
                # Image edit template with vision tokens - EXACT from DiffSynth-Studio
                template = (
                    "<|im_start|>system\n"
                    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
                    "then explain how the user's text instruction should alter or modify the image. "
                    "Generate a new image that meets the user's requirements while maintaining consistency "
                    "with the original input where appropriate.<|im_end|>\n"
                    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                text = template.format(text)
                drop_start = 64  # Tokens to drop from template
        else:
            drop_start = 0
        
        # Handle based on mode
        if mode == "image_edit" and edit_image is not None:
            # Multimodal encoding with image
            embeddings, vision_info = qwen_model.encode_multimodal(text, edit_image)
            
            # Drop template tokens if applied
            if apply_template and drop_start > 0:
                embeddings = embeddings[:, drop_start:]
            
            # Create conditioning format for ComfyUI
            conditioning = [[embeddings, {}]]
            
            # Package vision features for downstream use
            vision_features = {
                "samples": embeddings,  # For compatibility
                "has_vision": True,
                "vision_info": vision_info,
                "mode": "image_edit"
            }
            
        else:
            # Text-only encoding
            embeddings = qwen_model.encode_text(text)
            
            # Drop template tokens if applied  
            if apply_template and drop_start > 0:
                embeddings = embeddings[:, drop_start:]
            
            # Create conditioning format for ComfyUI
            conditioning = [[embeddings, {}]]
            
            # No vision features for T2I
            vision_features = {
                "samples": embeddings,  # For compatibility
                "has_vision": False,
                "mode": "text_to_image"
            }
        
        return (conditioning, vision_features)

class QwenVLEmptyLatent:
    """
    Generate empty latents compatible with Qwen's 16-channel VAE
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64
                }),
                "channels": ("INT", {
                    "default": 16,
                    "min": 16,
                    "max": 16,
                    "tooltip": "Qwen uses 16-channel VAE"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "QwenImage/Latents"
    DESCRIPTION = "Generate empty 16-channel latents for Qwen Image"
    
    def generate(self, width: int, height: int, batch_size: int = 1, channels: int = 16):
        """Generate empty latents"""
        
        # Calculate latent dimensions (VAE downscale by 8)
        latent_height = height // 8
        latent_width = width // 8
        
        # Generate empty latents
        latent = torch.zeros(
            (batch_size, channels, latent_height, latent_width),
            dtype=torch.float32
        )
        
        # Apply Qwen VAE normalization
        # Mean and std from DiffSynth qwen_image_vae.py
        mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]).view(1, 16, 1, 1)
        
        std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]).view(1, 16, 1, 1)
        
        # Initialize with normalized noise
        noise = torch.randn_like(latent)
        latent = noise * std + mean
        
        return ({"samples": latent},)

class QwenVLImageToLatent:
    """
    Encode image to Qwen's 16-channel latent space
    Note: Requires actual Qwen VAE model for proper encoding
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),  # Should be Qwen's 16-channel VAE
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "QwenImage/Latents"
    DESCRIPTION = "Encode image using Qwen's 16-channel VAE"
    
    def encode(self, image: torch.Tensor, vae):
        """Encode image to latent space"""
        
        # Image is [B, H, W, C] in ComfyUI
        # Convert to [B, C, H, W] for VAE
        if image.dim() == 4:
            image = image.permute(0, 3, 1, 2)
        elif image.dim() == 3:
            image = image.unsqueeze(0).permute(0, 3, 1, 2)
        
        # Ensure image is in [-1, 1] range (Qwen's expected range)
        if image.max() > 1.0:
            image = (image / 255.0) * 2.0 - 1.0
        elif image.min() >= 0.0 and image.max() <= 1.0:
            image = image * 2.0 - 1.0
        
        # Check if this is actually a Qwen VAE
        try:
            # Qwen VAE needs temporal dimension for 3D convolutions
            # Add temporal dimension: [B, C, H, W] -> [B, C, 1, H, W]
            image_3d = image.unsqueeze(2)
            
            # Encode through VAE
            latent = vae.encode(image_3d)
            
            # Remove temporal dimension and take first 16 channels
            if latent.dim() == 5:
                latent = latent.squeeze(2)  # Remove temporal
            if latent.shape[1] > 16:
                latent = latent[:, :16]  # Take first 16 channels
            
            # Apply Qwen normalization
            mean = torch.tensor([
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
            ]).view(1, 16, 1, 1).to(latent.device)
            
            std = torch.tensor([
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
            ]).view(1, 16, 1, 1).to(latent.device)
            
            latent = (latent - mean) * std
            
        except Exception as e:
            logger.warning(f"Failed to encode with Qwen VAE: {e}")
            logger.warning("Falling back to standard VAE encoding - results may be incorrect")
            
            # Fallback to standard encoding
            latent = vae.encode(image)
            
            # Pad to 16 channels if needed
            if latent.shape[1] < 16:
                padding = torch.zeros(
                    (latent.shape[0], 16 - latent.shape[1], latent.shape[2], latent.shape[3]),
                    device=latent.device,
                    dtype=latent.dtype
                )
                latent = torch.cat([latent, padding], dim=1)
        
        return ({"samples": latent},)

# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenVLTextEncoder": QwenVLTextEncoder,
    "QwenVLEmptyLatent": QwenVLEmptyLatent,
    "QwenVLImageToLatent": QwenVLImageToLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLTextEncoder": "Qwen2.5-VL Text Encoder",
    "QwenVLEmptyLatent": "Qwen Empty Latent (16ch)",
    "QwenVLImageToLatent": "Qwen Image to Latent (16ch)",
}