"""
Qwen VL Helper Nodes
Helper nodes for the Qwen Image Edit workflow
"""

import torch
import numpy as np
from PIL import Image


class QwenVLEmptyLatent:
    """Creates empty 16-channel latents for Qwen Image Edit model"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1024, 
                    "min": 16, 
                    "max": 8192, 
                    "step": 8,
                    "tooltip": "Width in pixels (will be converted to latent space)"
                }),
                "height": ("INT", {
                    "default": 1024, 
                    "min": 16, 
                    "max": 8192, 
                    "step": 8,
                    "tooltip": "Height in pixels (will be converted to latent space)"
                }),
                "batch_size": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 4096,
                    "tooltip": "Number of latents to generate"
                })
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "Qwen/Latent"
    
    def generate(self, width: int, height: int, batch_size: int = 1):
        """Generate empty 16-channel latents"""
        # Convert pixel dimensions to latent dimensions (8x downscale)
        latent_width = width // 8
        latent_height = height // 8
        
        # Qwen Image Edit uses 16-channel latents
        latent = torch.zeros([batch_size, 16, latent_height, latent_width])
        
        return ({"samples": latent},)


class QwenVLImageToLatent:
    """Converts images to 16-channel latents using VAE encoder"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "vae": ("VAE",)
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "Qwen/Latent"
    
    def encode(self, vae, images):
        """Encode images to latent space"""
        # Use VAE to encode images to latent space
        # This should work with any 16-channel VAE
        latent = vae.encode(images[:, :, :, :3])  # Remove alpha channel if present
        return ({"samples": latent},)


NODE_CLASS_MAPPINGS = {
    "QwenVLEmptyLatent": QwenVLEmptyLatent,
    "QwenVLImageToLatent": QwenVLImageToLatent
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLEmptyLatent": "Qwen VL Empty Latent",
    "QwenVLImageToLatent": "Qwen VL Image to Latent"
}