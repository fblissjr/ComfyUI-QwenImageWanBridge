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


class ZImageEmptyLatent:
    """
    Creates empty 16-channel latents for Z-Image with auto-alignment.

    Z-Image requires 16-pixel alignment. This node auto-corrects any input
    to the nearest valid resolution.

    Examples:
        1211x1024 → 1216x1024 (1211 rounds up to 1216)
        1000x1000 → 1008x1008 (1000 rounds to 1008)
        512x768   → 512x768   (already aligned)
    """

    # Z-Image uses 16-pixel alignment (Flux-derived VAE with 8x compression)
    ALIGNMENT = 16

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1024,
                    "min": 16,
                    "max": 8192,
                    "step": 1,  # Allow any value, we'll auto-correct
                    "tooltip": "Width in pixels (auto-aligned to 16px)"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 16,
                    "max": 8192,
                    "step": 1,  # Allow any value, we'll auto-correct
                    "tooltip": "Height in pixels (auto-aligned to 16px)"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4096,
                    "tooltip": "Number of latents to generate"
                })
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width", "height", "resolution_info")
    FUNCTION = "generate"
    CATEGORY = "ZImage/Latent"
    TITLE = "Z-Image Empty Latent"

    DESCRIPTION = "Creates 16-channel latents with auto-alignment to 16px (Z-Image requirement)."

    def generate(self, width: int, height: int, batch_size: int = 1):
        """Generate empty 16-channel latents with auto-aligned dimensions."""

        # Auto-align to 16-pixel boundaries (round to nearest)
        aligned_width = round(width / self.ALIGNMENT) * self.ALIGNMENT
        aligned_height = round(height / self.ALIGNMENT) * self.ALIGNMENT

        # Ensure minimum dimensions
        aligned_width = max(self.ALIGNMENT, aligned_width)
        aligned_height = max(self.ALIGNMENT, aligned_height)

        # Convert pixel dimensions to latent dimensions (8x downscale)
        latent_width = aligned_width // 8
        latent_height = aligned_height // 8

        # Z-Image uses 16-channel latents (Flux-derived VAE)
        latent = torch.zeros([batch_size, 16, latent_height, latent_width])

        # Build info string
        if width != aligned_width or height != aligned_height:
            resolution_info = f"{width}x{height} → {aligned_width}x{aligned_height} (auto-aligned)"
        else:
            resolution_info = f"{aligned_width}x{aligned_height} (already aligned)"

        return ({"samples": latent}, aligned_width, aligned_height, resolution_info)


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
    "QwenVLImageToLatent": QwenVLImageToLatent,
    "ZImageEmptyLatent": ZImageEmptyLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLEmptyLatent": "Qwen VL Empty Latent",
    "QwenVLImageToLatent": "Qwen VL Image to Latent",
    "ZImageEmptyLatent": "Z-Image Empty Latent",
}