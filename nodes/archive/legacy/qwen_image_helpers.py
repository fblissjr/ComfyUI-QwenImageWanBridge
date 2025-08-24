"""
Qwen-Image Helper Nodes
Utility nodes for resolution, VAE encoding, and format conversion
"""

import torch
import numpy as np
from PIL import Image
import math
from typing import Tuple, Dict, List, Optional

class QwenImageResolutionHelper:
    """
    Calculate optimal resolutions for Qwen-Image generation
    Based on 1024x1024 target area with aspect ratio preservation
    """
    
    # Predefined optimal resolutions (all maintain ~1M pixels)
    OPTIMAL_RESOLUTIONS = [
        # Square
        (1024, 1024),
        (768, 768),
        (512, 512),
        # Portrait (3:4, 2:3, 9:16)
        (832, 1216),
        (768, 1024),
        (640, 1024),
        (576, 1024),
        # Landscape (4:3, 3:2, 16:9)  
        (1216, 832),
        (1024, 768),
        (1024, 640),
        (1024, 576),
        # Ultra-wide/tall
        (1344, 768),
        (768, 1344),
        (1536, 640),
        (640, 1536),
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["closest", "contain", "cover", "exact"], {"default": "closest"}),
                "target_pixels": ("INT", {"default": 1048576, "min": 262144, "max": 4194304, "step": 65536}),
            },
            "optional": {
                "force_divisible": ("INT", {"default": 32, "min": 8, "max": 64}),
                "max_dimension": ("INT", {"default": 2048, "min": 512, "max": 4096}),
                "min_dimension": ("INT", {"default": 512, "min": 256, "max": 1024}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "DICT")
    RETURN_NAMES = ("image", "width", "height", "resize_info")
    FUNCTION = "calculate_resolution"
    CATEGORY = "QwenImage/Utils"
    
    def find_closest_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """Find the closest predefined resolution"""
        aspect = width / height
        min_diff = float('inf')
        best_res = (1024, 1024)
        
        for res_w, res_h in self.OPTIMAL_RESOLUTIONS:
            res_aspect = res_w / res_h
            aspect_diff = abs(aspect - res_aspect)
            pixel_diff = abs((width * height) - (res_w * res_h)) / 1000000
            total_diff = aspect_diff + pixel_diff * 0.5
            
            if total_diff < min_diff:
                min_diff = total_diff
                best_res = (res_w, res_h)
        
        return best_res
    
    def calculate_exact_resolution(self, width: int, height: int, target_pixels: int, 
                                  divisible: int = 32) -> Tuple[int, int]:
        """Calculate exact resolution maintaining aspect ratio"""
        aspect = width / height
        
        # Calculate dimensions for target pixel count
        new_width = math.sqrt(target_pixels * aspect)
        new_height = new_width / aspect
        
        # Round to nearest divisible value
        new_width = round(new_width / divisible) * divisible
        new_height = round(new_height / divisible) * divisible
        
        return int(new_width), int(new_height)
    
    def calculate_resolution(self, image: torch.Tensor, mode: str, target_pixels: int,
                           force_divisible: int = 32, max_dimension: int = 2048,
                           min_dimension: int = 512) -> Tuple[torch.Tensor, int, int, Dict]:
        """
        Calculate and apply optimal resolution for Qwen-Image
        """
        # Get original dimensions
        if len(image.shape) == 4:
            B, H, W, C = image.shape
        else:
            H, W, C = image.shape
            image = image.unsqueeze(0)
            B = 1
        
        resize_info = {
            "original_size": (W, H),
            "mode": mode,
            "target_pixels": target_pixels
        }
        
        # Calculate new dimensions based on mode
        if mode == "closest":
            new_w, new_h = self.find_closest_resolution(W, H)
        elif mode == "exact":
            new_w, new_h = self.calculate_exact_resolution(W, H, target_pixels, force_divisible)
        elif mode == "contain":
            # Scale to fit within target pixels
            scale = math.sqrt(target_pixels / (W * H))
            if scale < 1.0:
                new_w = int(W * scale / force_divisible) * force_divisible
                new_h = int(H * scale / force_divisible) * force_divisible
            else:
                new_w, new_h = W, H
        elif mode == "cover":
            # Scale to cover target pixels (may exceed)
            scale = math.sqrt(target_pixels / (W * H))
            new_w = int(W * scale / force_divisible) * force_divisible
            new_h = int(H * scale / force_divisible) * force_divisible
        else:
            new_w, new_h = W, H
        
        # Apply dimension limits
        new_w = max(min_dimension, min(new_w, max_dimension))
        new_h = max(min_dimension, min(new_h, max_dimension))
        
        # Ensure divisibility
        new_w = (new_w // force_divisible) * force_divisible
        new_h = (new_h // force_divisible) * force_divisible
        
        resize_info["new_size"] = (new_w, new_h)
        resize_info["scale_factor"] = (new_w / W, new_h / H)
        resize_info["pixel_count"] = new_w * new_h
        
        # Resize image if needed
        if (new_w, new_h) != (W, H):
            # Convert to PIL for high-quality resize
            output_images = []
            for b in range(B):
                img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                img_tensor = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
                output_images.append(img_tensor)
            
            image = torch.stack(output_images)
            resize_info["resized"] = True
        else:
            resize_info["resized"] = False
        
        return (image, new_w, new_h, resize_info)


class QwenImageVAEEncode:
    """
    Encode images for Qwen-Image with proper 16-channel VAE
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
            },
            "optional": {
                "add_noise": ("BOOLEAN", {"default": False}),
                "noise_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "DICT")
    RETURN_NAMES = ("latent", "encode_info")
    FUNCTION = "encode"
    CATEGORY = "QwenImage/VAE"
    
    def encode(self, image: torch.Tensor, vae, add_noise: bool = False, 
               noise_strength: float = 0.1) -> Tuple[Dict, Dict]:
        """
        Encode image to Qwen-Image 16-channel latent space
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        B, H, W, C = image.shape
        
        # Convert to VAE expected format (B, C, H, W)
        image = image.permute(0, 3, 1, 2)
        
        # Encode with VAE
        latent = vae.encode(image)
        
        # Add optional noise for variation
        if add_noise:
            noise = torch.randn_like(latent) * noise_strength
            latent = latent + noise
        
        encode_info = {
            "input_shape": [B, H, W, C],
            "latent_shape": list(latent.shape),
            "channels": latent.shape[1],  # Should be 16 for Qwen
            "noise_added": add_noise,
            "noise_strength": noise_strength if add_noise else 0.0
        }
        
        # Return in ComfyUI format
        return ({"samples": latent}, encode_info)


class QwenImageEditLatentPrepare:
    """
    Prepare edit latents for Qwen-Image-Edit workflow
    Handles the special requirements for edit mode
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edit_image": ("IMAGE",),
                "vae": ("VAE",),
                "mode": (["edit", "reference", "inpaint"], {"default": "edit"}),
            },
            "optional": {
                "mask": ("MASK",),
                "resize_to_target": ("BOOLEAN", {"default": True}),
                "target_width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32}),
                "target_height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "IMAGE", "DICT")
    RETURN_NAMES = ("edit_latent", "processed_image", "process_info")
    FUNCTION = "prepare"
    CATEGORY = "QwenImage/Edit"
    
    def prepare(self, edit_image: torch.Tensor, vae, mode: str,
                mask: Optional[torch.Tensor] = None,
                resize_to_target: bool = True,
                target_width: int = 1024, target_height: int = 1024) -> Tuple[Dict, torch.Tensor, Dict]:
        """
        Prepare edit latents with special handling for Qwen-Image-Edit
        """
        # Ensure batch dimension
        if len(edit_image.shape) == 3:
            edit_image = edit_image.unsqueeze(0)
        
        B, H, W, C = edit_image.shape
        process_info = {
            "original_size": (W, H),
            "mode": mode
        }
        
        # Resize if requested
        if resize_to_target and (W != target_width or H != target_height):
            # Convert to PIL for high-quality resize
            img_np = (edit_image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_img = pil_img.resize((target_width, target_height), Image.LANCZOS)
            edit_image = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0).unsqueeze(0)
            process_info["resized"] = True
            process_info["new_size"] = (target_width, target_height)
        else:
            process_info["resized"] = False
        
        # Convert to VAE format
        image_for_vae = edit_image.permute(0, 3, 1, 2)
        
        # Encode to latent
        latent = vae.encode(image_for_vae)
        
        # Handle different modes
        if mode == "inpaint" and mask is not None:
            # Add mask as additional channel for inpaint mode
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif len(mask.shape) == 3:
                mask = mask.unsqueeze(1)
            
            # Resize mask to match latent size
            latent_h, latent_w = latent.shape[-2:]
            mask = torch.nn.functional.interpolate(mask, size=(latent_h, latent_w), mode='nearest')
            
            # Concatenate mask to latent
            latent = torch.cat([latent, mask], dim=1)
            process_info["mask_added"] = True
            process_info["channels"] = latent.shape[1]
        else:
            process_info["mask_added"] = False
            process_info["channels"] = 16
        
        process_info["latent_shape"] = list(latent.shape)
        
        return ({"samples": latent}, edit_image, process_info)


class QwenToNativeLatentBridge:
    """
    Updated bridge that ensures compatibility with native ComfyUI samplers
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "target_format": (["comfyui_native", "wan_video", "stable_diffusion"], {"default": "comfyui_native"}),
            },
            "optional": {
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 1024}),
                "interpolation_mode": (["none", "repeat", "linear", "noise"], {"default": "none"}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "bridge"
    CATEGORY = "QwenImage/Bridge"
    
    def bridge(self, qwen_latent: Dict, target_format: str, 
               num_frames: int = 1, interpolation_mode: str = "none") -> Tuple[Dict]:
        """
        Bridge Qwen latents to various formats for compatibility
        """
        latent = qwen_latent["samples"]
        
        if target_format == "comfyui_native":
            # Ensure 4D tensor (B, C, H, W) for standard ComfyUI
            if len(latent.shape) == 5:
                # Take first frame if video format
                latent = latent[:, :, 0, :, :]
            elif len(latent.shape) != 4:
                raise ValueError(f"Unexpected latent shape: {latent.shape}")
            
            output = {"samples": latent}
            
        elif target_format == "wan_video":
            # Convert to WAN video format (B, C, T, H, W)
            if len(latent.shape) == 4:
                B, C, H, W = latent.shape
                temporal_frames = ((num_frames - 1) // 4) + 1
                
                if interpolation_mode == "repeat":
                    # Repeat frame
                    output_latent = latent.unsqueeze(2).repeat(1, 1, temporal_frames, 1, 1)
                elif interpolation_mode == "linear":
                    # Linear interpolation
                    output_latent = torch.zeros(B, C, temporal_frames, H, W, 
                                               dtype=latent.dtype, device=latent.device)
                    output_latent[:, :, 0] = latent
                    output_latent[:, :, -1] = latent + torch.randn_like(latent) * 0.1
                    for t in range(1, temporal_frames - 1):
                        alpha = t / (temporal_frames - 1)
                        output_latent[:, :, t] = (1 - alpha) * output_latent[:, :, 0] + alpha * output_latent[:, :, -1]
                elif interpolation_mode == "noise":
                    # Add noise variation
                    output_latent = torch.zeros(B, C, temporal_frames, H, W, 
                                               dtype=latent.dtype, device=latent.device)
                    output_latent[:, :, 0] = latent
                    for t in range(1, temporal_frames):
                        output_latent[:, :, t] = latent + torch.randn_like(latent) * 0.05 * t
                else:
                    # Single frame
                    output_latent = latent.unsqueeze(2)
                
                output = {"samples": output_latent}
            else:
                output = qwen_latent
                
        elif target_format == "stable_diffusion":
            # Convert from 16 to 4 channels for SD (simple projection)
            if latent.shape[1] == 16:
                # Take first 4 channels or apply learned projection
                latent = latent[:, :4, :, :]
            output = {"samples": latent}
        
        else:
            output = qwen_latent
        
        return (output,)


NODE_CLASS_MAPPINGS = {
    "QwenImageResolutionHelper": QwenImageResolutionHelper,
    "QwenImageVAEEncode": QwenImageVAEEncode,
    "QwenImageEditLatentPrepare": QwenImageEditLatentPrepare,
    "QwenToNativeLatentBridge": QwenToNativeLatentBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageResolutionHelper": "Qwen Image Resolution Helper",
    "QwenImageVAEEncode": "Qwen Image VAE Encode",
    "QwenImageEditLatentPrepare": "Qwen Image Edit Latent Prepare",
    "QwenToNativeLatentBridge": "Qwen to Native Latent Bridge",
}