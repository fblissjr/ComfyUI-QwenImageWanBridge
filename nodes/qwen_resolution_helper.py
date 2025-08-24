"""
Qwen Optimal Resolution Helper
Automatically selects and resizes images to Qwen-preferred resolutions
"""

import torch
import math
from typing import Tuple, List, Optional
import comfy.utils

class QwenOptimalResolution:
    """
    Resize images to the closest Qwen-preferred resolution without stretching.
    Maintains aspect ratio and uses padding/cropping as needed.
    """
    
    # Qwen-preferred resolutions (width, height)
    # Based on DiffSynth-Studio and official Qwen documentation
    QWEN_RESOLUTIONS = [
        # Square resolutions
        (1024, 1024),
        (1328, 1328),
        
        # Landscape resolutions
        (1328, 800),
        (1456, 720),
        (1584, 1056),
        (1920, 1080),  # 16:9
        (2048, 1024),  # 2:1
        
        # Portrait resolutions  
        (800, 1328),
        (720, 1456),
        (1056, 1584),
        (1080, 1920),  # 9:16
        (1024, 2048),  # 1:2
        
        # Additional common resolutions
        (1344, 768),   # 7:4
        (768, 1344),   # 4:7
        (1536, 640),   # 12:5
        (640, 1536),   # 5:12
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        resize_modes = ["fit", "fill", "exact", "pad", "crop"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (resize_modes, {
                    "default": "fit",
                    "tooltip": "fit: scale to fit inside | fill: scale to fill (crop) | exact: closest resolution | pad: add padding | crop: center crop"
                }),
            },
            "optional": {
                "target_pixels": ("INT", {
                    "default": 1048576,  # 1024x1024
                    "min": 262144,       # 512x512
                    "max": 4194304,      # 2048x2048
                    "step": 65536,
                    "tooltip": "Target total pixel count (width * height)"
                }),
                "custom_resolution": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Use custom resolution like '1024x1024' instead of auto-selection"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "resolution_info")
    FUNCTION = "resize_to_optimal"
    CATEGORY = "QwenImage/Utils"
    TITLE = "Qwen Optimal Resolution"
    DESCRIPTION = "Automatically resize images to Qwen-preferred resolutions"
    
    def find_closest_resolution(self, width: int, height: int, target_pixels: Optional[int] = None) -> Tuple[int, int]:
        """Find the closest Qwen resolution based on aspect ratio and target pixels"""
        
        aspect_ratio = width / height
        
        # Filter resolutions by target pixel count if specified
        resolutions = self.QWEN_RESOLUTIONS
        if target_pixels:
            # Allow 20% variance in pixel count
            min_pixels = target_pixels * 0.8
            max_pixels = target_pixels * 1.2
            resolutions = [(w, h) for w, h in resolutions 
                          if min_pixels <= w * h <= max_pixels]
            
            # If no resolutions in range, use all
            if not resolutions:
                resolutions = self.QWEN_RESOLUTIONS
        
        # Find resolution with closest aspect ratio
        best_resolution = None
        best_ratio_diff = float('inf')
        
        for res_w, res_h in resolutions:
            res_ratio = res_w / res_h
            ratio_diff = abs(math.log(res_ratio / aspect_ratio))
            
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_resolution = (res_w, res_h)
        
        return best_resolution
    
    def parse_custom_resolution(self, custom_str: str) -> Optional[Tuple[int, int]]:
        """Parse custom resolution string like '1024x768' or '1024,768'"""
        if not custom_str.strip():
            return None
            
        try:
            # Try different separators
            if 'x' in custom_str:
                parts = custom_str.split('x')
            elif ',' in custom_str:
                parts = custom_str.split(',')
            elif ' ' in custom_str:
                parts = custom_str.split()
            else:
                return None
                
            if len(parts) == 2:
                width = int(parts[0].strip())
                height = int(parts[1].strip())
                return (width, height)
        except:
            pass
            
        return None
    
    def resize_to_optimal(self, image: torch.Tensor, mode: str = "fit", 
                         target_pixels: Optional[int] = None,
                         custom_resolution: str = "") -> Tuple[torch.Tensor, int, int, str]:
        """
        Resize image to optimal Qwen resolution
        
        Modes:
        - fit: Scale to fit inside resolution (may have padding)
        - fill: Scale to fill resolution (may crop)
        - exact: Use exact closest resolution (may distort)
        - pad: Keep image size, add padding to reach resolution
        - crop: Keep image size, crop to reach resolution
        """
        
        # Get current dimensions
        batch, height, width, channels = image.shape
        
        # Parse custom resolution if provided
        target_resolution = self.parse_custom_resolution(custom_resolution)
        
        # If no custom resolution, find optimal
        if not target_resolution:
            target_resolution = self.find_closest_resolution(width, height, target_pixels)
        
        target_w, target_h = target_resolution
        
        # Calculate scaling based on mode
        if mode == "fit":
            # Scale to fit inside, maintaining aspect ratio
            scale = min(target_w / width, target_h / height)
            new_width = round(width * scale)
            new_height = round(height * scale)
            
        elif mode == "fill":
            # Scale to fill, maintaining aspect ratio (will crop)
            scale = max(target_w / width, target_h / height)
            new_width = round(width * scale)
            new_height = round(height * scale)
            
        elif mode == "exact":
            # Stretch to exact resolution (may distort)
            new_width = target_w
            new_height = target_h
            
        elif mode == "pad":
            # Keep original size if smaller, just pad
            new_width = min(width, target_w)
            new_height = min(height, target_h)
            
        elif mode == "crop":
            # Keep original size if larger, just crop
            new_width = width
            new_height = height
        
        # Move channels to correct position for resize
        samples = image.movedim(-1, 1)  # [B, H, W, C] -> [B, C, H, W]
        
        # Resize if dimensions changed
        if new_width != width or new_height != height:
            samples = comfy.utils.common_upscale(
                samples, new_width, new_height, 
                "bilinear", "disabled"
            )
        
        # Now handle padding or cropping to reach exact target
        if mode in ["fit", "pad"]:
            # Add padding if needed
            if new_width < target_w or new_height < target_h:
                pad_left = (target_w - new_width) // 2
                pad_right = target_w - new_width - pad_left
                pad_top = (target_h - new_height) // 2
                pad_bottom = target_h - new_height - pad_top
                
                # Pad with zeros (black)
                samples = torch.nn.functional.pad(
                    samples, 
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='constant', 
                    value=0
                )
                new_width = target_w
                new_height = target_h
                
        elif mode in ["fill", "crop"]:
            # Crop if needed
            if new_width > target_w or new_height > target_h:
                crop_left = (new_width - target_w) // 2
                crop_top = (new_height - target_h) // 2
                
                samples = samples[
                    :, :,
                    crop_top:crop_top + target_h,
                    crop_left:crop_left + target_w
                ]
                new_width = target_w
                new_height = target_h
        
        # Move channels back
        image_out = samples.movedim(1, -1)  # [B, C, H, W] -> [B, H, W, C]
        
        # Create info string
        info = f"Resized from {width}x{height} to {new_width}x{new_height} (mode: {mode})"
        if custom_resolution:
            info += f" | Custom: {custom_resolution}"
        else:
            info += f" | Auto-selected from Qwen resolutions"
        
        return (image_out, new_width, new_height, info)


class QwenResolutionSelector:
    """
    Simple resolution selector for Qwen Empty Latent generation.
    Provides dropdown of recommended resolutions.
    """
    
    RESOLUTIONS = {
        # Format: "Display Name": (width, height)
        "Square - 1024x1024": (1024, 1024),
        "Square - 1328x1328": (1328, 1328),
        
        "Landscape - 1328x800": (1328, 800),
        "Landscape - 1456x720": (1456, 720),
        "Landscape - 1584x1056": (1584, 1056),
        "Landscape - 1920x1080 (16:9)": (1920, 1080),
        "Landscape - 2048x1024 (2:1)": (2048, 1024),
        "Landscape - 1344x768": (1344, 768),
        "Landscape - 1536x640": (1536, 640),
        
        "Portrait - 800x1328": (800, 1328),
        "Portrait - 720x1456": (720, 1456),
        "Portrait - 1056x1584": (1056, 1584),
        "Portrait - 1080x1920 (9:16)": (1080, 1920),
        "Portrait - 1024x2048 (1:2)": (1024, 2048),
        "Portrait - 768x1344": (768, 1344),
        "Portrait - 640x1536": (640, 1536),
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        resolution_names = list(cls.RESOLUTIONS.keys())
        
        return {
            "required": {
                "resolution": (resolution_names, {
                    "default": "Square - 1024x1024"
                }),
            },
            "optional": {
                "custom_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "Override with custom width (0 to use preset)"
                }),
                "custom_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "tooltip": "Override with custom height (0 to use preset)"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "resolution_info")
    FUNCTION = "get_resolution"
    CATEGORY = "QwenImage/Utils"
    TITLE = "Qwen Resolution Selector"
    DESCRIPTION = "Select from Qwen-recommended resolutions"
    
    def get_resolution(self, resolution: str, custom_width: int = 0, custom_height: int = 0) -> Tuple[int, int, str]:
        """Get selected resolution or custom override"""
        
        if custom_width > 0 and custom_height > 0:
            # Use custom resolution
            width = custom_width
            height = custom_height
            info = f"Custom: {width}x{height}"
        else:
            # Use preset
            width, height = self.RESOLUTIONS[resolution]
            info = f"Preset: {resolution} ({width}x{height})"
        
        total_pixels = width * height
        megapixels = total_pixels / 1_000_000
        info += f" | {megapixels:.2f}MP"
        
        return (width, height, info)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenOptimalResolution": QwenOptimalResolution,
    "QwenResolutionSelector": QwenResolutionSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenOptimalResolution": "Qwen Optimal Resolution",
    "QwenResolutionSelector": "Qwen Resolution Selector",
}