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
    # From DiffSynth-Studio official implementation
    QWEN_RESOLUTIONS = [
        # Square
        (1024, 1024),
        
        # All aspect ratios from DiffSynth
        (672, 1568),
        (688, 1504), 
        (720, 1456),
        (752, 1392),
        (800, 1328),
        (832, 1248),
        (880, 1184),
        (944, 1104),
        (1104, 944),
        (1184, 880),
        (1248, 832),
        (1328, 800),
        (1392, 752),
        (1456, 720),
        (1504, 688),
        (1568, 672),
        
        # Additional larger resolutions
        (1328, 1328),
        (1920, 1080),  # 16:9
        (1080, 1920),  # 9:16
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        resize_modes = ["fit", "fill", "exact", "pad", "crop", "qwen_smart_resize", "diffsynth_auto_resize"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (resize_modes, {
                    "default": "qwen_smart_resize",
                    "tooltip": "fit: scale to fit inside | fill: scale to fill (crop) | exact: closest resolution | pad: add padding | crop: center crop | qwen_smart_resize: Official Qwen 28px method | diffsynth_auto_resize: DiffSynth 32px method"
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
    
    def qwen_smart_resize(self, width: int, height: int, min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28, factor: int = 28) -> Tuple[int, int]:
        """
        Official Qwen smart_resize: maintains aspect ratio with 28-pixel alignment
        Based on qwen_vl_utils.process_vision_info implementation
        """
        # Constants from official implementation
        MAX_RATIO = 200
        
        # Check aspect ratio constraint
        if max(height, width) / min(height, width) > MAX_RATIO:
            raise ValueError(f"Aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}")
        
        # Round by factor helper
        def round_by_factor(number: int, factor: int) -> int:
            return round(number / factor) * factor
        
        def floor_by_factor(number: int, factor: int) -> int:
            return math.floor(number / factor) * factor
            
        def ceil_by_factor(number: int, factor: int) -> int:
            return math.ceil(number / factor) * factor
        
        # Initial alignment to factor
        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        
        # Adjust if exceeds max pixels
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = max(factor, floor_by_factor(height / beta, factor))
            w_bar = max(factor, floor_by_factor(width / beta, factor))
        # Adjust if below min pixels
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)
            
        return w_bar, h_bar  # Return width, height
    
    def calculate_diffsynth_dimensions(self, width: int, height: int, target_pixels: int = 1048576) -> Tuple[int, int]:
        """
        DiffSynth-style auto-resize: maintains aspect ratio with 32-pixel alignment
        Uses target pixel area (default 1024x1024 = 1048576 pixels)
        """
        aspect_ratio = width / height
        
        # Calculate dimensions to fit within target pixel area
        optimal_width = math.sqrt(target_pixels * aspect_ratio)
        optimal_height = optimal_width / aspect_ratio
        
        # Align to 32-pixel boundaries (VAE requirement)
        aligned_width = round(optimal_width / 32) * 32
        aligned_height = round(optimal_height / 32) * 32
        
        return aligned_width, aligned_height
    
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
        - qwen_smart_resize: Official Qwen method - 28px alignment with smart pixel constraints
        - diffsynth_auto_resize: DiffSynth method - 32px alignment with target pixel area
        """
        
        # Get current dimensions
        batch, height, width, channels = image.shape
        
        # Parse custom resolution if provided
        target_resolution = self.parse_custom_resolution(custom_resolution)
        
        # If no custom resolution, find optimal based on mode
        if not target_resolution:
            if mode == "qwen_smart_resize":
                # Use official Qwen smart_resize with 28px alignment
                min_pixels = 4 * 28 * 28  # 3136 pixels minimum
                max_pixels = target_pixels or (16384 * 28 * 28)  # Use target_pixels or default max
                target_resolution = self.qwen_smart_resize(width, height, min_pixels, max_pixels, factor=28)
            elif mode == "diffsynth_auto_resize":
                # Use DiffSynth-style calculation with 32px alignment
                target_resolution = self.calculate_diffsynth_dimensions(width, height, target_pixels or 1048576)
            else:
                # Use existing preset-based selection
                target_resolution = self.find_closest_resolution(width, height, target_pixels)
        
        target_w, target_h = target_resolution
        
        # Calculate scaling based on mode
        if mode == "fit":
            # Scale to fit inside, maintaining aspect ratio
            scale = min(target_w / width, target_h / height)
            new_width = round(width * scale)
            new_height = round(height * scale)
            
        elif mode in ["qwen_smart_resize", "diffsynth_auto_resize"]:
            # Direct resize to calculated optimal dimensions
            new_width = target_w
            new_height = target_h
            
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
        elif mode == "qwen_smart_resize":
            actual_pixels = new_width * new_height
            alignment = "28px"
            info += f" | Official Qwen method ({alignment} alignment, {actual_pixels} pixels)"
        elif mode == "diffsynth_auto_resize":
            actual_pixels = new_width * new_height
            target_pixels_val = target_pixels or 1048576
            alignment = "32px"
            info += f" | DiffSynth method ({alignment} alignment, target: {target_pixels_val}, actual: {actual_pixels})"
        else:
            info += f" | Auto-selected from Qwen resolutions"
        
        return (image_out, new_width, new_height, info)


class QwenResolutionSelector:
    """
    Comprehensive resolution selector for Qwen Empty Latent generation.
    Provides curated list of Qwen-optimized resolutions including:
    - Square resolutions (512x512 to 1328x1328)
    - Modern aspect ratios (16:9, 2:1, 9:16, 1:2)
    - DiffSynth-Studio original resolutions
    - Low VRAM options
    """
    
    RESOLUTIONS = {
        # Format: "Display Name": (width, height)
        # Square resolutions
        "Square - 512x512": (512, 512),
        "Square - 768x768": (768, 768),
        "Square - 1024x1024": (1024, 1024),
        "Square - 1328x1328": (1328, 1328),
        
        # Common modern aspect ratios
        "Landscape - 1920x1080 (16:9)": (1920, 1080),
        "Landscape - 1584x1056": (1584, 1056),
        "Landscape - 1456x720": (1456, 720),
        "Landscape - 1328x800": (1328, 800),
        "Landscape - 2048x1024 (2:1)": (2048, 1024),
        "Landscape - 1344x768": (1344, 768),
        "Landscape - 1536x640": (1536, 640),
        "Landscape - 1024x768": (1024, 768),
        "Landscape - 1024x512": (1024, 512),
        
        "Portrait - 1080x1920 (9:16)": (1080, 1920),
        "Portrait - 1056x1584": (1056, 1584),
        "Portrait - 720x1456": (720, 1456),
        "Portrait - 800x1328": (800, 1328),
        "Portrait - 1024x2048 (1:2)": (1024, 2048),
        "Portrait - 768x1344": (768, 1344),
        "Portrait - 640x1536": (640, 1536),
        "Portrait - 768x1024": (768, 1024),
        "Portrait - 512x1024": (512, 1024),
        
        # DiffSynth-Studio original resolutions
        "DiffSynth - 672x1568": (672, 1568),
        "DiffSynth - 688x1504": (688, 1504),
        "DiffSynth - 752x1392": (752, 1392),
        "DiffSynth - 832x1248": (832, 1248),
        "DiffSynth - 880x1184": (880, 1184),
        "DiffSynth - 944x1104": (944, 1104),
        "DiffSynth - 1104x944": (1104, 944),
        "DiffSynth - 1184x880": (1184, 880),
        "DiffSynth - 1248x832": (1248, 832),
        "DiffSynth - 1392x752": (1392, 752),
        "DiffSynth - 1504x688": (1504, 688),
        "DiffSynth - 1568x672": (1568, 672),
        
        # Smaller resolutions for low VRAM
        "Low VRAM - 512x768": (512, 768),
        "Low VRAM - 768x512": (768, 512),
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
    DESCRIPTION = "Select from comprehensive list of Qwen-optimized resolutions including modern aspect ratios, DiffSynth presets, and low VRAM options"
    
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
        aspect_ratio = width / height
        
        # Determine aspect ratio category
        if abs(aspect_ratio - 1.0) < 0.05:
            aspect_desc = "Square"
        elif abs(aspect_ratio - 16/9) < 0.05:
            aspect_desc = "16:9"
        elif abs(aspect_ratio - 9/16) < 0.05:
            aspect_desc = "9:16"
        elif abs(aspect_ratio - 2.0) < 0.05:
            aspect_desc = "2:1"
        elif abs(aspect_ratio - 0.5) < 0.05:
            aspect_desc = "1:2"
        elif aspect_ratio > 1.0:
            aspect_desc = f"{aspect_ratio:.2f}:1"
        else:
            aspect_desc = f"1:{1/aspect_ratio:.2f}"
            
        info += f" | {aspect_desc} | {megapixels:.2f}MP"
        
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