"""
Multi-Reference Image Handler for Qwen
Allows combining multiple images with different reference methods
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
import comfy.utils
import logging

logger = logging.getLogger(__name__)

class QwenMultiReferenceHandler:
    """
    Combines multiple images into a reference package for QwenVLTextEncoder.
    Supports different combination methods: index, offset, concat.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_method": (["index", "offset", "concat", "grid"], {
                    "default": "index",
                    "tooltip": "index=sequential | offset=averaged | concat=side-by-side | grid=2x2 layout"
                }),
            },
            "optional": {
                "image1": ("IMAGE", {"tooltip": "Primary reference image"}),
                "image2": ("IMAGE", {"tooltip": "Secondary reference (e.g., style)"}),
                "image3": ("IMAGE", {"tooltip": "Third reference (optional)"}),
                "image4": ("IMAGE", {"tooltip": "Fourth reference (optional)"}),
                "weights": ("STRING", {
                    "default": "1.0,1.0,1.0,1.0",
                    "tooltip": "Weights for each image (comma-separated, used in offset mode)"
                }),
                "resize_mode": (["keep_proportion", "stretch", "resize", "pad", "pad_edge", "crop"], {
                    "default": "keep_proportion",
                    "tooltip": "keep_proportion=resize keeping aspect | stretch=force fit | resize=match avg | pad=black borders | pad_edge=repeat edge | crop=center"
                }),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {
                    "default": "nearest-exact",
                    "tooltip": "Interpolation method for resizing"
                }),
            }
        }
    
    RETURN_TYPES = ("QWEN_MULTI_REF", "IMAGE")
    RETURN_NAMES = ("multi_reference", "preview")
    FUNCTION = "combine_references"
    CATEGORY = "QwenImage/Reference"
    TITLE = "Multi-Reference Handler"
    DESCRIPTION = "Combine multiple images for advanced reference control"
    
    def combine_references(self, reference_method: str, 
                          image1: Optional[torch.Tensor] = None,
                          image2: Optional[torch.Tensor] = None, 
                          image3: Optional[torch.Tensor] = None,
                          image4: Optional[torch.Tensor] = None,
                          weights: str = "1.0,1.0,1.0,1.0",
                          resize_mode: str = "keep_proportion",
                          upscale_method: str = "nearest-exact") -> Tuple[Dict, torch.Tensor]:
        
        # Collect non-None images
        images = []
        for img in [image1, image2, image3, image4]:
            if img is not None:
                images.append(img)
        
        if not images:
            raise ValueError("At least one image must be provided")
        
        # Parse weights
        weight_values = [float(w.strip()) for w in weights.split(",")][:len(images)]
        while len(weight_values) < len(images):
            weight_values.append(1.0)
        
        # Normalize weights
        total_weight = sum(weight_values)
        weight_values = [w / total_weight for w in weight_values]
        
        # Handle resizing based on mode
        if len(images) > 1:
            if resize_mode == "keep_proportion":
                # Resize all images to same dimensions while keeping aspect ratio
                # Use first image as target
                target_h, target_w = images[0].shape[1:3]
                resized_images = [images[0]]
                
                for img in images[1:]:
                    h, w = img.shape[1:3]
                    # Calculate scale to fit
                    scale = min(target_w / w, target_h / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    # Resize
                    resized = comfy.utils.common_upscale(
                        img.movedim(-1, 1),
                        new_w, new_h, upscale_method, "disabled"
                    ).movedim(1, -1)
                    
                    # Pad to match target
                    pad_top = (target_h - new_h) // 2
                    pad_bottom = target_h - new_h - pad_top
                    pad_left = (target_w - new_w) // 2
                    pad_right = target_w - new_w - pad_left
                    
                    padded = torch.nn.functional.pad(
                        resized,
                        (0, 0, pad_left, pad_right, pad_top, pad_bottom),
                        mode='constant',
                        value=0
                    )
                    resized_images.append(padded)
                images = resized_images
                
            elif resize_mode == "stretch":
                # Force all to match first image dimensions (ignore aspect ratio)
                target_h, target_w = images[0].shape[1:3]
                resized_images = [images[0]]
                for img in images[1:]:
                    resized = comfy.utils.common_upscale(
                        img.movedim(-1, 1),
                        target_w, target_h, upscale_method, "disabled"
                    ).movedim(1, -1)
                    resized_images.append(resized)
                images = resized_images
                
            elif resize_mode == "resize":
                # Resize all to average dimensions
                avg_h = sum(img.shape[1] for img in images) // len(images)
                avg_w = sum(img.shape[2] for img in images) // len(images)
                # Round to nearest 32 for better compatibility
                target_h = ((avg_h + 16) // 32) * 32
                target_w = ((avg_w + 16) // 32) * 32
                
                resized_images = []
                for img in images:
                    resized = comfy.utils.common_upscale(
                        img.movedim(-1, 1),
                        target_w, target_h, upscale_method, "disabled"
                    ).movedim(1, -1)
                    resized_images.append(resized)
                images = resized_images
                
            elif resize_mode == "pad":
                # Pad all images to largest dimensions with black borders
                max_h = max(img.shape[1] for img in images)
                max_w = max(img.shape[2] for img in images)
                
                padded_images = []
                for img in images:
                    h, w = img.shape[1:3]
                    pad_top = (max_h - h) // 2
                    pad_bottom = max_h - h - pad_top
                    pad_left = (max_w - w) // 2
                    pad_right = max_w - w - pad_left
                    
                    padded = torch.nn.functional.pad(
                        img, 
                        (0, 0, pad_left, pad_right, pad_top, pad_bottom),
                        mode='constant', 
                        value=0
                    )
                    padded_images.append(padded)
                images = padded_images
                
            elif resize_mode == "pad_edge":
                # Pad all images to largest dimensions by repeating edge pixels
                max_h = max(img.shape[1] for img in images)
                max_w = max(img.shape[2] for img in images)
                
                padded_images = []
                for img in images:
                    h, w = img.shape[1:3]
                    pad_top = (max_h - h) // 2
                    pad_bottom = max_h - h - pad_top
                    pad_left = (max_w - w) // 2
                    pad_right = max_w - w - pad_left
                    
                    padded = torch.nn.functional.pad(
                        img, 
                        (0, 0, pad_left, pad_right, pad_top, pad_bottom),
                        mode='replicate'
                    )
                    padded_images.append(padded)
                images = padded_images
                
            elif resize_mode == "crop":
                # Center crop all images to smallest dimensions
                min_h = min(img.shape[1] for img in images)
                min_w = min(img.shape[2] for img in images)
                
                cropped_images = []
                for img in images:
                    h, w = img.shape[1:3]
                    top = (h - min_h) // 2
                    left = (w - min_w) // 2
                    cropped = img[:, top:top+min_h, left:left+min_w, :]
                    cropped_images.append(cropped)
                images = cropped_images
        
        # Create preview based on method
        if reference_method == "index":
            # Sequential: show images side by side
            preview = torch.cat(images, dim=2)  # Concatenate horizontally
            
        elif reference_method == "offset":
            # Weighted average
            preview = torch.zeros_like(images[0])
            for img, weight in zip(images, weight_values):
                preview = preview + img * weight
                
        elif reference_method == "concat":
            # Side by side concatenation
            preview = torch.cat(images, dim=2)  # Horizontal concat
            
        elif reference_method == "grid":
            # 2x2 grid layout
            if len(images) == 1:
                preview = images[0]
            elif len(images) == 2:
                # Stack vertically
                preview = torch.cat(images, dim=1)
            else:
                # 2x2 grid
                row1 = torch.cat(images[:2], dim=2) if len(images) >= 2 else images[0]
                row2 = torch.cat(images[2:4], dim=2) if len(images) >= 4 else torch.cat([images[2], torch.zeros_like(images[2])], dim=2) if len(images) == 3 else row1
                preview = torch.cat([row1, row2], dim=1)
        
        # Package data for encoder
        multi_ref_data = {
            "images": images,
            "method": reference_method,
            "weights": weight_values,
            "count": len(images),
            "dimensions": [(img.shape[1], img.shape[2]) for img in images]
        }
        
        logger.info(f"Created multi-reference with {len(images)} images using {reference_method} method")
        
        return (multi_ref_data, preview)


# Note: QwenVLTextEncoderMultiRef removed - functionality merged into main QwenVLTextEncoder
# The main encoder now accepts optional multi_reference input directly


NODE_CLASS_MAPPINGS = {
    "QwenMultiReferenceHandler": QwenMultiReferenceHandler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenMultiReferenceHandler": "Multi-Reference Handler",
}