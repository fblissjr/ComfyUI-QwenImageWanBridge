"""
Multi-Reference Image Handler for Qwen
Combines multiple images into a single high-resolution composite canvas.
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
import comfy.utils
import logging

logger = logging.getLogger(__name__)

class QwenMultiReferenceHandler:
    """
    Combines multiple images into a single composite canvas for spatial reference.
    This node outputs a single IMAGE tensor that can be fed into the QwenVLTextEncoder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # The 'index' mode is removed as it's not supported by the model.
        # 'offset' is kept as it produces a single, blended image.
        # 'concat' and 'grid' are the primary spatial composition methods.
        return {
            "required": {
                "reference_method": (["concat", "grid", "offset"], {
                    "default": "concat",
                    "tooltip": "concat: side-by-side | grid: 2x2 layout | offset: weighted blend"
                }),
            },
            "optional": {
                "image1": ("IMAGE", {"tooltip": "Primary image. All other images will be resized to match this one's dimensions."}),
                "image2": ("IMAGE", {"tooltip": "Second image."}),
                "image3": ("IMAGE", {"tooltip": "Third image (for grid mode)."}),
                "image4": ("IMAGE", {"tooltip": "Fourth image (for grid mode)."}),
                "weights": ("STRING", {
                    "default": "1.0,1.0,1.0,1.0",
                    "tooltip": "Weights for each image (comma-separated, used in 'offset' blend mode)"
                }),
                "resize_mode": (["match_first", "common_height", "common_width", "largest_dims"], {
                    "default": "common_height", 
                    "tooltip": "match_first: resize all to image1 size | common_height: same height, keep aspect (uniform in grid) | common_width: same width, keep aspect (uniform in grid) | largest_dims: resize all to largest dimensions"
                }),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {
                    "default": "bicubic",
                    "tooltip": "Interpolation method for resizing images."
                }),
            }
        }

    # The output is now a single IMAGE tensor, making it directly usable.
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite_image",)
    FUNCTION = "create_composite_canvas"
    CATEGORY = "QwenImage/Reference"
    TITLE = "Multi-Reference Canvas Composer"
    DESCRIPTION = """Combines multiple images into a single high-resolution canvas for spatial reference.

Resize modes:
• match_first: resize all to image1 dimensions (may distort)
• common_height: same height, preserve aspect ratios (uniform dimensions in grid mode)  
• common_width: same width, preserve aspect ratios (uniform dimensions in grid mode)
• largest_dims: resize all to largest width/height found

Composition methods:
• 'concat' creates a side-by-side image
• 'grid' creates a 2x2 image layout (requires uniform dimensions)
• 'offset' creates a weighted blend

The output is a single image tensor, ready for the QwenVLTextEncoder."""

    def create_composite_canvas(self, reference_method: str,
                                image1: Optional[torch.Tensor] = None,
                                image2: Optional[torch.Tensor] = None,
                                image3: Optional[torch.Tensor] = None,
                                image4: Optional[torch.Tensor] = None,
                                weights: str = "1.0,1.0,1.0,1.0",
                                resize_mode: str = "common_height",
                                upscale_method: str = "bicubic") -> Tuple[torch.Tensor]:

        # Collect all provided, non-None images
        images = [img for img in [image1, image2, image3, image4] if img is not None]

        if not images:
            raise ValueError("At least one image must be provided to the Multi-Reference Handler.")

        # Step 1: Calculate target dimensions based on resize mode
        if resize_mode == "match_first":
            # Original behavior: resize all to match first image
            target_h, target_w = images[0].shape[1:3]
            logger.info(f"[Multi-Reference] Resizing all images to match first image: {target_w}x{target_h}")
            
        elif resize_mode == "common_height":
            # Find common height (use minimum to avoid upscaling)
            target_h = min(img.shape[1] for img in images)
            # For grid mode, we need uniform dimensions to avoid tensor mismatch
            if reference_method == "grid":
                # Calculate average width to maintain reasonable proportions
                avg_aspect = sum(img.shape[2] / img.shape[1] for img in images) / len(images)
                target_w = int(target_h * avg_aspect)
                logger.info(f"[Multi-Reference] Grid mode: using common dimensions {target_w}x{target_h}")
            else:
                logger.info(f"[Multi-Reference] Using common height: {target_h}, aspect ratios preserved")
            
        elif resize_mode == "common_width": 
            # Find common width (use minimum to avoid upscaling)
            target_w = min(img.shape[2] for img in images)
            # For grid mode, we need uniform dimensions to avoid tensor mismatch
            if reference_method == "grid":
                # Calculate average height to maintain reasonable proportions
                avg_aspect = sum(img.shape[1] / img.shape[2] for img in images) / len(images)
                target_h = int(target_w * avg_aspect)
                logger.info(f"[Multi-Reference] Grid mode: using common dimensions {target_w}x{target_h}")
            else:
                logger.info(f"[Multi-Reference] Using common width: {target_w}, aspect ratios preserved")
            
        elif resize_mode == "largest_dims":
            # Use largest width and height found across all images
            target_h = max(img.shape[1] for img in images)
            target_w = max(img.shape[2] for img in images)
            logger.info(f"[Multi-Reference] Using largest dimensions: {target_w}x{target_h}")

        # Step 2: Resize all images according to the chosen mode
        standardized_images = []
        for i, img in enumerate(images):
            if resize_mode == "match_first":
                # Stretch to exact target dimensions
                new_w, new_h = target_w, target_h
                
            elif resize_mode == "common_height":
                if reference_method == "grid":
                    # Grid mode requires uniform dimensions
                    new_w, new_h = target_w, target_h
                else:
                    # Keep aspect ratio, adjust width based on common height
                    aspect_ratio = img.shape[2] / img.shape[1]  # w/h
                    new_h = target_h
                    new_w = int(target_h * aspect_ratio)
                
            elif resize_mode == "common_width":
                if reference_method == "grid":
                    # Grid mode requires uniform dimensions
                    new_w, new_h = target_w, target_h
                else:
                    # Keep aspect ratio, adjust height based on common width  
                    aspect_ratio = img.shape[1] / img.shape[2]  # h/w
                    new_w = target_w
                    new_h = int(target_w * aspect_ratio)
                
            elif resize_mode == "largest_dims":
                # Stretch to largest dimensions (may distort)
                new_w, new_h = target_w, target_h

            # Resize the image
            resized_img_chw = comfy.utils.common_upscale(
                img.movedim(-1, 1), # HWC to CHW for upscale function
                new_w, new_h, upscale_method, "disabled"
            )
            standardized_images.append(resized_img_chw.movedim(1, -1)) # CHW to HWC

        # Step 3: Create the composite canvas based on the chosen method
        if reference_method == "offset":
            # Weighted average blend. The result is a single image of target_w x target_h.
            logger.info(f"[Multi-Reference] Creating composite using 'offset' (blend) method.")
            weight_values = [float(w.strip()) for w in weights.split(",")][:len(standardized_images)]
            total_weight = sum(weight_values) if sum(weight_values) > 0 else 1.0
            weight_values = [w / total_weight for w in weight_values]

            composite_image = torch.zeros_like(standardized_images[0])
            for img, weight in zip(standardized_images, weight_values):
                composite_image += img * weight

        elif reference_method == "concat":
            # Side-by-side horizontal concatenation.
            logger.info(f"[Multi-Reference] Creating composite using 'concat' (horizontal) method.")
            composite_image = torch.cat(standardized_images, dim=2)  # Concatenate along the width dimension
            final_w, final_h = composite_image.shape[2], composite_image.shape[1]
            logger.info(f"[Multi-Reference] Final canvas size: {final_w}x{final_h}.")

        elif reference_method == "grid":
            # 2x2 grid layout.
            logger.info(f"[Multi-Reference] Creating composite using 'grid' (2x2) method.")

            # Pad the list with black images to ensure we have exactly 4 for a 2x2 grid
            num_images = len(standardized_images)
            if num_images < 4:
                black_image = torch.zeros_like(standardized_images[0])
                standardized_images.extend([black_image] * (4 - num_images))

            # Create the two rows
            row1 = torch.cat(standardized_images[0:2], dim=2) # Concat images 0 and 1 horizontally
            row2 = torch.cat(standardized_images[2:4], dim=2) # Concat images 2 and 3 horizontally

            # Stack the rows vertically
            composite_image = torch.cat([row1, row2], dim=1) # Concatenate along the height dimension
            final_w, final_h = composite_image.shape[2], composite_image.shape[1]
            logger.info(f"[Multi-Reference] Final canvas size: {final_w}x{final_h}.")

        return (composite_image,)
