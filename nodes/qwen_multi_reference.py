"""
Multi-Reference Image Handler for Qwen
[DEPRECATED] Use Image Batch node instead for multiple images.
This node is kept for backward compatibility only.
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
import comfy.utils
import logging
import math

logger = logging.getLogger(__name__)
logger.warning("[DEPRECATED] QwenMultiReferenceHandler is deprecated. Use Image Batch node to combine multiple images, then connect to QwenVLTextEncoder.")

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
                "reference_method": (["concat", "grid", "offset", "native_multi"], {
                    "default": "concat",
                    "tooltip": "concat: side-by-side | grid: 2x2 layout | offset: weighted blend | native_multi: separate images for native processing"
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
                "resize_mode": (["match_first", "common_height", "common_width", "largest_dims", "qwen_smart_resize", "diffsynth_auto_resize"], {
                    "default": "qwen_smart_resize", 
                    "tooltip": "match_first: resize all to image1 size | common_height: same height, keep aspect | common_width: same width, keep aspect | largest_dims: resize all to largest dimensions | qwen_smart_resize: Official Qwen 28px method | diffsynth_auto_resize: DiffSynth 32px method"
                }),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {
                    "default": "bicubic",
                    "tooltip": "Interpolation method for resizing images."
                }),
            }
        }

    # Output can be single composite or multiple separate images
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "create_composite_canvas"
    CATEGORY = "QwenImage/Reference"
    TITLE = "[DEPRECATED] Multi-Reference Composer"
    DESCRIPTION = """[DEPRECATED] Use Image Batch node instead.
This node is kept for backward compatibility.
For new workflows, use Image Batch to combine multiple images."""

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
        
        # Round by factor helper functions
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
            
        elif resize_mode == "qwen_smart_resize":
            # Use official Qwen smart_resize for each image, then find common dimensions
            qwen_dimensions = []
            for img in images:
                h, w = img.shape[1], img.shape[2]
                qw, qh = self.qwen_smart_resize(w, h)
                qwen_dimensions.append((qw, qh))
            
            # For uniform output, use the most common dimensions or average
            if reference_method in ["grid", "native_multi"]:
                # Need uniform dimensions - use average
                avg_w = int(sum(dim[0] for dim in qwen_dimensions) / len(qwen_dimensions))
                avg_h = int(sum(dim[1] for dim in qwen_dimensions) / len(qwen_dimensions))
                # Ensure still 28-pixel aligned
                target_w = round(avg_w / 28) * 28
                target_h = round(avg_h / 28) * 28
                logger.info(f"[Multi-Reference] Qwen smart resize (uniform): {target_w}x{target_h}")
            else:
                # Individual sizing will be handled in Step 2
                logger.info(f"[Multi-Reference] Qwen smart resize (individual aspect ratios preserved)")
                
        elif resize_mode == "diffsynth_auto_resize":
            # Use DiffSynth auto-resize for each image, then find common dimensions
            diffsynth_dimensions = []
            target_pixels = 1048576  # 1024x1024 default
            for img in images:
                h, w = img.shape[1], img.shape[2]
                dw, dh = self.calculate_diffsynth_dimensions(w, h, target_pixels)
                diffsynth_dimensions.append((dw, dh))
            
            # For uniform output, use average
            if reference_method in ["grid", "native_multi"]:
                # Need uniform dimensions - use average
                avg_w = int(sum(dim[0] for dim in diffsynth_dimensions) / len(diffsynth_dimensions))
                avg_h = int(sum(dim[1] for dim in diffsynth_dimensions) / len(diffsynth_dimensions))
                # Ensure still 32-pixel aligned
                target_w = round(avg_w / 32) * 32
                target_h = round(avg_h / 32) * 32
                logger.info(f"[Multi-Reference] DiffSynth auto resize (uniform): {target_w}x{target_h}")
            else:
                # Individual sizing will be handled in Step 2
                logger.info(f"[Multi-Reference] DiffSynth auto resize (individual aspect ratios preserved)")

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
                    # For native_multi, must have uniform dimensions
                    if reference_method == "native_multi":
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
                    # For native_multi, must have uniform dimensions
                    if reference_method == "native_multi":
                        new_w, new_h = target_w, target_h
                    else:
                        # Keep aspect ratio, adjust height based on common width
                        aspect_ratio = img.shape[1] / img.shape[2]  # h/w
                        new_w = target_w
                        new_h = int(target_w * aspect_ratio)
                
            elif resize_mode == "largest_dims":
                # Stretch to largest dimensions (may distort)
                new_w, new_h = target_w, target_h
                
            elif resize_mode == "qwen_smart_resize":
                # Use official Qwen smart_resize method
                h, w = img.shape[1], img.shape[2]
                if reference_method in ["grid", "native_multi"]:
                    # Use uniform target dimensions
                    new_w, new_h = target_w, target_h
                else:
                    # Calculate optimal dimensions for each image individually
                    new_w, new_h = self.qwen_smart_resize(w, h)
                    
            elif resize_mode == "diffsynth_auto_resize":
                # Use DiffSynth auto-resize method
                h, w = img.shape[1], img.shape[2]
                if reference_method in ["grid", "native_multi"]:
                    # Use uniform target dimensions
                    new_w, new_h = target_w, target_h
                else:
                    # Calculate optimal dimensions for each image individually
                    target_pixels = 1048576  # 1024x1024 default
                    new_w, new_h = self.calculate_diffsynth_dimensions(w, h, target_pixels)

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

        elif reference_method == "native_multi":
            # Return multiple separate images for native Qwen2.5-VL processing
            logger.info(f"[Multi-Reference] Returning {len(standardized_images)} separate images for native processing.")

            # CRITICAL: For native_multi, ALL images MUST have uniform dimensions
            # Otherwise, VAE encoding will produce different sized latents
            if len(standardized_images) > 1:
                first_shape = standardized_images[0].shape
                shapes_match = all(img.shape == first_shape for img in standardized_images)

                if not shapes_match:
                    logger.warning(f"[Multi-Reference] native_multi detected non-uniform dimensions!")
                    for i, img in enumerate(standardized_images):
                        logger.warning(f"  Image {i}: {img.shape}")

                    # Force uniform dimensions using the first image's size
                    # This ensures consistent latent dimensions after VAE encoding
                    target_h, target_w = standardized_images[0].shape[1:3]
                    logger.info(f"[Multi-Reference] Forcing all images to {target_w}x{target_h} for uniform latents")

                    fixed_images = []
                    for i, img in enumerate(standardized_images):
                        if img.shape[1:3] != (target_h, target_w):
                            # Resize to match first image
                            resized_img_chw = comfy.utils.common_upscale(
                                img.movedim(-1, 1), # HWC to CHW
                                target_w, target_h, upscale_method, "disabled"
                            )
                            fixed_images.append(resized_img_chw.movedim(1, -1)) # CHW to HWC
                            logger.info(f"  Resized image {i} from {img.shape} to match target")
                        else:
                            fixed_images.append(img)

                    standardized_images = fixed_images

            # Stack all images along batch dimension
            composite_image = torch.cat(standardized_images, dim=0)
            logger.info(f"[Multi-Reference] Final output: {composite_image.shape}")

        return (composite_image,)
