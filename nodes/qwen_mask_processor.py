"""
QwenMaskProcessor

Processes spatial editor output into inpainting-ready masks.
Follows exact patterns from QwenImageEditInpaintPipeline.
Reuses all existing image processing methods from the codebase.
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
import json
import base64
import io
from typing import Tuple, Optional
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class QwenMaskProcessor:
    """
    Processes spatial editor output into inpainting-ready masks.
    Follows exact patterns from QwenImageEditInpaintPipeline.
    """
    
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image for editing"}),
                "mask_data": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Base64 mask from spatial editor"
                }),
                "inpaint_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "placeholder": "Editing instructions from spatial regions"
                }),
                "mask_blur": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Gaussian blur for mask edges"
                }),
                "mask_expand": ("INT", {
                    "default": 0, "min": -50, "max": 50,
                    "tooltip": "Expand (+) or contract (-) mask pixels"
                }),
                "mask_feather": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply edge feathering for smoother blending"
                })
            },
            "optional": {
                "mask_override": ("MASK", {"tooltip": "Override generated mask"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "IMAGE", "IMAGE") 
    RETURN_NAMES = ("image", "mask", "prompt", "preview", "mask_preview")
    FUNCTION = "process_mask"
    CATEGORY = "Qwen/Mask"
    OUTPUT_NODE = True
    
    def process_mask(self, image: torch.Tensor, mask_data: str, inpaint_prompt: str, 
                    mask_blur: float, mask_expand: int, mask_feather: bool,
                    mask_override: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        """
        Process mask following QwenImageEditInpaintPipeline patterns
        
        Args:
            image: Input image tensor
            mask_data: Base64 encoded mask from JavaScript
            inpaint_prompt: Inpainting instructions
            mask_blur: Gaussian blur amount
            mask_expand: Pixel expansion/contraction
            mask_feather: Whether to apply edge feathering
            mask_override: Optional mask override
            
        Returns:
            (image, mask_tensor, prompt, preview_image)
        """
        
        logger.info("=== QWEN MASK PROCESSOR START ===")
        logger.info(f"Inputs - mask_data length: {len(mask_data)}")
        logger.info(f"Inputs - inpaint_prompt: {inpaint_prompt}")
        logger.info(f"Inputs - blur: {mask_blur}, expand: {mask_expand}, feather: {mask_feather}")
        logger.info(f"Inputs - mask_override provided: {mask_override is not None}")
        
        # Convert image to PIL (reuse existing pattern)
        pil_image = self._tensor_to_pil(image)
        img_width, img_height = pil_image.size
        logger.info(f"Image dimensions: {img_width}x{img_height}")
        
        if mask_override is not None:
            logger.info("Using provided mask override")
            # Use provided mask
            mask_tensor = mask_override
            mask_pil = self._mask_tensor_to_pil(mask_tensor, img_width, img_height)
        elif mask_data:
            logger.info("Processing spatial editor mask data")
            # Process spatial editor mask
            mask_pil = self._base64_to_pil(mask_data)
            mask_pil = self._process_mask_with_options(
                mask_pil, mask_blur, mask_expand, mask_feather, img_width, img_height
            )
            mask_tensor = self._pil_to_mask_tensor(mask_pil)
        else:
            logger.info("No mask data provided - creating empty mask")
            # Empty mask - no inpainting
            mask_pil = Image.new('L', (img_width, img_height), 0)
            mask_tensor = torch.zeros((1, img_height, img_width))
        
        # Create preview showing inpainting areas
        preview_pil = self._create_inpaint_preview(pil_image, mask_pil)
        preview_tensor = self._pil_to_tensor(preview_pil)
        
        # Create mask preview (black/white visualization)
        mask_preview_tensor = self._create_mask_preview(mask_pil)
        
        logger.info(f"Final mask tensor shape: {mask_tensor.shape}")
        logger.info(f"Final preview tensor shape: {preview_tensor.shape}")
        logger.info(f"Final mask preview tensor shape: {mask_preview_tensor.shape}")
        logger.info(f"Output prompt: {inpaint_prompt}")
        logger.info("=== QWEN MASK PROCESSOR END ===")
        
        return (image, mask_tensor, inpaint_prompt, preview_tensor, mask_preview_tensor)
    
    def _process_mask_with_options(self, mask_pil: Image.Image, blur: float, expand: int, 
                                 feather: bool, target_width: int, target_height: int) -> Image.Image:
        """Process mask with specified options - follows diffusers preprocessing patterns"""
        
        logger.info(f"Processing mask with options - blur: {blur}, expand: {expand}, feather: {feather}")
        logger.info(f"Input mask size: {mask_pil.size}, target: {target_width}x{target_height}")
        
        # Resize to target dimensions
        if mask_pil.size != (target_width, target_height):
            logger.info(f"Resizing mask from {mask_pil.size} to {target_width}x{target_height}")
            mask_pil = mask_pil.resize((target_width, target_height), Image.LANCZOS)
        
        # Convert to grayscale
        if mask_pil.mode != 'L':
            logger.info(f"Converting mask from {mask_pil.mode} to L mode")
            mask_pil = mask_pil.convert('L')
        
        # Apply expand/contract using morphological operations
        if expand != 0:
            logger.info(f"Applying morphological operations with expand: {expand}")
            mask_array = np.array(mask_pil)
            if expand > 0:
                # Dilate (expand) - follows diffusers dilation pattern
                logger.info("Applying dilation (expand)")
                kernel = np.ones((expand*2+1, expand*2+1), np.uint8)
                try:
                    from scipy import ndimage
                    mask_array = ndimage.binary_dilation(mask_array > 128, kernel).astype(np.uint8) * 255
                except ImportError:
                    logger.warning("scipy not available, skipping morphological operations")
            else:
                # Erode (contract) - follows diffusers erosion pattern  
                logger.info("Applying erosion (contract)")
                kernel = np.ones((-expand*2+1, -expand*2+1), np.uint8)
                try:
                    from scipy import ndimage
                    mask_array = ndimage.binary_erosion(mask_array > 128, kernel).astype(np.uint8) * 255
                except ImportError:
                    logger.warning("scipy not available, skipping morphological operations")
                
            mask_pil = Image.fromarray(mask_array, mode='L')
        
        # Apply blur - follows diffusers blur pattern
        if blur > 0:
            logger.info(f"Applying Gaussian blur: {blur}")
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(blur))
        
        # Apply feathering for smooth edges
        if feather:
            logger.info("Applying edge feathering")
            mask_pil = self._feather_mask_edges(mask_pil)
        
        logger.info(f"Final processed mask size: {mask_pil.size}")
        return mask_pil
    
    def _feather_mask_edges(self, mask_pil: Image.Image) -> Image.Image:
        """Apply edge feathering for smoother inpainting blends"""
        logger.info("Applying edge feathering for smoother blends")
        
        # Simple feathering with additional blur on edges
        mask_array = np.array(mask_pil).astype(np.float32) / 255.0
        
        try:
            # Find edges
            from scipy import ndimage
            edges = ndimage.sobel(mask_array) > 0.1
            
            # Apply extra blur to edge areas
            blurred = ndimage.gaussian_filter(mask_array, sigma=1.5)
            mask_array = np.where(edges, blurred, mask_array)
            
            logger.info("Edge feathering applied successfully")
        except ImportError:
            logger.warning("scipy not available, skipping advanced edge feathering")
            # Fallback: simple blur
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(1.0))
            return mask_pil
        
        return Image.fromarray((mask_array * 255).astype(np.uint8), mode='L')
    
    def _create_inpaint_preview(self, image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
        """Create preview showing what areas will be inpainted"""
        logger.info("Creating inpaint preview")
        
        # Convert mask to RGBA overlay
        mask_array = np.array(mask_pil)
        
        # Create red overlay for inpainting areas (follows diffusers preview pattern)
        overlay_array = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
        overlay_array[mask_array > 128] = [255, 0, 0, 128]  # Semi-transparent red
        mask_rgba = Image.fromarray(overlay_array, 'RGBA')
        
        # Composite with original image
        if image_pil.mode != 'RGBA':
            image_pil = image_pil.convert('RGBA')
        
        preview = Image.alpha_composite(image_pil, mask_rgba)
        logger.info(f"Created preview image: {preview.size}")
        return preview.convert('RGB')
    
    def _create_mask_preview(self, mask_pil: Image.Image) -> torch.Tensor:
        """Create black/white mask visualization as tensor"""
        logger.info("Creating mask preview (black/white visualization)")
        
        # Convert grayscale mask to RGB for better visibility
        mask_array = np.array(mask_pil)
        
        # Create 3-channel RGB mask (white = inpaint, black = preserve)
        mask_rgb = np.stack([mask_array, mask_array, mask_array], axis=2)
        mask_rgb_pil = Image.fromarray(mask_rgb, 'RGB')
        
        # Convert to tensor
        mask_preview_tensor = self._pil_to_tensor(mask_rgb_pil)
        logger.info(f"Created mask preview: {mask_preview_tensor.shape}")
        return mask_preview_tensor
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert ComfyUI tensor to PIL - REUSE existing implementation"""
        logger.debug(f"_tensor_to_pil: Input tensor shape: {tensor.shape}")
        if len(tensor.shape) == 4:
            logger.debug("_tensor_to_pil: Removing batch dimension")
            tensor = tensor[0]
        logger.debug(f"_tensor_to_pil: Working tensor shape: {tensor.shape}")
        np_image = tensor.cpu().numpy()
        logger.debug(f"_tensor_to_pil: Numpy array shape: {np_image.shape}, dtype: {np_image.dtype}")
        logger.debug(f"_tensor_to_pil: Numpy array min/max: {np_image.min():.4f}/{np_image.max():.4f}")
        if np_image.max() <= 1.0:
            logger.debug("_tensor_to_pil: Scaling from 0-1 to 0-255")
            np_image = (np_image * 255).astype(np.uint8)
        else:
            logger.debug("_tensor_to_pil: Converting to uint8 without scaling")
            np_image = np_image.astype(np.uint8)
        logger.debug(f"_tensor_to_pil: Final numpy array dtype: {np_image.dtype}")
        pil_image = Image.fromarray(np_image)
        logger.debug(f"_tensor_to_pil: Output PIL image size: {pil_image.size}, mode: {pil_image.mode}")
        return pil_image
    
    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert PIL to ComfyUI tensor - REUSE existing implementation"""
        logger.debug(f"_pil_to_tensor: Input PIL image size: {pil_image.size}, mode: {pil_image.mode}")
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        logger.debug(f"_pil_to_tensor: Numpy array shape: {np_image.shape}, dtype: {np_image.dtype}")
        logger.debug(f"_pil_to_tensor: Numpy array min/max: {np_image.min():.4f}/{np_image.max():.4f}")
        tensor = torch.from_numpy(np_image).unsqueeze(0)
        logger.debug(f"_pil_to_tensor: Output tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        return tensor
    
    def _base64_to_pil(self, base64_data: str) -> Image.Image:
        """Convert base64 data to PIL image - REUSE existing implementation"""
        logger.debug(f"_base64_to_pil: Input data length: {len(base64_data)} characters")
        try:
            # Remove data URL prefix if present
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',', 1)[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_data)
            logger.debug(f"_base64_to_pil: Decoded to {len(image_bytes)} bytes")
            
            # Create PIL image from bytes
            pil_image = Image.open(io.BytesIO(image_bytes))
            logger.debug(f"_base64_to_pil: Created PIL image size: {pil_image.size}, mode: {pil_image.mode}")
            return pil_image
        except Exception as e:
            logger.error(f"_base64_to_pil: Error converting base64 to PIL: {e}")
            raise
    
    def _pil_to_mask_tensor(self, mask_pil: Image.Image) -> torch.Tensor:
        """Convert PIL mask to ComfyUI mask tensor"""
        logger.debug(f"_pil_to_mask_tensor: Input PIL mask size: {mask_pil.size}, mode: {mask_pil.mode}")
        mask_array = np.array(mask_pil).astype(np.float32) / 255.0
        logger.debug(f"_pil_to_mask_tensor: Mask array shape: {mask_array.shape}, dtype: {mask_array.dtype}")
        logger.debug(f"_pil_to_mask_tensor: Mask array min/max: {mask_array.min():.4f}/{mask_array.max():.4f}")
        
        # ComfyUI expects mask tensor in format [batch, height, width]
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        logger.debug(f"_pil_to_mask_tensor: Output tensor shape: {mask_tensor.shape}, dtype: {mask_tensor.dtype}")
        return mask_tensor
    
    def _mask_tensor_to_pil(self, mask_tensor: torch.Tensor, width: int, height: int) -> Image.Image:
        """Convert ComfyUI mask tensor to PIL"""
        logger.debug(f"_mask_tensor_to_pil: Input tensor shape: {mask_tensor.shape}, target size: {width}x{height}")
        
        # Handle different tensor formats
        if len(mask_tensor.shape) == 3:
            # Remove batch dimension if present
            mask_array = mask_tensor[0].cpu().numpy()
        else:
            mask_array = mask_tensor.cpu().numpy()
        
        # Convert to uint8
        mask_array = (mask_array * 255).astype(np.uint8)
        logger.debug(f"_mask_tensor_to_pil: Converted array shape: {mask_array.shape}, dtype: {mask_array.dtype}")
        
        mask_pil = Image.fromarray(mask_array, mode='L')
        if mask_pil.size != (width, height):
            logger.debug(f"_mask_tensor_to_pil: Resizing from {mask_pil.size} to {width}x{height}")
            mask_pil = mask_pil.resize((width, height), Image.LANCZOS)
        
        logger.debug(f"_mask_tensor_to_pil: Output PIL mask size: {mask_pil.size}, mode: {mask_pil.mode}")
        return mask_pil


NODE_CLASS_MAPPINGS = {
    "QwenMaskProcessor": QwenMaskProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenMaskProcessor": "Qwen Mask Processor"
}