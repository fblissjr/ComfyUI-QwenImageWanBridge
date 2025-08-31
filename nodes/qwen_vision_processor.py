"""
Custom Vision Processor for Qwen2.5-VL Multi-Frame Support
Implements the missing pieces from ComfyUI's architecture
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class QwenVisionProcessor:
    """
    Processes images for Qwen2.5-VL with proper multi-frame temporal indexing.
    This implements what Qwen2VLProcessor does in HuggingFace.
    """
    
    def __init__(self, 
                 patch_size: int = 14,
                 temporal_patch_size: int = 2,
                 merge_size: int = 2,
                 min_pixels: int = 3136,
                 max_pixels: int = 12845056):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
    
    def calculate_optimal_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """
        Calculate the optimal resolution for vision processing.
        Ensures dimensions are multiples of patch_size * merge_size.
        """
        factor = self.patch_size * self.merge_size
        
        # Round to nearest factor
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        
        # Apply pixel constraints
        total_pixels = h_bar * w_bar
        
        if total_pixels > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif total_pixels < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        
        return w_bar, h_bar
    
    def process_single_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Process a single image into normalized patches.
        Returns normalized image and grid dimensions.
        """
        # Handle batch dimension
        if len(image.shape) == 4:
            image = image[0]
        
        height, width, channels = image.shape
        
        # Convert to CHW
        img = image.permute(2, 0, 1)
        
        # Calculate optimal resolution
        w_bar, h_bar = self.calculate_optimal_resolution(width, height)
        
        # Resize
        img_resized = F.interpolate(
            img.unsqueeze(0),
            size=(h_bar, w_bar),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Normalize
        normalized = torch.zeros_like(img_resized)
        for c in range(3):
            normalized[c] = (img_resized[c] - self.image_mean[c]) / self.image_std[c]
        
        # Calculate grid dimensions
        grid_h = h_bar // self.patch_size
        grid_w = w_bar // self.patch_size
        
        return normalized, (grid_h, grid_w)
    
    def create_vision_patches(self, images: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create vision patches from multiple images with temporal indexing.
        
        Args:
            images: List of images to process as temporal frames
            
        Returns:
            patches: Flattened vision patches [seq_len, patch_dim]
            grid_thw: Grid dimensions [T, H, W]
        """
        num_frames = len(images)
        processed_frames = []
        grid_dims = None
        
        # Process each frame
        for idx, image in enumerate(images):
            normalized, (grid_h, grid_w) = self.process_single_image(image)
            processed_frames.append(normalized)
            
            if grid_dims is None:
                grid_dims = (grid_h, grid_w)
            else:
                # Ensure all frames have same dimensions
                assert grid_dims == (grid_h, grid_w), "All frames must have same grid dimensions"
        
        # Stack frames: [T, C, H, W]
        pixel_values = torch.stack(processed_frames, dim=0)
        
        # Create grid_thw
        grid_t = num_frames
        grid_h, grid_w = grid_dims
        device = pixel_values.device
        
        # For temporal indexing, we need to handle frames properly
        if num_frames == 1:
            # Single frame: duplicate for temporal_patch_size
            pixel_values = pixel_values.repeat(self.temporal_patch_size, 1, 1, 1)
            grid_t = 1  # Still logically 1 frame
        
        # Reshape for patch extraction
        channel = pixel_values.shape[1]
        patches = pixel_values.reshape(
            grid_t,
            self.temporal_patch_size if grid_t == 1 else 1,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size
        )
        
        # Permute to correct order
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        
        # Flatten
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        
        # Create grid tensor
        grid_thw = torch.tensor([grid_t, grid_h, grid_w], device=device, dtype=torch.long)
        
        return flatten_patches, grid_thw
    
    def create_image_grid_thw(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Create image_grid_thw tensor for multi-image processing.
        This is what's missing from ComfyUI's pipeline.
        """
        _, grid_thw = self.create_vision_patches(images)
        # Stack for batch dimension
        return grid_thw.unsqueeze(0)


class MultiFrameVisionEncoder:
    """
    Custom vision encoder that properly handles multi-frame inputs.
    This replaces ComfyUI's single-frame vision processing.
    """
    
    def __init__(self, vision_model=None):
        self.vision_model = vision_model
        self.processor = QwenVisionProcessor()
    
    def encode_frames(self, frames: List[torch.Tensor], 
                     return_dict: bool = True) -> Dict[str, torch.Tensor]:
        """
        Encode multiple frames with proper temporal indexing.
        
        Args:
            frames: List of frame tensors
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary with:
                - hidden_states: Encoded vision features
                - image_grid_thw: Grid dimensions
                - pixel_values: Processed patches
        """
        # Create vision patches
        patches, grid_thw = self.processor.create_vision_patches(frames)
        
        # If we have access to the vision model, use it
        if self.vision_model is not None:
            try:
                # The vision model expects batched input
                batched_patches = patches.unsqueeze(0)
                batched_grid = grid_thw.unsqueeze(0)
                
                # Forward through vision encoder
                hidden_states = self.vision_model(
                    batched_patches,
                    batched_grid
                )
                
                if return_dict:
                    return {
                        "hidden_states": hidden_states,
                        "image_grid_thw": batched_grid,
                        "pixel_values": batched_patches
                    }
                return hidden_states
                
            except Exception as e:
                logger.warning(f"Could not use vision model directly: {e}")
        
        # Return processed data for external use
        if return_dict:
            return {
                "pixel_values": patches,
                "image_grid_thw": grid_thw,
                "hidden_states": None  # Would be filled by vision model
            }
        return patches


def inject_multiframe_vision(clip_model, frames: List[torch.Tensor], 
                            debug: bool = False) -> Dict[str, Any]:
    """
    Inject multi-frame vision processing into a CLIP model.
    This is the key function that makes multi-frame work.
    
    Args:
        clip_model: The CLIP model from ComfyUI
        frames: List of frames to process
        debug: Enable debug logging
        
    Returns:
        Vision encoding data to inject into the model
    """
    processor = QwenVisionProcessor()
    
    # Process frames
    patches, grid_thw = processor.create_vision_patches(frames)
    
    if debug:
        logger.info(f"[Inject] Processing {len(frames)} frames")
        logger.info(f"[Inject] Patches shape: {patches.shape}")
        logger.info(f"[Inject] Grid THW: {grid_thw}")
    
    # Try to find the vision model in CLIP
    vision_model = None
    if hasattr(clip_model, 'cond_stage_model'):
        model = clip_model.cond_stage_model
        if hasattr(model, 'visual'):
            vision_model = model.visual
            if debug:
                logger.info(f"[Inject] Found vision model in CLIP")
    
    # Create injection data
    injection_data = {
        "pixel_values": patches,
        "image_grid_thw": grid_thw,
        "num_frames": len(frames),
        "vision_model": vision_model
    }
    
    # If we have the vision model, we can process directly
    if vision_model is not None:
        try:
            # Process through vision encoder
            encoder = MultiFrameVisionEncoder(vision_model)
            vision_output = encoder.encode_frames(frames, return_dict=True)
            injection_data.update(vision_output)
            
            if debug:
                logger.info(f"[Inject] Processed through vision encoder successfully")
                
        except Exception as e:
            if debug:
                logger.error(f"[Inject] Vision encoding failed: {e}")
    
    return injection_data


# Export functions for use in nodes
__all__ = [
    'QwenVisionProcessor',
    'MultiFrameVisionEncoder', 
    'inject_multiframe_vision'
]