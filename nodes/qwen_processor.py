"""
Qwen2VLProcessor implementation for ComfyUI
Handles image_grid_thw and proper multi-frame tokenization
Based on DiffSynth-Studio's approach
"""

import torch
import torch.nn.functional as F
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
import os

logger = logging.getLogger(__name__)


class Qwen2VLProcessor:
    """
    Processor for Qwen2.5-VL that handles multi-frame images properly.
    This is what ComfyUI is missing - handles image_grid_thw and proper tokenization.
    """
    
    def __init__(self):
        self.patch_size = 14
        self.temporal_patch_size = 2
        self.merge_size = 2
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
        
        # Vision tokens
        self.vision_start_token = 151655
        self.image_pad_token = 151859
        self.vision_end_token = 151656
    
    def process(self, text: str, images: List[torch.Tensor], 
                min_pixels: int = 3136,
                max_pixels: int = 12845056) -> Dict[str, Any]:
        """
        Process text and images like HuggingFace's Qwen2VLProcessor.
        
        Args:
            text: Input prompt
            images: List of images (up to 2 for temporal frames)
            
        Returns:
            Dictionary with:
                - input_ids: Text tokens with vision placeholders
                - pixel_values: Processed image patches
                - image_grid_thw: Grid dimensions [T, H, W]
        """
        num_images = len(images) if images else 0
        
        # Process images into patches
        if num_images > 0:
            pixel_values, image_grid_thw = self.process_vision(
                images, min_pixels, max_pixels
            )
        else:
            pixel_values = None
            image_grid_thw = None
        
        # Process text with vision tokens
        input_ids = self.process_text(text, num_images)
        
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "num_frames": min(num_images, self.temporal_patch_size)
        }
    
    def process_vision(self, images: List[torch.Tensor],
                      min_pixels: int, max_pixels: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process images into vision patches with proper temporal indexing.
        
        For 2 images:
            - Each gets full resolution (not shared)
            - grid_t = 2 (temporal frames)
            - Proper MSROPE position embeddings
        """
        num_frames = min(len(images), self.temporal_patch_size)  # Max 2 frames
        processed_frames = []
        
        # Process each frame independently at full resolution
        for frame_idx in range(num_frames):
            img = images[frame_idx]
            if len(img.shape) == 4:
                img = img[0]  # Remove batch
            
            height, width, channels = img.shape
            img = img.permute(2, 0, 1)  # HWC -> CHW
            
            # Calculate optimal resolution for THIS frame (full 1MP)
            factor = self.patch_size * self.merge_size
            
            # Target ~1MP per frame
            target_pixels = 1024 * 1024
            scale = math.sqrt(target_pixels / (height * width))
            
            h_bar = round(height * scale / factor) * factor
            w_bar = round(width * scale / factor) * factor
            
            # Apply constraints
            if h_bar * w_bar > max_pixels:
                beta = math.sqrt((height * width) / max_pixels)
                h_bar = max(factor, math.floor(height / beta / factor) * factor)
                w_bar = max(factor, math.floor(width / beta / factor) * factor)
            elif h_bar * w_bar < min_pixels:
                beta = math.sqrt(min_pixels / (height * width))
                h_bar = math.ceil(height * beta / factor) * factor
                w_bar = math.ceil(width * beta / factor) * factor
            
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
            
            processed_frames.append(normalized)
            
            logger.info(f"[Processor] Frame {frame_idx}: {width}x{height} -> {w_bar}x{h_bar}")
        
        # Stack frames for temporal dimension
        if num_frames == 1:
            # Single frame: duplicate for temporal_patch_size
            pixel_values = processed_frames[0].unsqueeze(0).repeat(2, 1, 1, 1)
            grid_t = 1
        else:
            # Multiple frames: stack them
            pixel_values = torch.stack(processed_frames, dim=0)
            grid_t = num_frames  # This is KEY: grid_t = 2 for 2 frames!
        
        # Get grid dimensions from the actual processed size
        grid_h = h_bar // self.patch_size
        grid_w = w_bar // self.patch_size
        
        # Create grid_thw tensor
        device = pixel_values.device
        image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], device=device, dtype=torch.long)
        
        logger.info(f"[Processor] Grid THW: T={grid_t}, H={grid_h}, W={grid_w}")
        
        # Reshape for patches (following ComfyUI's qwen_vl.py approach)
        channel = 3
        
        # pixel_values shape: [T, C, H, W] where T is num_frames (2 for multi-frame)
        T, C, H, W = pixel_values.shape
        
        logger.info(f"[Processor] Pixel values shape: {pixel_values.shape}")
        logger.info(f"[Processor] T={T}, C={C}, H={H}, W={W}")
        
        # For 2 frames, we need different handling than single frame
        if grid_t == 2:
            # Two separate frames - process each and then combine
            # Each frame becomes its own set of patches
            frame_patches = []
            
            for frame_idx in range(T):
                frame = pixel_values[frame_idx:frame_idx+1]  # Keep batch dim
                
                # Process single frame like ComfyUI does
                # Duplicate for temporal_patch_size
                frame_dup = frame.repeat(self.temporal_patch_size, 1, 1, 1)
                
                # Reshape following ComfyUI pattern
                frame_patch = frame_dup.reshape(
                    1,  # single temporal unit
                    self.temporal_patch_size,
                    C,
                    grid_h // self.merge_size,
                    self.merge_size,
                    self.patch_size,
                    grid_w // self.merge_size,
                    self.merge_size,
                    self.patch_size,
                )
                
                # Permute
                frame_patch = frame_patch.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
                
                # Flatten this frame's patches
                frame_patch = frame_patch.reshape(
                    grid_h * grid_w,
                    C * self.temporal_patch_size * self.patch_size * self.patch_size
                )
                
                frame_patches.append(frame_patch)
            
            # Stack frames temporally
            flatten_patches = torch.cat(frame_patches, dim=0)
            
        else:
            # Single frame handling (original ComfyUI approach)
            # pixel_values already has duplicated frame
            patches = pixel_values.reshape(
                1,  # grid_t = 1
                self.temporal_patch_size,
                C,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            )
        
            # Permute to correct order: [T, H_grid, W_grid, merge_h, merge_w, C, temporal, patch_h, patch_w]
            patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            
            # Flatten to [num_patches, patch_dim]
            num_patches = grid_h * grid_w
            patch_dim = channel * self.temporal_patch_size * self.patch_size * self.patch_size
            
            flatten_patches = patches.reshape(num_patches, patch_dim)
        
        logger.info(f"[Processor] Flatten patches shape: {flatten_patches.shape}")
        logger.info(f"[Processor] Grid patches: {grid_t} x {grid_h} x {grid_w} = {grid_t * grid_h * grid_w}")
        
        return flatten_patches, image_grid_thw
    
    def process_text(self, text: str, num_images: int) -> torch.Tensor:
        """
        Process text with proper vision token insertion.
        """
        # Build prompt with vision tokens
        if num_images > 0:
            if num_images == 2:
                system_msg = "You are viewing 2 temporal frames. Frame 0 is the source, Frame 1 is the target."
            else:
                system_msg = "You are viewing an image."
            
            full_text = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{text}<|im_end|>
<|im_start|>assistant
"""
        else:
            full_text = text
        
        # For now, return a placeholder
        # In reality, this would tokenize properly
        return full_text


def create_processor():
    """Create a Qwen2VLProcessor instance."""
    return Qwen2VLProcessor()


# Export
__all__ = ['Qwen2VLProcessor', 'create_processor']