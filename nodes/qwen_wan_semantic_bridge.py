"""
Semantic Bridge leveraging the Qwen2.5-VL connection
WAN 2.2 was trained on Qwen2.5-VL captions, creating natural alignment
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

class QwenWANSemanticBridge:
    """
    Leverages the fact that WAN 2.2 was trained on Qwen2.5-VL captioned data
    This creates a natural semantic alignment between the models
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "target_width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 8}),
                "target_height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 4}),
                "resize_mode": (["bilinear", "nearest", "area", "bicubic"], {"default": "bilinear"}),
                "preserve_aspect": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "qwen_prompt": ("STRING", {"multiline": True}),  # Original Qwen prompt
            }
        }
    
    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "STRING")
    RETURN_NAMES = ("image_embeds", "analysis")
    FUNCTION = "bridge"
    CATEGORY = "QwenWANBridge"
    
    def bridge(self, qwen_latent, target_width, target_height, num_frames, 
               resize_mode, preserve_aspect, qwen_prompt=""):
        
        analysis = []
        analysis.append("Qwen-WAN Semantic Bridge Analysis")
        analysis.append("="*50)
        analysis.append("Leveraging Qwen2.5-VL training connection")
        analysis.append("")
        
        # Extract Qwen latent
        qwen = qwen_latent["samples"]
        
        # Handle input shapes
        if len(qwen.shape) == 5:
            # (B, C, F, H, W) - Qwen produces single frame
            B, C, F, H_orig, W_orig = qwen.shape
            qwen = qwen[0, :, 0, :, :]  # Extract first frame
            analysis.append(f"Input: 5D tensor {tuple(qwen.shape)}, extracted frame 0")
        elif len(qwen.shape) == 4:
            # (B, C, H, W)
            B, C, H_orig, W_orig = qwen.shape
            qwen = qwen[0]
            analysis.append(f"Input: 4D tensor {tuple(qwen.shape)}")
        else:
            # (C, H, W)
            C, H_orig, W_orig = qwen.shape
            analysis.append(f"Input: 3D tensor {tuple(qwen.shape)}")
        
        device = qwen.device
        dtype = qwen.dtype
        
        # Calculate target latent dimensions (8x downscale from pixel space)
        lat_h = target_height // 8
        lat_w = target_width // 8
        
        analysis.append(f"Original latent size: {H_orig}x{W_orig}")
        analysis.append(f"Target latent size: {lat_h}x{lat_w}")
        
        # CRITICAL: Resize to match WAN's expected dimensions
        if (H_orig, W_orig) != (lat_h, lat_w):
            # Add batch dim for interpolation
            qwen_resized = qwen.unsqueeze(0)
            
            if preserve_aspect:
                # Calculate scaling to preserve aspect ratio
                scale_h = lat_h / H_orig
                scale_w = lat_w / W_orig
                scale = min(scale_h, scale_w)
                
                new_h = int(H_orig * scale)
                new_w = int(W_orig * scale)
                
                # Resize preserving aspect
                qwen_resized = F.interpolate(
                    qwen_resized, 
                    size=(new_h, new_w),
                    mode=resize_mode,
                    align_corners=False if resize_mode in ['bilinear', 'bicubic'] else None
                )
                
                # Pad to target size
                pad_h = lat_h - new_h
                pad_w = lat_w - new_w
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                qwen_resized = F.pad(
                    qwen_resized,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='constant',
                    value=0
                )
                
                analysis.append(f"Resized with aspect preservation: {new_h}x{new_w} -> {lat_h}x{lat_w}")
            else:
                # Direct resize to target
                qwen_resized = F.interpolate(
                    qwen_resized,
                    size=(lat_h, lat_w),
                    mode=resize_mode,
                    align_corners=False if resize_mode in ['bilinear', 'bicubic'] else None
                )
                analysis.append(f"Direct resize to {lat_h}x{lat_w}")
            
            qwen_resized = qwen_resized.squeeze(0)
        else:
            qwen_resized = qwen
            analysis.append("No resize needed - dimensions match")
        
        # Align frames to WAN requirements
        num_frames_aligned = ((num_frames - 1) // 4) * 4 + 1
        temporal_frames = (num_frames_aligned - 1) // 4 + 1
        
        analysis.append(f"\nFrame alignment:")
        analysis.append(f"  Requested: {num_frames}")
        analysis.append(f"  Aligned: {num_frames_aligned}")
        analysis.append(f"  Temporal: {temporal_frames}")
        
        # Since WAN was trained on Qwen2.5-VL captions, the latent distributions
        # should already be semantically aligned. We just need structural alignment.
        
        # Analyze latent statistics
        qwen_mean = qwen_resized.mean().item()
        qwen_std = qwen_resized.std().item()
        qwen_min = qwen_resized.min().item()
        qwen_max = qwen_resized.max().item()
        
        analysis.append(f"\nQwen latent statistics (resized):")
        analysis.append(f"  Mean: {qwen_mean:.3f}")
        analysis.append(f"  Std:  {qwen_std:.3f}")
        analysis.append(f"  Range: [{qwen_min:.3f}, {qwen_max:.3f}]")
        
        # Create temporal tensor
        y = torch.zeros(C, temporal_frames, lat_h, lat_w, device=device, dtype=dtype)
        
        # Place Qwen frame as first frame
        # No normalization needed - models share semantic space
        y[:, 0] = qwen_resized
        
        # For multi-frame, add very subtle continuation
        # WAN will generate motion from this seed
        if temporal_frames > 1:
            # Add tiny amount of the first frame to next few frames
            # This helps WAN understand the content should persist
            for t in range(1, min(3, temporal_frames)):
                y[:, t] = qwen_resized * (0.01 / (t + 1))
            analysis.append(f"\nAdded {min(3, temporal_frames)-1} continuation frames")
        
        # Create mask - first frame is the key frame
        mask = torch.zeros(1, num_frames_aligned, lat_h, lat_w, device=device)
        mask[:, 0] = 1.0
        
        # Reshape mask for WAN's expected format
        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)
        
        # Ensure correct temporal alignment
        frames_needed = temporal_frames * 4
        if mask.shape[1] < frames_needed:
            padding = torch.zeros(1, frames_needed - mask.shape[1], lat_h, lat_w, device=device)
            mask = torch.cat([mask, padding], dim=1)
        elif mask.shape[1] > frames_needed:
            mask = mask[:, :frames_needed]
        
        # Reshape into WAN's expected format
        mask = mask.view(1, temporal_frames, 4, lat_h, lat_w)
        mask = mask.movedim(1, 2)[0]  # (4, T, H, W)
        
        analysis.append(f"\nOutput shapes:")
        analysis.append(f"  Latent: {y.shape}")
        analysis.append(f"  Mask: {mask.shape}")
        
        # Add prompt information if provided
        if qwen_prompt:
            analysis.append(f"\nOriginal Qwen prompt:")
            analysis.append(f'  "{qwen_prompt[:100]}{"..." if len(qwen_prompt) > 100 else ""}"')
            analysis.append(f"  (WAN was trained on similar Qwen2.5-VL descriptions)")
        
        # Create I2V embedding structure
        image_embeds = {
            "image_embeds": y,
            "mask": mask,
            "num_frames": num_frames_aligned,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "target_shape": (C, temporal_frames, lat_h, lat_w),
            "has_ref": False,
            "fun_or_fl2v_model": False,
            # Metadata
            "clip_context": None,
            "negative_clip_context": None,
            "control_embeds": None,
            "add_cond_latents": None,
            "end_image": None,
            "max_seq_len": lat_h * lat_w // 4 * temporal_frames,
        }
        
        analysis.append("\nSemantic alignment: Native (via Qwen2.5-VL training)")
        analysis.append("Ready for WAN I2V generation")
        
        return (image_embeds, "\n".join(analysis))


class QwenWANDimensionHelper:
    """
    Helper to find optimal dimensions for Qwen->WAN transfer
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "wan_model_type": (["wan2.2_i2v", "wan2.0", "custom"], {"default": "wan2.2_i2v"}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "num_frames", "recommendation")
    FUNCTION = "calculate"
    CATEGORY = "QwenWANBridge"
    
    def calculate(self, qwen_latent, wan_model_type):
        # Extract Qwen dimensions
        qwen = qwen_latent["samples"]
        
        if len(qwen.shape) >= 4:
            H = qwen.shape[-2]
            W = qwen.shape[-1]
        else:
            H = W = 64  # Default
        
        # Common WAN dimensions
        wan_dimensions = {
            "wan2.2_i2v": [
                (832, 480, 81),   # Standard
                (768, 768, 81),   # Square
                (512, 512, 81),   # Small square
                (1024, 576, 81),  # Wide
            ],
            "wan2.0": [
                (512, 512, 49),
                (768, 432, 49),
            ],
            "custom": [
                (W * 8, H * 8, 81),  # Direct scale
            ]
        }
        
        options = wan_dimensions.get(wan_model_type, wan_dimensions["wan2.2_i2v"])
        
        # Find closest match
        current_pixels = H * W
        best_match = options[0]
        best_diff = float('inf')
        
        for width, height, frames in options:
            lat_h = height // 8
            lat_w = width // 8
            diff = abs(lat_h * lat_w - current_pixels)
            if diff < best_diff:
                best_diff = diff
                best_match = (width, height, frames)
        
        width, height, frames = best_match
        
        rec = f"""Dimension Recommendations
========================
Qwen latent shape: {qwen.shape}
Qwen latent size: {H}x{W} = {H*W} elements

Recommended WAN settings:
  Width: {width}px (latent: {width//8})
  Height: {height}px (latent: {height//8})
  Frames: {frames} (temporal: {(frames-1)//4+1})

Why these dimensions:
- WAN 2.2 prefers 832x480 or 768x768
- Must be divisible by 8 (VAE requirement)
- Frames must be 4n+1 (e.g., 1, 5, 9, ..., 81)
- Smaller dimensions = faster generation
- Start with frames=1 for testing

Alternative options for {wan_model_type}:
"""
        for w, h, f in options:
            rec += f"\n  {w}x{h} @ {f} frames"
        
        return (width, height, frames, rec)