"""
Qwen-Image to WAN Bridge V2 - Exact replication of WanVideoImageToVideoEncode
But using Qwen latents directly instead of VAE encoding
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import gc

# Import from WAN - assuming these are available
try:
    from comfy.utils import common_upscale
    from comfy import model_management as mm
    device = mm.intermediate_device()
    offload_device = mm.unet_offload_device()
except:
    # Fallback for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    offload_device = torch.device("cpu")
    def common_upscale(x, w, h, method, crop):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

# Constants from WAN
PATCH_SIZE = (1, 2, 2)  # Default patch size

def add_noise_to_reference_video(video, ratio=0.0):
    """Add noise augmentation to reference video"""
    if ratio > 0:
        noise = torch.randn_like(video) * ratio
        video = video + noise
    return video

class QwenWANBridgeV2:
    """
    Exact replication of WanVideoImageToVideoEncode logic
    But takes Qwen latents instead of images
    """
    
    @classmethod  
    def INPUT_TYPES(s):
        return {"required": {
            "qwen_latent": ("LATENT",),
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4}),
            "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            "start_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            "end_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
            "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS",),
                "end_latent": ("LATENT",),  # Optional end frame for interpolation
                "control_embeds": ("WANVIDIMAGE_EMBEDS",),
                "fun_or_fl2v_model": ("BOOLEAN", {"default": False}),
                "temporal_mask": ("MASK",),
                "extra_latents": ("LATENT",),
                "add_cond_latents": ("ADD_COND_LATENTS",),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "QwenWANBridge"

    def process(self, qwen_latent, width, height, num_frames, force_offload, 
                noise_aug_strength, start_latent_strength, end_latent_strength, 
                end_latent=None, control_embeds=None, fun_or_fl2v_model=False, 
                temporal_mask=None, extra_latents=None, clip_embeds=None, 
                add_cond_latents=None):
        
        H = height
        W = width
        lat_h = H // 8
        lat_w = W // 8
        
        # Extract and prepare start latent from Qwen
        start_latent = qwen_latent["samples"]
        
        # Handle various input shapes - Qwen might give us different formats
        if len(start_latent.shape) == 5:
            # (B, C, F, H, W) - take first batch and first frame
            start_latent = start_latent[0, :, 0, :, :]
        elif len(start_latent.shape) == 4:
            # (B, C, H, W) - take first batch
            if start_latent.shape[0] > 1:
                print(f"[QwenWANBridgeV2] Using first batch only")
            start_latent = start_latent[0]
        elif len(start_latent.shape) == 3:
            # Already (C, H, W)
            pass
        else:
            raise ValueError(f"Unexpected latent shape: {start_latent.shape}")
        
        C = start_latent.shape[0]
        assert C == 16, f"Expected 16 channels, got {C}"
        
        # Resize if needed
        if start_latent.shape[-2:] != (lat_h, lat_w):
            start_latent = start_latent.unsqueeze(0)
            start_latent = F.interpolate(start_latent, size=(lat_h, lat_w), 
                                        mode='bilinear', align_corners=False)
            start_latent = start_latent.squeeze(0)
        
        # Process end latent if provided
        if end_latent is not None:
            end_samples = end_latent["samples"]
            if len(end_samples.shape) >= 4:
                end_samples = end_samples[0] if end_samples.shape[0] > 0 else end_samples
            if len(end_samples.shape) == 4:
                end_samples = end_samples[0]
            if end_samples.shape[-2:] != (lat_h, lat_w):
                end_samples = F.interpolate(end_samples.unsqueeze(0), 
                                           size=(lat_h, lat_w), mode='bilinear', align_corners=False).squeeze(0)
        else:
            end_samples = None
        
        # Align frames to WAN requirements
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        two_ref_images = start_latent is not None and end_samples is not None
        
        if start_latent is None and end_samples is not None:
            fun_or_fl2v_model = True
        
        base_frames = num_frames + (1 if two_ref_images and not fun_or_fl2v_model else 0)
        
        # Create mask (in pixel frame space)
        if temporal_mask is None:
            mask = torch.zeros(1, base_frames, lat_h, lat_w, device=device)
            # For I2V, only first frame has content
            mask[:, 0] = 1  # First frame has content
            if end_samples is not None:
                mask[:, -1] = 1  # End frame if exists
        else:
            mask = common_upscale(temporal_mask.unsqueeze(1).to(device), lat_w, lat_h, "nearest", "disabled").squeeze(1)
            if mask.shape[0] > base_frames:
                mask = mask[:base_frames]
            elif mask.shape[0] < base_frames:
                mask = torch.cat([mask, torch.zeros(base_frames - mask.shape[0], lat_h, lat_w, device=device)])
            mask = mask.unsqueeze(0).to(device)
        
        # Repeat first frame mask (exactly as WAN does)
        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        if end_samples is not None and not fun_or_fl2v_model:
            end_mask_repeated = torch.repeat_interleave(mask[:, -1:], repeats=4, dim=1)
            mask = torch.cat([start_mask_repeated, mask[:, 1:-1], end_mask_repeated], dim=1)
        else:
            mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)
        
        # Reshape mask into groups of 4 frames
        mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w)
        mask = mask.movedim(1, 2)[0]  # (4, T, H, W)
        
        # Create latent tensor structure (replaces VAE encoding step)
        # Calculate temporal frames
        temporal_frames = (num_frames - 1) // 4 + (2 if end_samples is not None and not fun_or_fl2v_model else 1)
        
        # Initialize latent tensor
        y = torch.zeros(C, temporal_frames, lat_h, lat_w, device=device, dtype=start_latent.dtype)
        
        # Place start frame
        y[:, 0] = start_latent.to(device)
        
        # Apply noise augmentation if requested
        if noise_aug_strength > 0:
            noise = torch.randn_like(y[:, 0]) * noise_aug_strength
            y[:, 0] = y[:, 0] + noise
        
        # Place end frame if provided
        if end_samples is not None:
            if fun_or_fl2v_model:
                y[:, -1] = end_samples.to(device)
            else:
                # For non-fun models, end frame goes to second-to-last position
                y[:, -2] = end_samples.to(device)
        
        # Handle extra latents (for reference frames)
        has_ref = False
        if extra_latents is not None:
            samples = extra_latents["samples"].squeeze(0)
            if len(samples.shape) == 4 and samples.shape[0] == 1:
                samples = samples.squeeze(0)
            y = torch.cat([samples, y], dim=1)
            mask = torch.cat([torch.ones_like(mask[:, 0:samples.shape[1]]), mask], dim=1)
            num_frames += samples.shape[1] * 4
            has_ref = True
        
        # Apply latent strength multipliers
        y[:, :1] *= start_latent_strength
        if end_samples is not None:
            y[:, -1:] *= end_latent_strength
        
        # Calculate maximum sequence length (for transformer)
        patches_per_frame = lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])
        frames_per_stride = (num_frames - 1) // 4 + (2 if end_samples is not None and not fun_or_fl2v_model else 1)
        max_seq_len = frames_per_stride * patches_per_frame
        
        # Memory management
        if force_offload:
            mm.soft_empty_cache()
            gc.collect()
        
        # For T2V models, we need target_shape instead of image_embeds
        # But for I2V, we provide the actual latents
        target_shape = (16, (num_frames - 1) // 4 + 1, lat_h, lat_w)
        
        # Create the exact image_embeds structure WAN expects
        image_embeds = {
            "image_embeds": y,
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "negative_clip_context": clip_embeds.get("negative_clip_embeds", None) if clip_embeds is not None else None,
            "max_seq_len": max_seq_len,
            "num_frames": num_frames,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "control_embeds": control_embeds["control_embeds"] if control_embeds is not None else None,
            "end_image": None,  # We don't have pixel-space end image
            "fun_or_fl2v_model": fun_or_fl2v_model,
            "has_ref": has_ref,
            "add_cond_latents": add_cond_latents,
            "mask": mask,
            "target_shape": target_shape  # Include for compatibility
        }
        
        print(f"[QwenWANBridgeV2] Processed Qwen latent to WAN format:")
        print(f"  Input shape: {qwen_latent['samples'].shape}")
        print(f"  Output image_embeds: {y.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Num frames: {num_frames}, Temporal frames: {temporal_frames}")
        print(f"  Max seq len: {max_seq_len}")
        print(f"  Has ref: {has_ref}")
        
        return (image_embeds,)