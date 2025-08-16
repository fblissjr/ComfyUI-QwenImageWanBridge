"""
Qwen-Image to WAN 2.2 Direct Latent Bridge
Converts Qwen latents directly to WAN format - no VAE needed!
Based on DiffSynth-Studio's WAN implementation
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple

class QwenImageToWANLatentBridge:
    """
    Direct latent bridge from Qwen-Image to WAN 2.2 I2V
    
    Key insight from DiffSynth: WAN needs mask channels to indicate
    which frames have content (frame 0 from Qwen) vs frames to generate
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "num_frames": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 241,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "height": ("INT", {"default": 480, "min": 256, "max": 1024, "step": 8}),
                "width": ("INT", {"default": 832, "min": 256, "max": 1024, "step": 8}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("wan_latent",)
    FUNCTION = "bridge_latent"
    CATEGORY = "QwenWANBridge"
    
    def bridge_latent(self, qwen_latent: Dict, num_frames: int, 
                     height: int = 480, width: int = 832) -> Tuple[Dict]:
        """
        Convert Qwen-Image latent directly to WAN 2.2 I2V format
        
        Args:
            qwen_latent: Qwen latent dict with 'samples' key
                        Shape: (B, 16, H, W) 
            num_frames: Target video frames (e.g., 81)
            height: Video height (default 480 for WAN)
            width: Video width (default 832 for WAN)
            
        Returns:
            wan_latent: Properly formatted for WAN 2.2 I2V
                       Shape: (B, 16, F, H, W) where F = (num_frames-1)//4 + 1
        """
        
        # Extract Qwen latent
        qwen_samples = qwen_latent["samples"]
        
        # Handle both 4D (B, C, H, W) and 5D (B, C, F, H, W) inputs
        if len(qwen_samples.shape) == 4:
            B, C, H_latent, W_latent = qwen_samples.shape
            has_frames = False
        elif len(qwen_samples.shape) == 5:
            B, C, F_input, H_latent, W_latent = qwen_samples.shape
            has_frames = True
            # Take first frame if already has temporal dimension
            qwen_samples = qwen_samples[:, :, 0, :, :]
            B, C, H_latent, W_latent = qwen_samples.shape
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {qwen_samples.shape}")
        
        # Verify it's 16-channel (both models use z_dim=16)
        assert C == 16, f"Expected 16 channels, got {C}"
        
        # Calculate target latent dimensions (8x downsampled)
        H_target = height // 8
        W_target = width // 8
        
        # Resize if needed (WAN is picky about dimensions)
        if H_latent != H_target or W_latent != W_target:
            qwen_samples = F.interpolate(
                qwen_samples,
                size=(H_target, W_target),
                mode='bilinear',
                align_corners=False
            )
        
        # Calculate temporal frames (WAN uses 4x temporal compression)
        temporal_frames = (num_frames - 1) // 4 + 1
        
        # Create WAN latent structure (B, C, F, H, W)
        wan_samples = torch.zeros(
            B, C, temporal_frames, H_target, W_target,
            device=qwen_samples.device,
            dtype=qwen_samples.dtype
        )
        
        # Place Qwen frame as first frame
        wan_samples[:, :, 0] = qwen_samples
        
        # Debug info
        print(f"[QwenToWAN Bridge] Input shape: {qwen_latent['samples'].shape}")
        print(f"[QwenToWAN Bridge] Output shape: {wan_samples.shape}")
        print(f"[QwenToWAN Bridge] Target frames: {num_frames}, Temporal frames: {temporal_frames}")
        
        # Return in ComfyUI format
        # WAN expects (B, C, F, H, W) where F is temporal dimension
        wan_latent = {
            "samples": wan_samples,  # Shape: (B, 16, F, H, W)
            "num_frames": num_frames,
            "temporal_frames": temporal_frames,
        }
        
        return (wan_latent,)

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "QwenImageToWANLatentBridge": QwenImageToWANLatentBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageToWANLatentBridge": "Qwen→WAN Latent Bridge",
}