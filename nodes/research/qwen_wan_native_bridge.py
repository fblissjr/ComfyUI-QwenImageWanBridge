"""
Simple Qwen to Native ComfyUI WAN Bridge
Works with standard KSampler, not Kijai's wrapper
"""

import torch
import torch.nn.functional as F

class QwenWANNativeBridge:
    """
    Bridge Qwen latents to native ComfyUI WAN format
    Simple approach - just repeat the frame
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "num_frames": ("INT", {"default": 49, "min": 1, "max": 1024, "step": 4}),
                "mode": (["repeat", "noise", "interpolate"], {"default": "repeat"}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "bridge"
    CATEGORY = "QwenWANBridge"
    
    def bridge(self, qwen_latent, num_frames, mode):
        # Get Qwen latent (16 channels)
        qwen = qwen_latent["samples"]
        
        # Handle different input shapes
        if len(qwen.shape) == 5:
            # (B, C, F, H, W)
            B, C, F, H, W = qwen.shape
            qwen = qwen[:, :, 0, :, :]  # Take first frame
        elif len(qwen.shape) == 4:
            # (B, C, H, W)
            B, C, H, W = qwen.shape
        else:
            raise ValueError(f"Unexpected shape: {qwen.shape}")
        
        # Ensure batch dimension
        if B == 0:
            B = 1
            
        # Calculate temporal frames for WAN
        temporal_frames = ((num_frames - 1) // 4) + 1
        
        # Create output tensor (B, C, T, H, W)
        output = torch.zeros(B, C, temporal_frames, H, W, 
                           dtype=qwen.dtype, device=qwen.device)
        
        if mode == "repeat":
            # Repeat first frame across all temporal positions
            for t in range(temporal_frames):
                output[:, :, t, :, :] = qwen
                
        elif mode == "noise":
            # Random noise but keep first frame
            output = torch.randn_like(output)
            output[:, :, 0, :, :] = qwen
            
        elif mode == "interpolate":
            # Interpolate between first and slightly modified last
            output[:, :, 0, :, :] = qwen
            # Add slight variation to last frame
            output[:, :, -1, :, :] = qwen + torch.randn_like(qwen) * 0.1
            # Linear interpolation for middle frames
            if temporal_frames > 2:
                for t in range(1, temporal_frames - 1):
                    alpha = t / (temporal_frames - 1)
                    output[:, :, t, :, :] = (1 - alpha) * output[:, :, 0, :, :] + alpha * output[:, :, -1, :, :]
        
        print(f"[QwenWANNativeBridge] Bridged latent:")
        print(f"  Input: {qwen_latent['samples'].shape}")
        print(f"  Output: {output.shape}")
        print(f"  Mode: {mode}")
        print(f"  Frames: {num_frames} → {temporal_frames} temporal")
        
        return ({"samples": output},)