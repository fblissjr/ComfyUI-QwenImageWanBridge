"""
Bridge Nodes for Qwen-WAN
Practical solutions for cross-model workflows
"""

import torch
import torch.nn.functional as F

class QwenWANBridge:
    """Bridge between Qwen and WAN models"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "source_model": (["qwen", "wan"],),
                "target_model": (["qwen", "wan"],),
                "wan_vae": ("VAE",),
            },
            "optional": {
                "fix_temporal": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "bridge"
    CATEGORY = "QwenWAN/Bridge"
    
    def bridge(self, latent, source_model, target_model, wan_vae, fix_temporal=True):
        """Bridge latents between models"""
        
        # If same model, no bridging needed
        if source_model == target_model:
            return (latent,)
        
        # Key insight: Use WAN VAE for everything
        # This avoids the bizarre frames issue entirely
        
        samples = latent["samples"]
        
        if source_model == "qwen" and target_model == "wan":
            # Qwen to WAN
            if fix_temporal and samples.dim() == 4:
                # Add temporal dimension if missing
                # WAN expects (B, C, F, H, W) for video
                samples = samples.unsqueeze(2)
        
        elif source_model == "wan" and target_model == "qwen":
            # WAN to Qwen
            if samples.dim() == 5:
                # Remove temporal dimension for single frame
                # Take first frame
                samples = samples[:, :, 0, :, :]
        
        return ({"samples": samples},)



class LatentMixer:
    """Mix latents from different models"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
                "mix_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "mix_mode": (["linear", "slerp"],),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "mix"
    CATEGORY = "QwenWAN/Bridge"
    
    def mix(self, latent_a, latent_b, mix_ratio, mix_mode):
        """Mix latents in shared space"""
        
        a = latent_a["samples"]
        b = latent_b["samples"]
        
        # Ensure same shape
        if a.shape != b.shape:
            # Try to match dimensions
            if a.dim() == 4 and b.dim() == 5:
                a = a.unsqueeze(2)
            elif a.dim() == 5 and b.dim() == 4:
                b = b.unsqueeze(2)
            
            # If still mismatch, take smaller shape
            if a.shape != b.shape:
                min_shape = [min(s1, s2) for s1, s2 in zip(a.shape, b.shape)]
                a = a[:min_shape[0], :min_shape[1]]
                b = b[:min_shape[0], :min_shape[1]]
                if len(min_shape) > 2:
                    a = a[:, :, :min_shape[2]] if a.dim() > 2 else a
                    b = b[:, :, :min_shape[2]] if b.dim() > 2 else b
                if len(min_shape) > 3:
                    a = a[:, :, :, :min_shape[3]] if a.dim() > 3 else a
                    b = b[:, :, :, :min_shape[3]] if b.dim() > 3 else b
                if len(min_shape) > 4:
                    a = a[:, :, :, :, :min_shape[4]] if a.dim() > 4 else a
                    b = b[:, :, :, :, :min_shape[4]] if b.dim() > 4 else b
        
        if mix_mode == "linear":
            mixed = (1 - mix_ratio) * a + mix_ratio * b
        else:  # slerp
            mixed = self.slerp(a, b, mix_ratio)
        
        return ({"samples": mixed},)
    
    def slerp(self, v0, v1, t):
        """Spherical linear interpolation"""
        v0_flat = v0.flatten(1)
        v1_flat = v1.flatten(1)
        
        # Normalize
        v0_norm = v0_flat / (torch.norm(v0_flat, dim=1, keepdim=True) + 1e-8)
        v1_norm = v1_flat / (torch.norm(v1_flat, dim=1, keepdim=True) + 1e-8)
        
        # Compute angle
        dot = torch.sum(v0_norm * v1_norm, dim=1, keepdim=True)
        dot = torch.clamp(dot, -1, 1)
        theta = torch.acos(dot)
        
        # Interpolate
        sin_theta = torch.sin(theta)
        
        # Handle small angles
        small_angle = sin_theta < 0.01
        
        result = torch.where(
            small_angle,
            (1 - t) * v0_flat + t * v1_flat,  # Linear for small angles
            (torch.sin((1 - t) * theta) / sin_theta) * v0_flat + 
            (torch.sin(t * theta) / sin_theta) * v1_flat
        )
        
        return result.reshape_as(v0)