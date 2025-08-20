"""
Proper Native ComfyUI Bridge understanding the actual architectures
"""

import torch
import torch.nn.functional as F

class QwenWANNativeProper:
    """
    Proper bridge that understands:
    - Qwen: 16 channels
    - WAN 2.1: 16 channels (compatible!)
    - WAN 2.2: 48 channels (needs channel expansion)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "wan_version": (["wan21", "wan22"], {"default": "wan21"}),
                "channel_mode": ([
                    "repeat",      # Repeat 16→48 (3x)
                    "zeros",       # Pad with zeros
                    "interleave",  # Interleave copies
                    "learned"      # Learned projection (placeholder)
                ], {"default": "repeat"}),
                "normalization": ([
                    "none",
                    "wan_normalize",    # Apply WAN's expected normalization
                    "match_distribution" # Match WAN's distribution
                ], {"default": "wan_normalize"}),
                "num_frames": ("INT", {"default": 21, "min": 1, "max": 1024}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "info")
    FUNCTION = "bridge"
    CATEGORY = "QwenWANBridge/Native"
    
    def bridge(self, qwen_latent, wan_version, channel_mode, normalization, num_frames, seed):
        
        torch.manual_seed(seed)
        
        info = []
        info.append("Proper Native Bridge")
        info.append("="*40)
        
        # Extract Qwen latent
        qwen = qwen_latent["samples"]
        
        # Ensure batch dimension
        if len(qwen.shape) == 3:
            qwen = qwen.unsqueeze(0).unsqueeze(2)  # Add batch and temporal
        elif len(qwen.shape) == 4:
            if qwen.shape[1] != 16:
                info.append(f"WARNING: Expected 16 channels, got {qwen.shape[1]}")
            qwen = qwen.unsqueeze(2)  # Add temporal
        elif len(qwen.shape) == 5:
            # Already has temporal
            pass
        
        B, C_in, _, H, W = qwen.shape
        device = qwen.device
        dtype = qwen.dtype
        
        info.append(f"Input: {tuple(qwen.shape)}")
        info.append(f"WAN version: {wan_version}")
        
        # Determine target channels
        if wan_version == "wan21":
            C_out = 16
            info.append("WAN 2.1: 16 channels (compatible!)")
        else:  # wan22
            C_out = 48
            info.append("WAN 2.2: 48 channels (needs expansion)")
        
        # Extract single frame from Qwen
        qwen_frame = qwen[:, :, 0, :, :]  # (B, 16, H, W)
        
        # Channel handling
        if C_out == 16:
            # WAN 2.1 - direct compatibility!
            expanded = qwen_frame
            info.append("Direct channel match")
        else:
            # WAN 2.2 - need to expand 16→48
            if channel_mode == "repeat":
                # Repeat each channel 3x: 16*3=48
                expanded = qwen_frame.repeat(1, 3, 1, 1)
                info.append("Channels repeated 3x")
                
            elif channel_mode == "zeros":
                # Pad with zeros
                padding = torch.zeros(B, 32, H, W, device=device, dtype=dtype)
                expanded = torch.cat([qwen_frame, padding], dim=1)
                info.append("Padded with 32 zero channels")
                
            elif channel_mode == "interleave":
                # Interleave: ch0, ch0, ch0, ch1, ch1, ch1, ...
                expanded = qwen_frame.unsqueeze(2).repeat(1, 1, 3, 1, 1)
                expanded = expanded.reshape(B, 48, H, W)
                info.append("Channels interleaved")
                
            elif channel_mode == "learned":
                # Placeholder for learned projection
                # For now, use intelligent mixing
                expanded = torch.zeros(B, 48, H, W, device=device, dtype=dtype)
                # Copy original to first 16
                expanded[:, :16] = qwen_frame
                # Add transformed versions
                expanded[:, 16:32] = qwen_frame * 0.7 + torch.randn_like(qwen_frame) * 0.3
                expanded[:, 32:48] = qwen_frame * 0.5 + torch.randn_like(qwen_frame) * 0.5
                info.append("Pseudo-learned projection")
        
        # Apply normalization
        if normalization == "wan_normalize":
            if wan_version == "wan21":
                # WAN 2.1 normalization values
                mean = torch.tensor([
                    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
                ], device=device, dtype=dtype).view(1, 16, 1, 1)
                
                std = torch.tensor([
                    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
                ], device=device, dtype=dtype).view(1, 16, 1, 1)
                
                if C_out == 48:
                    # Repeat for 48 channels
                    mean = mean.repeat(1, 3, 1, 1)
                    std = std.repeat(1, 3, 1, 1)
            else:
                # WAN 2.2 would have its own normalization
                # For now, use standard normalization
                mean = expanded.mean(dim=(2, 3), keepdim=True)
                std = expanded.std(dim=(2, 3), keepdim=True) + 1e-6
            
            expanded = (expanded - mean) / std
            info.append("Applied WAN normalization")
            
        elif normalization == "match_distribution":
            # Match to typical WAN distribution
            current_mean = expanded.mean()
            current_std = expanded.std()
            target_mean = 0.0
            target_std = 2.0
            
            expanded = (expanded - current_mean) / (current_std + 1e-6) * target_std + target_mean
            info.append(f"Matched distribution: μ={target_mean}, σ={target_std}")
        
        # Create temporal dimension
        latent = torch.zeros(B, C_out, num_frames, H, W, device=device, dtype=dtype)
        
        # Place expanded frame
        latent[:, :, 0] = expanded.squeeze(0) if expanded.shape[0] == 1 else expanded[0]
        
        # Add subtle continuation for temporal coherence
        for t in range(1, min(3, num_frames)):
            decay = 0.9 ** t
            latent[:, :, t] = latent[:, :, 0] * decay * 0.1
        
        # Statistics
        info.append(f"\nOutput shape: {latent.shape}")
        info.append(f"Channel expansion: {C_in}→{C_out}")
        info.append(f"Stats: μ={latent.mean():.3f}, σ={latent.std():.3f}")
        
        # Important notes
        info.append("\nNotes:")
        if wan_version == "wan21":
            info.append("✓ WAN 2.1 has native 16-channel compatibility")
            info.append("✓ Should work better than WAN 2.2")
        else:
            info.append("⚠ WAN 2.2 expects 48 channels")
            info.append("⚠ Channel expansion is approximate")
            info.append("Consider using WAN 2.1 for better compatibility")
        
        return ({"samples": latent}, "\n".join(info))


class QwenWANChannelAdapter:
    """
    Dedicated channel adapter for 16→48 conversion
    More sophisticated than simple repeat
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_16ch": ("LATENT",),
                "adaptation_mode": ([
                    "frequency_based",    # Different channels for different frequencies
                    "multi_scale",        # Different scales
                    "phase_shifted",      # Phase variations
                    "mixed"              # Combination
                ], {"default": "mixed"}),
                "preserve_original": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "adapt"
    CATEGORY = "QwenWANBridge/Native"
    
    def adapt(self, latent_16ch, adaptation_mode, preserve_original):
        
        latent = latent_16ch["samples"]
        
        # Ensure proper shape
        if len(latent.shape) == 4:
            latent = latent.unsqueeze(2)  # Add temporal
        
        B, C, T, H, W = latent.shape
        device = latent.device
        dtype = latent.dtype
        
        # Create 48 channel output
        output = torch.zeros(B, 48, T, H, W, device=device, dtype=dtype)
        
        if preserve_original:
            # Keep original 16 channels intact
            output[:, :16] = latent
            
        if adaptation_mode == "frequency_based":
            # Low freq in channels 0-15 (original)
            # Mid freq in channels 16-31
            # High freq in channels 32-47
            
            for t in range(T):
                frame = latent[:, :, t, :, :]
                
                # Mid frequency - slight blur
                mid_freq = F.avg_pool2d(frame, 3, stride=1, padding=1)
                output[:, 16:32, t] = mid_freq
                
                # High frequency - edge detection
                high_freq = frame - F.avg_pool2d(frame, 5, stride=1, padding=2)
                output[:, 32:48, t] = high_freq
                
        elif adaptation_mode == "multi_scale":
            # Different scales in different channel groups
            for t in range(T):
                frame = latent[:, :, t, :, :]
                
                # Slightly downscaled and up
                small = F.interpolate(frame, scale_factor=0.5, mode='bilinear')
                small_up = F.interpolate(small, size=(H, W), mode='bilinear')
                output[:, 16:32, t] = small_up
                
                # Slightly upscaled and down  
                large = F.interpolate(frame, scale_factor=1.5, mode='bilinear')
                large_down = F.interpolate(large, size=(H, W), mode='bilinear')
                output[:, 32:48, t] = large_down
                
        elif adaptation_mode == "phase_shifted":
            # Phase-shifted versions
            output[:, 16:32] = torch.roll(latent, shifts=1, dims=-1)  # Horizontal shift
            output[:, 32:48] = torch.roll(latent, shifts=1, dims=-2)  # Vertical shift
            
        elif adaptation_mode == "mixed":
            # Combination of techniques
            for t in range(T):
                frame = latent[:, :, t, :, :]
                
                # Group 2: Blurred (low freq)
                output[:, 16:32, t] = F.avg_pool2d(frame, 3, stride=1, padding=1)
                
                # Group 3: Sharpened (high freq emphasis)
                sharp = frame * 2 - F.avg_pool2d(frame, 3, stride=1, padding=1)
                output[:, 32:48, t] = torch.clamp(sharp, -10, 10)
        
        return ({"samples": output},)