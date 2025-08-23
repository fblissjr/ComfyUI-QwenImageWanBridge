"""
Minimal Qwen to WAN Bridge - Preserve first frame exactly
Focus on why single frame works but multi-frame doesn't
"""

import torch
import torch.nn.functional as F

class QwenWANMinimalBridge:
    """
    Minimal bridge focusing on preserving Qwen's first frame exactly
    No normalization, no modification - just temporal extension
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 4}),
                "extension_mode": (["zeros", "repeat", "decay", "noise_blend"], {"default": "zeros"}),
                "decay_factor": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    FUNCTION = "bridge"
    CATEGORY = "QwenWANBridge"
    
    def bridge(self, qwen_latent, num_frames, extension_mode, decay_factor):
        # Extract Qwen latent - preserve exactly as-is
        qwen = qwen_latent["samples"]
        
        # Handle input shapes
        if len(qwen.shape) == 5:
            # (B, C, F, H, W) - take first frame
            qwen = qwen[0, :, 0, :, :]
        elif len(qwen.shape) == 4:
            # (B, C, H, W)
            qwen = qwen[0]
        # Now we have (C, H, W)
        
        C, H, W = qwen.shape
        device = qwen.device
        dtype = qwen.dtype
        
        # Calculate temporal frames for WAN (4x compression)
        num_frames_aligned = ((num_frames - 1) // 4) * 4 + 1
        temporal_frames = (num_frames_aligned - 1) // 4 + 1
        
        print(f"[QwenWANMinimalBridge] Processing:")
        print(f"  Input Qwen shape: {qwen.shape}")
        print(f"  Frames: {num_frames} → {num_frames_aligned} (aligned) → {temporal_frames} (temporal)")
        print(f"  Extension mode: {extension_mode}")
        print(f"  First frame stats: min={qwen.min():.3f}, max={qwen.max():.3f}, mean={qwen.mean():.3f}, std={qwen.std():.3f}")
        
        # Create temporal tensor
        y = torch.zeros(C, temporal_frames, H, W, device=device, dtype=dtype)
        
        # CRITICAL: Preserve first frame EXACTLY as Qwen provides
        y[:, 0] = qwen
        
        # Extend to additional frames based on mode
        if temporal_frames > 1:
            if extension_mode == "zeros":
                # Already zeros, do nothing
                pass
                
            elif extension_mode == "repeat":
                # Repeat first frame
                for t in range(1, temporal_frames):
                    y[:, t] = qwen
                    
            elif extension_mode == "decay":
                # Gradually decay to zero
                for t in range(1, temporal_frames):
                    factor = decay_factor ** t
                    y[:, t] = qwen * factor
                    
            elif extension_mode == "noise_blend":
                # Blend with noise over time
                for t in range(1, temporal_frames):
                    alpha = t / (temporal_frames - 1)
                    noise = torch.randn_like(qwen) * qwen.std()
                    y[:, t] = (1 - alpha) * qwen + alpha * noise
        
        # Create mask - first frame has content
        mask = torch.zeros(1, num_frames_aligned, H, W, device=device)
        mask[:, 0] = 1.0
        
        # Reshape mask for WAN (groups of 4)
        # Repeat first frame mask 4 times
        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)
        
        # Ensure we have right number of frames for reshaping
        frames_needed = temporal_frames * 4
        if mask.shape[1] < frames_needed:
            # Pad with zeros
            padding = torch.zeros(1, frames_needed - mask.shape[1], H, W, device=device)
            mask = torch.cat([mask, padding], dim=1)
        elif mask.shape[1] > frames_needed:
            # Trim
            mask = mask[:, :frames_needed]
        
        # Reshape into groups of 4
        mask = mask.view(1, temporal_frames, 4, H, W)
        mask = mask.movedim(1, 2)[0]  # (4, T, H, W)
        
        print(f"  Output latent shape: {y.shape}")
        print(f"  Output mask shape: {mask.shape}")
        print(f"  Latent stats: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}, std={y.std():.3f}")
        
        # Create I2V structure for WAN
        image_embeds = {
            "image_embeds": y,
            "mask": mask,
            "num_frames": num_frames_aligned,
            "lat_h": H,
            "lat_w": W,
            "target_shape": (C, temporal_frames, H, W),
            "has_ref": False,
            "fun_or_fl2v_model": False,
            # Minimal metadata
            "clip_context": None,
            "negative_clip_context": None,
            "control_embeds": None,
            "add_cond_latents": None,
            "end_image": None,
            "max_seq_len": H * W // 4 * temporal_frames,  # Approximate
        }
        
        return (image_embeds,)


class QwenFrameAnalyzer:
    """
    Analyze what happens when WAN processes Qwen frames
    Test with different frame counts to understand the issue
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "wan_output": ("LATENT",),
                "frame_to_check": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze"
    CATEGORY = "QwenWANBridge"
    
    def analyze(self, qwen_latent, wan_output, frame_to_check):
        qwen = qwen_latent["samples"]
        wan = wan_output["samples"]
        
        # Get first frame from Qwen
        if len(qwen.shape) == 5:
            qwen_frame = qwen[0, :, 0, :, :]
        elif len(qwen.shape) == 4:
            qwen_frame = qwen[0]
        else:
            qwen_frame = qwen
        
        # Get frame from WAN output
        if len(wan.shape) == 5:
            # (B, C, T, H, W)
            if frame_to_check < wan.shape[2]:
                wan_frame = wan[0, :, frame_to_check, :, :]
            else:
                wan_frame = wan[0, :, 0, :, :]
        elif len(wan.shape) == 4:
            # No temporal dimension
            wan_frame = wan[0]
        else:
            wan_frame = wan
        
        # Resize if needed for comparison
        if qwen_frame.shape != wan_frame.shape:
            wan_frame = F.interpolate(
                wan_frame.unsqueeze(0), 
                size=qwen_frame.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Calculate similarity metrics
        mse = ((qwen_frame - wan_frame) ** 2).mean().item()
        rmse = mse ** 0.5
        
        # Cosine similarity
        qwen_flat = qwen_frame.flatten()
        wan_flat = wan_frame.flatten()
        cosine_sim = (torch.dot(qwen_flat, wan_flat) / 
                     (qwen_flat.norm() * wan_flat.norm())).item()
        
        # Channel-wise correlation
        correlations = []
        for c in range(min(qwen_frame.shape[0], wan_frame.shape[0])):
            q_chan = qwen_frame[c].flatten()
            w_chan = wan_frame[c].flatten()
            if q_chan.std() > 0 and w_chan.std() > 0:
                corr = torch.corrcoef(torch.stack([q_chan, w_chan]))[0, 1].item()
                correlations.append(corr)
        avg_correlation = sum(correlations) / len(correlations) if correlations else 0
        
        report = f"""Frame Analysis Report
====================
Input Shapes:
  Qwen: {qwen_latent['samples'].shape}
  WAN:  {wan_output['samples'].shape}

Frame {frame_to_check} Comparison:
  MSE:              {mse:.6f}
  RMSE:             {rmse:.6f}
  Cosine Similarity: {cosine_sim:.4f}
  Avg Channel Corr:  {avg_correlation:.4f}

Qwen Frame Stats:
  Min:  {qwen_frame.min():.3f}
  Max:  {qwen_frame.max():.3f}
  Mean: {qwen_frame.mean():.3f}
  Std:  {qwen_frame.std():.3f}

WAN Frame Stats:
  Min:  {wan_frame.min():.3f}
  Max:  {wan_frame.max():.3f}
  Mean: {wan_frame.mean():.3f}
  Std:  {wan_frame.std():.3f}

Interpretation:
  Cosine > 0.9: Excellent similarity
  Cosine > 0.7: Good similarity
  Cosine > 0.5: Moderate similarity
  Cosine < 0.5: Poor similarity
  
  Current: {"Excellent" if cosine_sim > 0.9 else "Good" if cosine_sim > 0.7 else "Moderate" if cosine_sim > 0.5 else "Poor"}
"""
        
        return (report,)