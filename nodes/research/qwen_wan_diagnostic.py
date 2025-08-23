"""
Diagnostic node to understand why 1 frame works but multiple frames don't
"""

import torch
import torch.nn.functional as F

class QwenWANSingleFrameDiagnostic:
    """
    Diagnostic to understand exactly why single frame works
    Test incrementally: 1, 5, 9... frames to find breaking point
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "test_mode": ([
                    "single_frame_only",
                    "repeat_frame",
                    "decay_frames", 
                    "noise_frames",
                    "interpolate_frames"
                ], {"default": "single_frame_only"}),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 81, "step": 4}),
                "debug_level": ("INT", {"default": 2, "min": 0, "max": 3}),
            }
        }
    
    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "STRING")
    RETURN_NAMES = ("image_embeds", "diagnostic_report")
    FUNCTION = "diagnose"
    CATEGORY = "QwenWANBridge/Debug"
    
    def diagnose(self, qwen_latent, test_mode, num_frames, debug_level):
        report = []
        report.append("Single Frame Diagnostic Report")
        report.append("="*50)
        
        # Extract Qwen latent
        qwen = qwen_latent["samples"]
        if len(qwen.shape) == 5:
            qwen = qwen[0, :, 0, :, :]
        elif len(qwen.shape) == 4:
            qwen = qwen[0]
        
        C, H, W = qwen.shape
        device = qwen.device
        dtype = qwen.dtype
        
        # Frame alignment
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        temporal_frames = (num_frames - 1) // 4 + 1
        
        report.append(f"Input: {qwen.shape}")
        report.append(f"Frames: {num_frames} → {temporal_frames} temporal")
        report.append(f"Test mode: {test_mode}")
        report.append("")
        
        # Analyze input statistics
        if debug_level >= 1:
            report.append("Input Statistics:")
            report.append(f"  Mean: {qwen.mean():.4f}")
            report.append(f"  Std: {qwen.std():.4f}")
            report.append(f"  Min: {qwen.min():.4f}")
            report.append(f"  Max: {qwen.max():.4f}")
            
            # Channel-wise stats
            if debug_level >= 2:
                report.append("\n  Channel stats:")
                for c in range(min(4, C)):
                    c_mean = qwen[c].mean().item()
                    c_std = qwen[c].std().item()
                    report.append(f"    Ch{c}: mean={c_mean:.3f}, std={c_std:.3f}")
        
        # Create temporal tensor based on test mode
        y = torch.zeros(C, temporal_frames, H, W, device=device, dtype=dtype)
        
        if test_mode == "single_frame_only":
            # ONLY first frame, rest are zeros
            y[:, 0] = qwen
            report.append("\nMode: Single frame only")
            report.append("  Frame 0: Qwen latent")
            report.append("  Rest: zeros")
            
        elif test_mode == "repeat_frame":
            # Repeat the same frame
            for t in range(temporal_frames):
                y[:, t] = qwen
            report.append("\nMode: Repeat frame")
            report.append(f"  All {temporal_frames} frames: identical Qwen latent")
            
        elif test_mode == "decay_frames":
            # Exponential decay
            for t in range(temporal_frames):
                decay = 0.9 ** t
                y[:, t] = qwen * decay
            report.append("\nMode: Decay frames")
            report.append(f"  Decay factor: 0.9^t")
            
        elif test_mode == "noise_frames":
            # First frame exact, rest noise
            y[:, 0] = qwen
            for t in range(1, temporal_frames):
                y[:, t] = torch.randn_like(qwen) * qwen.std()
            report.append("\nMode: Noise frames")
            report.append("  Frame 0: Qwen latent")
            report.append("  Rest: random noise")
            
        elif test_mode == "interpolate_frames":
            # Interpolate to slightly different end
            y[:, 0] = qwen
            if temporal_frames > 1:
                # Create slight variation for end
                end = qwen + torch.randn_like(qwen) * 0.1
                y[:, -1] = end
                
                # Linear interpolation
                for t in range(1, temporal_frames - 1):
                    alpha = t / (temporal_frames - 1)
                    y[:, t] = (1 - alpha) * qwen + alpha * end
            report.append("\nMode: Interpolate frames")
            report.append("  Interpolating to slight variation")
        
        # Analyze output
        if debug_level >= 1:
            report.append("\nOutput Statistics:")
            for t in range(min(3, temporal_frames)):
                frame_mean = y[:, t].mean().item()
                frame_std = y[:, t].std().item()
                report.append(f"  Frame {t}: mean={frame_mean:.3f}, std={frame_std:.3f}")
        
        # Create mask - this might be critical
        mask = torch.zeros(1, num_frames, H, W, device=device)
        
        if test_mode == "single_frame_only":
            # Only first frame has content
            mask[:, 0] = 1.0
        elif test_mode == "repeat_frame":
            # All frames have content
            mask[:, :] = 1.0
        else:
            # First frame definitely has content
            mask[:, 0] = 1.0
            # Maybe mark other frames partially?
            if debug_level >= 3:
                report.append("\nExperimental: Partial mask for other frames")
                for t in range(1, min(num_frames, 5)):
                    mask[:, t] = 0.1  # Very light mask
        
        # Reshape mask for WAN
        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        if mask.shape[1] > 1:
            mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)
        else:
            mask = start_mask_repeated
        
        # Pad/trim to correct size
        frames_needed = temporal_frames * 4
        if mask.shape[1] < frames_needed:
            padding = torch.zeros(1, frames_needed - mask.shape[1], H, W, device=device)
            mask = torch.cat([mask, padding], dim=1)
        elif mask.shape[1] > frames_needed:
            mask = mask[:, :frames_needed]
        
        mask = mask.view(1, temporal_frames, 4, H, W)
        mask = mask.movedim(1, 2)[0]
        
        if debug_level >= 2:
            report.append(f"\nMask shape: {mask.shape}")
            report.append(f"  Non-zero elements: {(mask > 0).sum().item()}")
            report.append(f"  Max value: {mask.max().item()}")
        
        # Critical insight tracking
        report.append("\n" + "="*50)
        report.append("CRITICAL INSIGHTS:")
        
        if num_frames == 1:
            report.append("• Single frame mode - WAN treats as image processing")
            report.append("• No temporal generation needed")
            report.append("• This is why it works!")
        else:
            report.append(f"• {temporal_frames} temporal frames")
            report.append("• WAN must generate temporal coherence")
            report.append("• I2V conditioning might not be recognized")
            
        report.append("\nHYPOTHESIS:")
        report.append("WAN I2V expects first frame from its own VAE.")
        report.append("Even 99.98% similar VAE has different latent distribution.")
        report.append("Single frame works because no temporal generation.")
        report.append("Multiple frames fail because I2V can't recognize conditioning.")
        
        report.append("\nRECOMMENDATION:")
        report.append("1. Try T2V models instead of I2V")
        report.append("2. Use very low denoise (0.1-0.3) to preserve structure")
        report.append("3. Or accept VAE decode/encode for now")
        
        # Create embeds
        image_embeds = {
            "image_embeds": y,
            "mask": mask,
            "num_frames": num_frames,
            "lat_h": H,
            "lat_w": W,
            "target_shape": (C, temporal_frames, H, W),
            "has_ref": False,
            "fun_or_fl2v_model": False,
            "clip_context": None,
            "negative_clip_context": None,
            "control_embeds": None,
            "add_cond_latents": None,
            "end_image": None,
            "max_seq_len": H * W // 4 * temporal_frames,
        }
        
        return (image_embeds, "\n".join(report))