"""
Native ComfyUI Bridge for Qwen to WAN
Handles batch dimensions and noise properly for native KSampler
"""

import torch
import torch.nn.functional as F

class QwenWANNativeBridge:
    """
    Bridge for native ComfyUI (not Kijai's wrapper)
    Handles batch dimension (B, C, T, H, W) and noise options
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 9, "min": 1, "max": 1024, "step": 4}),
                "noise_mode": ([
                    "no_noise",           # Pure Qwen latent
                    "add_noise",          # Add noise to Qwen
                    "mix_noise",          # Mix Qwen with noise
                    "replace_with_noise", # Full noise (for testing)
                    "scaled_noise",       # Scale noise by frame distance
                    "reference_mode",     # Qwen as reference/phantom
                    "vace_style"          # VACE-like reference conditioning
                ], {"default": "no_noise"}),
                "noise_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "info")
    FUNCTION = "bridge"
    CATEGORY = "QwenWANBridge/Native"
    
    def bridge(self, qwen_latent, width, height, num_frames, 
               noise_mode, noise_strength, seed):
        
        info = []
        info.append("Native ComfyUI Bridge")
        info.append("="*40)
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Extract Qwen latent
        qwen = qwen_latent["samples"]
        
        # Handle input shape - native ComfyUI expects batch
        if len(qwen.shape) == 3:
            # (C, H, W) -> add batch and temporal
            qwen = qwen.unsqueeze(0).unsqueeze(2)  # (1, C, 1, H, W)
        elif len(qwen.shape) == 4:
            # (B, C, H, W) -> add temporal
            if qwen.shape[0] > 1:
                qwen = qwen[0:1]  # Take first batch only
            qwen = qwen.unsqueeze(2)  # (1, C, 1, H, W)
        elif len(qwen.shape) == 5:
            # Already (B, C, T, H, W)
            if qwen.shape[0] > 1:
                qwen = qwen[0:1]
            if qwen.shape[2] > 1:
                qwen = qwen[:, :, 0:1, :, :]  # Take first frame
        
        B, C, _, H_orig, W_orig = qwen.shape
        device = qwen.device
        dtype = qwen.dtype
        
        info.append(f"Input shape: {tuple(qwen.shape)}")
        
        # Calculate target dimensions
        lat_h = height // 8
        lat_w = width // 8
        
        # Resize if needed
        if (H_orig, W_orig) != (lat_h, lat_w):
            # Reshape for interpolation: (B*T, C, H, W)
            qwen_2d = qwen.squeeze(2)  # Remove temporal
            qwen_resized = F.interpolate(
                qwen_2d,
                size=(lat_h, lat_w),
                mode='bilinear',
                align_corners=False
            )
            qwen = qwen_resized.unsqueeze(2)  # Add temporal back
            info.append(f"Resized to: {lat_h}x{lat_w}")
        
        # Frame alignment
        num_frames_aligned = ((num_frames - 1) // 4) * 4 + 1
        temporal_frames = (num_frames_aligned - 1) // 4 + 1
        
        info.append(f"Frames: {num_frames} → {temporal_frames} temporal")
        info.append(f"Noise mode: {noise_mode}")
        
        # Extract single frame from Qwen
        qwen_frame = qwen[:, :, 0, :, :]  # (B, C, H, W)
        
        # Create temporal tensor WITH BATCH DIMENSION
        latent = torch.zeros(B, C, temporal_frames, lat_h, lat_w, 
                            device=device, dtype=dtype)
        
        # Apply noise based on mode
        if noise_mode == "no_noise":
            # Pure Qwen in first frame
            latent[:, :, 0] = qwen_frame.squeeze(0) if qwen_frame.shape[0] == 1 else qwen_frame[0]
            info.append("Frame 0: Pure Qwen latent")
            
        elif noise_mode == "add_noise":
            # Add noise to Qwen
            noise = torch.randn_like(qwen_frame) * noise_strength
            latent[:, :, 0] = (qwen_frame + noise).squeeze(0) if qwen_frame.shape[0] == 1 else (qwen_frame + noise)[0]
            info.append(f"Frame 0: Qwen + {noise_strength:.0%} noise")
            
        elif noise_mode == "mix_noise":
            # Mix Qwen with noise
            noise = torch.randn_like(qwen_frame)
            alpha = noise_strength
            mixed = (1 - alpha) * qwen_frame + alpha * noise
            latent[:, :, 0] = mixed.squeeze(0) if mixed.shape[0] == 1 else mixed[0]
            info.append(f"Frame 0: {100-noise_strength*100:.0f}% Qwen, {noise_strength*100:.0f}% noise")
            
        elif noise_mode == "replace_with_noise":
            # Full noise (for testing if noise alone works)
            latent = torch.randn(B, C, temporal_frames, lat_h, lat_w, 
                                device=device, dtype=dtype)
            info.append("All frames: Pure noise (testing)")
            
        elif noise_mode == "scaled_noise":
            # Scale noise by temporal distance
            latent[:, :, 0] = qwen_frame.squeeze(0) if qwen_frame.shape[0] == 1 else qwen_frame[0]
            for t in range(1, temporal_frames):
                # Increasing noise over time
                t_alpha = (t / temporal_frames) * noise_strength
                noise = torch.randn(C, lat_h, lat_w, device=device, dtype=dtype)
                latent[:, :, t] = (1 - t_alpha) * latent[:, :, 0] + t_alpha * noise
            info.append(f"Scaled noise: 0% → {noise_strength*100:.0f}% over time")
            
        elif noise_mode == "reference_mode":
            # Qwen as reference/phantom - subtle influence throughout
            # Start with noise but blend in Qwen structure
            latent = torch.randn(B, C, temporal_frames, lat_h, lat_w, 
                                device=device, dtype=dtype)
            
            # Blend Qwen into all frames as subtle reference
            qwen_influence = noise_strength * 0.3  # Subtle influence
            for t in range(temporal_frames):
                # Decreasing influence over time
                t_factor = 1.0 - (t / max(temporal_frames - 1, 1)) * 0.5
                latent[:, :, t] = (1 - qwen_influence * t_factor) * latent[:, :, t] + \
                                 (qwen_influence * t_factor) * qwen_frame.squeeze(0)
            info.append(f"Reference mode: Qwen as {noise_strength*30:.0f}% phantom influence")
            
        elif noise_mode == "vace_style":
            # VACE-like: Strong reference at key points, interpolated
            latent = torch.randn(B, C, temporal_frames, lat_h, lat_w, 
                                device=device, dtype=dtype) * 0.1  # Start with weak noise
            
            # Place Qwen as reference at strategic points
            latent[:, :, 0] = qwen_frame.squeeze(0)  # Strong at start
            
            if temporal_frames > 1:
                # Add reference points
                mid = temporal_frames // 2
                latent[:, :, mid] = qwen_frame.squeeze(0) * (1 - noise_strength) + \
                                   torch.randn_like(qwen_frame.squeeze(0)) * noise_strength
                
                if temporal_frames > 2:
                    # Interpolate between references
                    for t in range(1, mid):
                        alpha = t / mid
                        latent[:, :, t] = (1 - alpha) * latent[:, :, 0] + alpha * latent[:, :, mid]
                    
                    for t in range(mid + 1, temporal_frames):
                        alpha = (t - mid) / (temporal_frames - mid)
                        end_noise = torch.randn(C, lat_h, lat_w, device=device, dtype=dtype)
                        latent[:, :, t] = (1 - alpha) * latent[:, :, mid] + alpha * end_noise * 0.5
            
            info.append(f"VACE-style: Reference points with interpolation")
        
        # Add noise to other frames if needed (for multi-frame)
        if noise_mode != "replace_with_noise" and temporal_frames > 1:
            if noise_mode in ["no_noise", "add_noise", "mix_noise"]:
                # Other frames get decreasing amounts of Qwen influence
                for t in range(1, min(3, temporal_frames)):
                    decay = 0.5 ** t
                    latent[:, :, t] = latent[:, :, 0] * decay * 0.1
                info.append(f"Frames 1-{min(3, temporal_frames)-1}: Decay from frame 0")
        
        # Statistics
        info.append(f"\nOutput shape: {latent.shape}")
        info.append(f"Stats: mean={latent.mean():.3f}, std={latent.std():.3f}")
        info.append(f"  Frame 0: mean={latent[:,:,0].mean():.3f}, std={latent[:,:,0].std():.3f}")
        
        # Important notes for native ComfyUI
        info.append("\nNative ComfyUI Notes:")
        info.append("• Shape is (B, C, T, H, W) with batch")
        info.append("• Use with native KSampler")
        info.append("• Connect to 'latent_image' input")
        info.append("• For I2V: Use low denoise (0.2-0.5)")
        info.append("• For T2V: Text drives generation more than latent")
        
        # Return in ComfyUI LATENT format
        return ({"samples": latent}, "\n".join(info))


class QwenWANNoiseAnalyzer:
    """
    Analyze the effect of noise on Qwen latents
    Helps understand why pixelation occurs
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original": ("LATENT",),
                "processed": ("LATENT",),
                "analysis_mode": ([
                    "statistics",
                    "distribution",
                    "frequency",
                    "correlation"
                ], {"default": "statistics"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze"
    CATEGORY = "QwenWANBridge/Native"
    
    def analyze(self, original, processed, analysis_mode):
        
        orig = original["samples"]
        proc = processed["samples"]
        
        # Ensure same shape for comparison
        if orig.shape != proc.shape:
            # Resize to match
            if len(orig.shape) < len(proc.shape):
                while len(orig.shape) < len(proc.shape):
                    orig = orig.unsqueeze(2)  # Add temporal
            if orig.shape[-2:] != proc.shape[-2:]:
                orig = F.interpolate(
                    orig.view(-1, orig.shape[-3], orig.shape[-2], orig.shape[-1]),
                    size=proc.shape[-2:],
                    mode='bilinear'
                ).view(*proc.shape)
        
        report = []
        report.append(f"Noise Analysis ({analysis_mode})")
        report.append("="*50)
        
        if analysis_mode == "statistics":
            report.append("\nBasic Statistics:")
            report.append(f"Original: mean={orig.mean():.4f}, std={orig.std():.4f}")
            report.append(f"Processed: mean={proc.mean():.4f}, std={proc.std():.4f}")
            
            diff = (proc - orig).abs()
            report.append(f"\nDifference:")
            report.append(f"  Mean absolute: {diff.mean():.4f}")
            report.append(f"  Max: {diff.max():.4f}")
            report.append(f"  Min: {diff.min():.4f}")
            
            # Signal-to-noise ratio
            signal_power = (orig ** 2).mean()
            noise_power = ((proc - orig) ** 2).mean()
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
            report.append(f"\nSNR: {snr:.2f} dB")
            
        elif analysis_mode == "distribution":
            report.append("\nDistribution Analysis:")
            
            # Histogram-like analysis
            orig_flat = orig.flatten()
            proc_flat = proc.flatten()
            
            # Percentiles
            for p in [0, 25, 50, 75, 100]:
                orig_p = torch.quantile(orig_flat, p/100.0).item()
                proc_p = torch.quantile(proc_flat, p/100.0).item()
                report.append(f"  {p:3d}%: orig={orig_p:+.3f}, proc={proc_p:+.3f}")
            
            # Check for clipping
            orig_clip = ((orig_flat < -3) | (orig_flat > 3)).float().mean()
            proc_clip = ((proc_flat < -3) | (proc_flat > 3)).float().mean()
            report.append(f"\nClipping (|x|>3):")
            report.append(f"  Original: {orig_clip*100:.1f}%")
            report.append(f"  Processed: {proc_clip*100:.1f}%")
            
        elif analysis_mode == "frequency":
            report.append("\nFrequency Analysis:")
            
            # Simple frequency analysis via FFT
            if len(orig.shape) >= 4:
                # Take first batch, first channel, first frame
                orig_2d = orig[0, 0, 0] if len(orig.shape) == 5 else orig[0, 0]
                proc_2d = proc[0, 0, 0] if len(proc.shape) == 5 else proc[0, 0]
                
                # 2D FFT
                orig_fft = torch.fft.fft2(orig_2d).abs()
                proc_fft = torch.fft.fft2(proc_2d).abs()
                
                # Low vs high frequency energy
                h, w = orig_fft.shape
                center_h, center_w = h // 4, w // 4
                
                # Low frequency (center)
                orig_low = orig_fft[h//2-center_h:h//2+center_h, 
                                   w//2-center_w:w//2+center_w].mean()
                proc_low = proc_fft[h//2-center_h:h//2+center_h,
                                   w//2-center_w:w//2+center_w].mean()
                
                # High frequency (edges)
                orig_high = orig_fft.mean() - orig_low
                proc_high = proc_fft.mean() - proc_low
                
                report.append(f"  Low freq energy:")
                report.append(f"    Original: {orig_low:.3f}")
                report.append(f"    Processed: {proc_low:.3f}")
                report.append(f"  High freq energy:")
                report.append(f"    Original: {orig_high:.3f}")
                report.append(f"    Processed: {proc_high:.3f}")
                
                if proc_high > orig_high * 1.5:
                    report.append("\n⚠️ High frequency noise detected!")
                    report.append("This may cause pixelation/artifacts")
                    
        elif analysis_mode == "correlation":
            report.append("\nCorrelation Analysis:")
            
            # Flatten for correlation
            orig_flat = orig.flatten()
            proc_flat = proc.flatten()
            
            # Pearson correlation
            corr = torch.corrcoef(torch.stack([orig_flat, proc_flat]))[0, 1].item()
            report.append(f"  Pearson correlation: {corr:.4f}")
            
            # Interpretation
            if corr > 0.95:
                report.append("  → Very high correlation (minimal change)")
            elif corr > 0.8:
                report.append("  → High correlation (structure preserved)")
            elif corr > 0.5:
                report.append("  → Moderate correlation (significant change)")
            else:
                report.append("  → Low correlation (mostly noise)")
            
            # Channel-wise correlation
            if orig.shape[1] >= 4:  # At least 4 channels
                report.append("\n  Channel correlations:")
                for c in range(min(4, orig.shape[1])):
                    c_orig = orig[:, c].flatten()
                    c_proc = proc[:, c].flatten()
                    c_corr = torch.corrcoef(torch.stack([c_orig, c_proc]))[0, 1].item()
                    report.append(f"    Ch{c}: {c_corr:.3f}")
        
        # Recommendations
        report.append("\nRecommendations:")
        if analysis_mode == "statistics" and snr < 10:
            report.append("• Low SNR - reduce noise strength")
        if analysis_mode == "frequency" and proc_high > orig_high * 1.5:
            report.append("• High frequency noise - try 'mix_noise' mode")
        if analysis_mode == "correlation" and corr < 0.5:
            report.append("• Low correlation - latent structure lost")
            report.append("• Try 'add_noise' with low strength (0.05-0.1)")
        
        return ("\n".join(report),)