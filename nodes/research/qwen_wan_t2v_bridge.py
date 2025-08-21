"""
T2V Bridge - Use WAN T2V models with Qwen as target
Instead of I2V (image to video), use T2V with Qwen as the generation target
"""

import torch
import torch.nn.functional as F

class QwenWANT2VBridge:
    """
    Use WAN's T2V models with Qwen latent as the target/guide
    T2V models are more flexible and might handle the latent better
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 1024, "step": 4}),
                "injection_method": (["target", "init_noise", "hybrid"], {"default": "target"}),
                "injection_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "WANVIDIMAGE_EMBEDS", "STRING")
    RETURN_NAMES = ("samples", "target_embeds", "info")
    FUNCTION = "bridge"
    CATEGORY = "QwenWANBridge"
    
    def bridge(self, qwen_latent, width, height, num_frames, 
               injection_method, injection_strength):
        
        info = []
        info.append("T2V Bridge Mode")
        info.append("="*40)
        
        # Extract and prepare Qwen latent
        qwen = qwen_latent["samples"]
        if len(qwen.shape) == 5:
            qwen = qwen[0, :, 0, :, :]
        elif len(qwen.shape) == 4:
            qwen = qwen[0]
        
        C, H_orig, W_orig = qwen.shape
        device = qwen.device
        dtype = qwen.dtype
        
        # Target dimensions
        lat_h = height // 8
        lat_w = width // 8
        
        # Resize Qwen to target
        if (H_orig, W_orig) != (lat_h, lat_w):
            qwen = F.interpolate(
                qwen.unsqueeze(0),
                size=(lat_h, lat_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            info.append(f"Resized: {H_orig}x{W_orig} → {lat_h}x{lat_w}")
        
        # Align frames
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        temporal_frames = (num_frames - 1) // 4 + 1
        
        info.append(f"Frames: {num_frames} (temporal: {temporal_frames})")
        info.append(f"Injection: {injection_method} @ {injection_strength:.0%}")
        
        # Create base latent for T2V
        if injection_method == "target":
            # T2V models expect empty latent, we'll use Qwen as target
            samples = torch.randn(1, C, temporal_frames, lat_h, lat_w, 
                                device=device, dtype=dtype)
            
            # Create target structure with Qwen
            target = torch.zeros(C, temporal_frames, lat_h, lat_w, 
                               device=device, dtype=dtype)
            target[:, -1] = qwen  # Place Qwen as end target
            
            # T2V will generate towards this target
            info.append("Qwen set as generation target (end frame)")
            
        elif injection_method == "init_noise":
            # Mix Qwen with initial noise
            noise = torch.randn(1, C, temporal_frames, lat_h, lat_w,
                              device=device, dtype=dtype)
            
            # Inject Qwen into first frame of noise
            noise[0, :, 0] = (1 - injection_strength) * noise[0, :, 0] + \
                            injection_strength * qwen
            samples = noise
            
            # No explicit target for T2V
            target = None
            info.append("Qwen mixed into initial noise")
            
        elif injection_method == "hybrid":
            # Use Qwen both as init and target
            samples = torch.randn(1, C, temporal_frames, lat_h, lat_w,
                                device=device, dtype=dtype)
            
            # Subtle injection into noise
            samples[0, :, 0] = samples[0, :, 0] * (1 - injection_strength * 0.3) + \
                              qwen * injection_strength * 0.3
            
            # Also set as target
            target = torch.zeros(C, temporal_frames, lat_h, lat_w,
                               device=device, dtype=dtype)
            target[:, -1] = qwen * injection_strength
            
            info.append("Hybrid: Qwen in both init and target")
        
        # For T2V mode, we need different structure
        if target is not None:
            # Create target embedding structure for T2V
            target_embeds = {
                "target_shape": (C, temporal_frames, lat_h, lat_w),
                "target_latent": target,  # Optional target guidance
                "num_frames": num_frames,
                "lat_h": lat_h,
                "lat_w": lat_w,
                # T2V specific
                "is_t2v": True,
                "has_target": True,
            }
        else:
            # Minimal structure for T2V
            target_embeds = {
                "target_shape": (C, temporal_frames, lat_h, lat_w),
                "num_frames": num_frames,
                "lat_h": lat_h,
                "lat_w": lat_w,
                "is_t2v": True,
                "has_target": False,
            }
        
        # Statistics
        info.append("\nLatent statistics:")
        info.append(f"  Qwen: mean={qwen.mean():.3f}, std={qwen.std():.3f}")
        info.append(f"  Samples: mean={samples.mean():.3f}, std={samples.std():.3f}")
        
        info_text = "\n".join(info)
        
        # Return samples for T2V sampling
        return ({"samples": samples}, target_embeds, info_text)


class QwenWANNoiseScheduler:
    """
    Control noise scheduling for Qwen→WAN transfer
    Critical for getting good results
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "denoise_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "denoise_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "sampler": (["DPM-Solver++", "DDIM", "Euler", "Euler a"], {"default": "DPM-Solver++"}),
                "scheduler": (["karras", "normal", "simple", "ddim_uniform"], {"default": "karras"}),
                "test_mode": (["single_frame", "short_video", "full_video"], {"default": "single_frame"}),
            }
        }
    
    RETURN_TYPES = ("NOISE_SCHEDULE", "STRING")
    RETURN_NAMES = ("schedule", "recommendations")
    FUNCTION = "create_schedule"
    CATEGORY = "QwenWANBridge"
    
    def create_schedule(self, denoise_start, denoise_end, cfg_scale, 
                       steps, sampler, scheduler, test_mode):
        
        rec = []
        rec.append("Noise Schedule Recommendations")
        rec.append("="*40)
        
        # Adjust based on test mode
        if test_mode == "single_frame":
            # Conservative settings for testing
            actual_denoise = min(denoise_start, 0.5)
            actual_cfg = min(cfg_scale, 5.0)
            actual_steps = min(steps, 10)
            rec.append("Single Frame Test Mode:")
            rec.append("  Using conservative settings")
            rec.append(f"  Denoise: {actual_denoise:.1f} (capped at 0.5)")
            rec.append(f"  CFG: {actual_cfg:.1f} (capped at 5.0)")
            rec.append(f"  Steps: {actual_steps} (capped at 10)")
            
        elif test_mode == "short_video":
            # Balanced settings
            actual_denoise = denoise_start * 0.8
            actual_cfg = cfg_scale
            actual_steps = steps
            rec.append("Short Video Mode:")
            rec.append("  Balanced quality/speed")
            rec.append(f"  Denoise: {actual_denoise:.1f}")
            rec.append(f"  CFG: {actual_cfg:.1f}")
            rec.append(f"  Steps: {actual_steps}")
            
        else:  # full_video
            # Full quality
            actual_denoise = denoise_start
            actual_cfg = cfg_scale
            actual_steps = steps
            rec.append("Full Video Mode:")
            rec.append("  Maximum quality")
            rec.append(f"  Denoise: {actual_denoise:.1f}")
            rec.append(f"  CFG: {actual_cfg:.1f}")
            rec.append(f"  Steps: {actual_steps}")
        
        rec.append(f"\nSampler: {sampler}")
        rec.append(f"Scheduler: {scheduler}")
        
        # Specific recommendations for Qwen→WAN
        rec.append("\nQwen→WAN Specific Tips:")
        rec.append("• Start with denoise=0.3-0.5 to preserve Qwen structure")
        rec.append("• Lower CFG (3-5) often works better than high")
        rec.append("• DPM-Solver++ with karras is most stable")
        rec.append("• For T2V models, try denoise=1.0")
        rec.append("• For I2V models, try denoise=0.3-0.7")
        
        # Create schedule dict (for use by sampler)
        schedule = {
            "denoise": actual_denoise,
            "cfg_scale": actual_cfg,
            "steps": actual_steps,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise_start": denoise_start,
            "denoise_end": denoise_end,
        }
        
        return (schedule, "\n".join(rec))