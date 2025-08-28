"""
Minimal Keyframe V2V Implementation
Focus on technical parameters, not prompt generation
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

class MinimalKeyframeV2V:
    """
    Technical implementation of keyframe V2V
    No prompt generation fluff - just parameters that matter
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Video input
                "video_latents": ("LATENT",),  # From WAN VAE encode
                "edited_frames": ("IMAGE",),    # Qwen edited frames
                "keyframe_indices": ("STRING", {"default": "0,20,40,60,80"}),
                
                # Critical parameters
                "base_denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "keyframe_denoise": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.05}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                
                # Scheduler
                "scheduler": (["DPM-Solver++", "Euler", "Euler a", "DDIM"], {"default": "DPM-Solver++"}),
                "scheduler_type": (["karras", "normal", "simple"], {"default": "karras"}),
                
                # Denoise schedule
                "denoise_schedule": (["stepped", "linear", "exponential"], {"default": "stepped"}),
                "decay_distance": ("INT", {"default": 10, "min": 1, "max": 50}),
            },
            "optional": {
                "wan_vae": ("WAN_VAE",),  # For encoding edited frames
            }
        }
    
    RETURN_TYPES = ("LATENT", "FLOAT_ARRAY", "STRING")
    RETURN_NAMES = ("output_latents", "denoise_schedule", "technical_log")
    FUNCTION = "process"
    CATEGORY = "QwenWAN/Technical"
    
    def process(
        self,
        video_latents,
        edited_frames,
        keyframe_indices,
        base_denoise,
        keyframe_denoise,
        cfg_scale,
        steps,
        scheduler,
        scheduler_type,
        denoise_schedule,
        decay_distance,
        wan_vae=None
    ):
        """
        Process video with technical parameters only
        """
        
        # Parse keyframe indices
        indices = [int(i.strip()) for i in keyframe_indices.split(",")]
        
        # Extract video latent info
        if isinstance(video_latents, dict):
            latents = video_latents["samples"]
        else:
            latents = video_latents
            
        # Get dimensions
        if len(latents.shape) == 5:
            B, C, T, H, W = latents.shape
        else:
            # Add temporal dimension if needed
            B, C, H, W = latents.shape
            T = 1
            latents = latents.unsqueeze(2)
        
        # Build denoise schedule
        schedule = self.compute_denoise_schedule(
            T, indices, keyframe_denoise, base_denoise, 
            denoise_schedule, decay_distance
        )
        
        # Replace keyframes in latents
        if wan_vae is not None:
            latents = self.inject_edited_keyframes(
                latents, edited_frames, indices, wan_vae
            )
        
        # Create temporal mask
        mask = self.create_temporal_mask(T, H, W, indices)
        
        # Prepare output config
        config = {
            "latents": latents,
            "mask": mask,
            "denoise_schedule": schedule,
            "cfg": cfg_scale,
            "steps": steps,
            "scheduler": scheduler,
            "scheduler_type": scheduler_type
        }
        
        # Technical log
        log = self.generate_technical_log(config, indices)
        
        return ({"samples": latents}, schedule, log)
    
    def compute_denoise_schedule(
        self, 
        num_frames: int,
        keyframes: List[int],
        keyframe_denoise: float,
        base_denoise: float,
        schedule_type: str,
        decay_distance: int
    ) -> List[float]:
        """
        Compute per-frame denoise values
        """
        schedule = []
        
        for i in range(num_frames):
            if i in keyframes:
                # At keyframe - minimum denoise
                denoise = keyframe_denoise
            else:
                # Calculate distance to nearest keyframe
                distances = [abs(i - kf) for kf in keyframes]
                min_dist = min(distances) if distances else num_frames
                
                if schedule_type == "stepped":
                    # Stepped schedule
                    if min_dist <= 2:
                        denoise = keyframe_denoise + 0.1
                    elif min_dist <= 5:
                        denoise = keyframe_denoise + 0.2
                    elif min_dist <= 10:
                        denoise = base_denoise
                    else:
                        denoise = min(base_denoise + 0.2, 1.0)
                        
                elif schedule_type == "linear":
                    # Linear interpolation
                    ratio = min(1.0, min_dist / decay_distance)
                    denoise = keyframe_denoise + (base_denoise - keyframe_denoise) * ratio
                    
                elif schedule_type == "exponential":
                    # Exponential decay
                    decay_rate = 0.9
                    denoise = keyframe_denoise + (base_denoise - keyframe_denoise) * (1 - decay_rate ** min_dist)
                
            schedule.append(float(denoise))
        
        return schedule
    
    def inject_edited_keyframes(
        self,
        latents: torch.Tensor,
        edited_frames: torch.Tensor,
        indices: List[int],
        wan_vae
    ) -> torch.Tensor:
        """
        Replace latents at keyframe positions with edited frames
        """
        B, C, T, H, W = latents.shape
        
        # Process each edited frame
        for idx, frame_idx in enumerate(indices):
            if frame_idx < T:
                # Get edited frame
                if idx < edited_frames.shape[0]:
                    frame = edited_frames[idx:idx+1]
                    
                    # Encode with WAN VAE
                    encoded = wan_vae.encode(frame)
                    
                    # Ensure correct shape
                    if len(encoded.shape) == 4:
                        encoded = encoded.unsqueeze(2)
                    
                    # Replace in latents
                    if encoded.shape[2] == 1:
                        # Single frame encoded
                        latents[:, :, frame_idx] = encoded[:, :, 0]
                    else:
                        # Multiple frames - take first
                        latents[:, :, frame_idx] = encoded[:, :, 0]
        
        return latents
    
    def create_temporal_mask(
        self,
        num_frames: int,
        height: int,
        width: int,
        keyframe_indices: List[int]
    ) -> torch.Tensor:
        """
        Create mask for WAN V2V
        """
        # Base mask
        mask = torch.zeros(1, num_frames, height, width)
        
        # Mark keyframes
        for idx in keyframe_indices:
            if idx < num_frames:
                mask[0, idx] = 1.0
        
        # WAN-specific reshaping
        # First frame repeated 4x
        if num_frames > 0:
            first_frame = mask[:, 0:1]
            repeated = first_frame.repeat(1, 4, 1, 1)
            rest = mask[:, 1:]
            mask = torch.cat([repeated, rest], dim=1)
        
        # Reshape for temporal compression
        temporal_frames = (num_frames - 1) // 4 + 1
        expected_frames = temporal_frames * 4
        
        # Pad or trim to match expected size
        if mask.shape[1] < expected_frames:
            padding = torch.zeros(1, expected_frames - mask.shape[1], height, width)
            mask = torch.cat([mask, padding], dim=1)
        elif mask.shape[1] > expected_frames:
            mask = mask[:, :expected_frames]
        
        # Final reshape
        mask = mask.view(1, temporal_frames, 4, height, width)
        mask = mask.transpose(1, 2).squeeze(0)
        
        return mask
    
    def generate_technical_log(
        self,
        config: Dict,
        keyframes: List[int]
    ) -> str:
        """
        Generate technical parameter log
        """
        log = []
        log.append("=== Technical Parameters ===")
        log.append(f"Keyframes: {keyframes}")
        log.append(f"Latent shape: {config['latents'].shape}")
        log.append(f"Mask shape: {config['mask'].shape}")
        log.append(f"Scheduler: {config['scheduler']} ({config['scheduler_type']})")
        log.append(f"Steps: {config['steps']}, CFG: {config['cfg']}")
        
        # Denoise schedule summary
        schedule = config['denoise_schedule']
        log.append(f"Denoise range: {min(schedule):.2f} - {max(schedule):.2f}")
        
        # Memory estimate
        latent_memory = config['latents'].numel() * 4 / (1024**3)  # GB
        log.append(f"Latent memory: {latent_memory:.2f} GB")
        
        return "\n".join(log)


class DenoiseCurveVisualizer:
    """
    Visualize denoise schedule for debugging
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "denoise_schedule": ("FLOAT_ARRAY",),
                "keyframe_indices": ("STRING", {"default": "0,20,40,60,80"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("schedule_plot", "statistics")
    FUNCTION = "visualize"
    CATEGORY = "QwenWAN/Technical"
    
    def visualize(self, denoise_schedule, keyframe_indices):
        """
        Create visual representation of denoise schedule
        """
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image
        
        # Parse keyframes
        keyframes = [int(i.strip()) for i in keyframe_indices.split(",")]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot denoise curve
        frames = range(len(denoise_schedule))
        ax.plot(frames, denoise_schedule, 'b-', linewidth=2, label='Denoise')
        
        # Mark keyframes
        for kf in keyframes:
            if kf < len(denoise_schedule):
                ax.axvline(x=kf, color='r', linestyle='--', alpha=0.5)
                ax.scatter([kf], [denoise_schedule[kf]], color='r', s=100, zorder=5)
        
        # Formatting
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Denoise Strength')
        ax.set_title('Denoise Schedule for V2V')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Convert to image
        buf = BytesIO()
        plt.savefig(buf, format='PNG', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        # Convert to tensor
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        # Generate statistics
        stats = []
        stats.append(f"Frames: {len(denoise_schedule)}")
        stats.append(f"Keyframes: {len(keyframes)}")
        stats.append(f"Min denoise: {min(denoise_schedule):.3f}")
        stats.append(f"Max denoise: {max(denoise_schedule):.3f}")
        stats.append(f"Mean denoise: {np.mean(denoise_schedule):.3f}")
        stats.append(f"Std denoise: {np.std(denoise_schedule):.3f}")
        
        return (img_tensor, "\n".join(stats))


class LatentStatisticsMonitor:
    """
    Monitor latent statistics for debugging
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
                "label": ("STRING", {"default": "Latent"}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latents", "statistics")
    FUNCTION = "analyze"
    CATEGORY = "QwenWAN/Technical"
    
    def analyze(self, latents, label):
        """
        Analyze latent statistics
        """
        # Extract tensor
        if isinstance(latents, dict):
            tensor = latents["samples"]
        else:
            tensor = latents
        
        # Compute statistics
        stats = []
        stats.append(f"=== {label} Statistics ===")
        stats.append(f"Shape: {list(tensor.shape)}")
        stats.append(f"Dtype: {tensor.dtype}")
        stats.append(f"Device: {tensor.device}")
        stats.append(f"Min: {tensor.min().item():.4f}")
        stats.append(f"Max: {tensor.max().item():.4f}")
        stats.append(f"Mean: {tensor.mean().item():.4f}")
        stats.append(f"Std: {tensor.std().item():.4f}")
        
        # Check for NaN/Inf
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        stats.append(f"NaN: {has_nan}, Inf: {has_inf}")
        
        # Memory usage
        memory_bytes = tensor.element_size() * tensor.nelement()
        memory_mb = memory_bytes / (1024 * 1024)
        stats.append(f"Memory: {memory_mb:.2f} MB")
        
        return (latents, "\n".join(stats))


# Node registration
NODE_CLASS_MAPPINGS = {
    "MinimalKeyframeV2V": MinimalKeyframeV2V,
    "DenoiseCurveVisualizer": DenoiseCurveVisualizer,
    "LatentStatisticsMonitor": LatentStatisticsMonitor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimalKeyframeV2V": "Minimal Keyframe V2V (Technical)",
    "DenoiseCurveVisualizer": "Visualize Denoise Schedule",
    "LatentStatisticsMonitor": "Monitor Latent Statistics",
}