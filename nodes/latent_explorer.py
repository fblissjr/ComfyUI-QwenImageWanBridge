"""
Latent Space Explorer
Find semantic directions in the shared Qwen/WAN latent space
"""

import torch
import numpy as np
import json
from typing import List, Tuple

class SemanticDirectionFinder:
    """Find meaningful directions in latent space"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
                "prompt_a": ("STRING", {"default": "a photo"}),
                "prompt_b": ("STRING", {"default": "a painting"}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("direction_vector", "analysis")
    FUNCTION = "find_direction"
    CATEGORY = "QwenWAN/Explorer"
    
    def find_direction(self, latent_a, latent_b, prompt_a, prompt_b):
        """Find the direction vector between two latents"""
        
        # Get samples
        a = latent_a["samples"] if isinstance(latent_a, dict) else latent_a
        b = latent_b["samples"] if isinstance(latent_b, dict) else latent_b
        
        # Calculate direction vector
        direction = b - a
        
        # Normalize to unit vector
        direction_norm = direction / (torch.norm(direction) + 1e-8)
        
        # Analyze the direction
        analysis = {
            "semantic_direction": f"{prompt_a} → {prompt_b}",
            "vector_stats": {
                "magnitude": float(torch.norm(direction)),
                "dimensions_affected": int((torch.abs(direction) > 0.01).sum()),
                "primary_channels": self._get_primary_channels(direction)
            }
        }
        
        return ({"samples": direction_norm}, json.dumps(analysis, indent=2))
    
    def _get_primary_channels(self, direction):
        """Find which channels are most affected"""
        if direction.dim() >= 2:
            channel_magnitudes = torch.abs(direction).mean(dim=tuple(range(2, direction.dim())))
            if direction.shape[0] > 0:
                channel_magnitudes = channel_magnitudes[0]
            top_channels = torch.topk(channel_magnitudes, min(3, channel_magnitudes.shape[0]))
            return [int(idx) for idx in top_channels.indices.tolist()]
        return []


class LatentWalk:
    """Walk along a semantic direction in latent space"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "direction": ("LATENT",),
                "steps": ("INT", {"default": 5, "min": 1, "max": 20}),
                "step_size": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("LATENT",) * 5  # Return up to 5 steps
    RETURN_NAMES = tuple([f"step_{i}" for i in range(5)])
    FUNCTION = "walk"
    CATEGORY = "QwenWAN/Explorer"
    
    def walk(self, latent, direction, steps, step_size):
        """Walk along a direction in latent space"""
        
        # Get samples
        base = latent["samples"] if isinstance(latent, dict) else latent
        dir_vector = direction["samples"] if isinstance(direction, dict) else direction
        
        # Generate steps
        results = []
        for i in range(min(steps, 5)):
            # Calculate position along direction
            offset = (i - steps//2) * step_size
            new_latent = base + offset * dir_vector
            results.append({"samples": new_latent})
        
        # Pad with empty results if needed
        while len(results) < 5:
            results.append({"samples": base.clone()})
        
        return tuple(results)


class LatentPCA:
    """Find principal components in a set of latents"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "latent_2": ("LATENT",),
                "latent_3": ("LATENT",),
                "latent_4": ("LATENT",),
                "latent_5": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("pc1", "pc2", "analysis")
    FUNCTION = "compute_pca"
    CATEGORY = "QwenWAN/Explorer"
    
    def compute_pca(self, latent, latent_2=None, latent_3=None, latent_4=None, latent_5=None):
        """Compute PCA on a set of latents to find main variation axes"""
        
        # Collect all latents
        latents = [latent]
        for l in [latent_2, latent_3, latent_4, latent_5]:
            if l is not None:
                latents.append(l)
        
        # Extract samples and flatten
        samples = []
        for l in latents:
            s = l["samples"] if isinstance(l, dict) else l
            samples.append(s.flatten().unsqueeze(0))
        
        if len(samples) < 2:
            # Need at least 2 samples for PCA
            dummy = torch.zeros_like(samples[0])
            return (
                {"samples": dummy.reshape(latent["samples"].shape)},
                {"samples": dummy.reshape(latent["samples"].shape)},
                json.dumps({"error": "Need at least 2 latents for PCA"})
            )
        
        # Stack samples
        X = torch.cat(samples, dim=0)
        
        # Center the data
        X_centered = X - X.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov = torch.mm(X_centered.t(), X_centered) / (X.shape[0] - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Get first two principal components
        pc1 = eigenvectors[:, 0].reshape(latent["samples"].shape)
        pc2 = eigenvectors[:, 1].reshape(latent["samples"].shape) if eigenvectors.shape[1] > 1 else torch.zeros_like(pc1)
        
        # Calculate explained variance
        total_var = eigenvalues.sum()
        explained_var = eigenvalues[:2] / total_var if total_var > 0 else torch.zeros(2)
        
        analysis = {
            "method": "PCA on latent space",
            "n_samples": len(latents),
            "explained_variance": {
                "pc1": float(explained_var[0]) if len(explained_var) > 0 else 0,
                "pc2": float(explained_var[1]) if len(explained_var) > 1 else 0
            },
            "interpretation": "PC1 and PC2 represent the main axes of variation in your latent samples"
        }
        
        return (
            {"samples": pc1},
            {"samples": pc2},
            json.dumps(analysis, indent=2)
        )


class StyleContentSeparator:
    """Attempt to separate style and content in latents"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_latent": ("LATENT",),
                "style_latent": ("LATENT",),
                "separation_mode": (["frequency", "channel", "spatial"],),
            }
        }
    
    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("content_only", "style_only", "report")
    FUNCTION = "separate"
    CATEGORY = "QwenWAN/Explorer"
    
    def separate(self, content_latent, style_latent, separation_mode):
        """Separate style and content components"""
        
        content = content_latent["samples"] if isinstance(content_latent, dict) else content_latent
        style = style_latent["samples"] if isinstance(style_latent, dict) else style_latent
        
        report = {"mode": separation_mode}
        
        if separation_mode == "frequency":
            # Use frequency domain separation
            content_only, style_only = self._frequency_separation(content, style)
            report["method"] = "FFT-based separation: low freq = content, high freq = style"
            
        elif separation_mode == "channel":
            # Use channel-wise separation
            content_only, style_only = self._channel_separation(content, style)
            report["method"] = "Channel-based: first 8 channels = content, last 8 = style"
            
        else:  # spatial
            # Use spatial statistics
            content_only, style_only = self._spatial_separation(content, style)
            report["method"] = "Spatial statistics: mean = content, variance = style"
        
        return (
            {"samples": content_only},
            {"samples": style_only},
            json.dumps(report, indent=2)
        )
    
    def _frequency_separation(self, content, style):
        """Separate using frequency domain"""
        # FFT
        content_fft = torch.fft.fft2(content, dim=(-2, -1))
        style_fft = torch.fft.fft2(style, dim=(-2, -1))
        
        # Create frequency mask (low pass for content, high pass for style)
        h, w = content.shape[-2:]
        mask = torch.zeros_like(content_fft.real)
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4
        
        # Create circular mask
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y = y.to(content.device)
        x = x.to(content.device)
        dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
        mask[..., dist <= radius] = 1.0
        
        # Apply masks
        content_only_fft = content_fft * mask
        style_only_fft = style_fft * (1 - mask)
        
        # Inverse FFT
        content_only = torch.fft.ifft2(content_only_fft, dim=(-2, -1)).real
        style_only = torch.fft.ifft2(style_only_fft, dim=(-2, -1)).real
        
        return content_only, style_only
    
    def _channel_separation(self, content, style):
        """Separate using channel groups"""
        if content.shape[1] >= 16:
            # First half channels for content, second half for style
            mid = content.shape[1] // 2
            content_only = content.clone()
            content_only[:, mid:] = 0
            
            style_only = style.clone()
            style_only[:, :mid] = 0
        else:
            # Fallback for fewer channels
            content_only = content
            style_only = style
        
        return content_only, style_only
    
    def _spatial_separation(self, content, style):
        """Separate using spatial statistics"""
        # Content = spatial structure (keep mean)
        content_mean = content.mean(dim=(-2, -1), keepdim=True)
        content_only = content_mean.expand_as(content)
        
        # Style = spatial variation (keep variance)
        style_std = style.std(dim=(-2, -1), keepdim=True)
        style_normalized = (style - style.mean(dim=(-2, -1), keepdim=True)) / (style_std + 1e-8)
        style_only = style_normalized * style_std
        
        return content_only, style_only