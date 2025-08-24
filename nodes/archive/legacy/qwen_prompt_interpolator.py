"""
Qwen Prompt Interpolator
Based on DiffSynth's semantic interpolation implementation
Allows smooth blending between text prompts at the embedding level
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from typing import List, Tuple, Dict, Optional
import folder_paths
import os
import comfy.model_management
import comfy.utils

class TemporalTimesteps(nn.Module):
    """Temporal timestep encoding from DiffSynth"""
    def __init__(self, num_channels: int, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = self.get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb

    @staticmethod
    def get_timestep_embedding(
        timesteps: torch.Tensor,
        embedding_dim: int,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 1,
        scale: float = 1,
        max_period: int = 10000,
    ):
        half_dim = embedding_dim // 2
        exponent = -np.log(max_period) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - downscale_freq_shift)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

        return emb


class ValueEncoder(nn.Module):
    """Value encoder for interpolation from DiffSynth"""
    def __init__(self, dim_in=256, dim_out=3584, value_emb_length=32):
        super().__init__()
        self.value_emb = TemporalTimesteps(num_channels=dim_in, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.positional_emb = nn.Parameter(torch.randn(1, value_emb_length, dim_out))
        self.proj_value = nn.Linear(dim_in, dim_out)
        self.proj_out = nn.Linear(dim_out, dim_out)
        self.value_emb_length = value_emb_length

    def forward(self, value):
        value = value * 1000  # Scale for better encoding
        emb = self.value_emb(value).to(value.dtype)
        emb = self.proj_value(emb)
        emb = repeat(emb, "b d -> b s d", s=self.value_emb_length)
        emb = emb + self.positional_emb.to(dtype=emb.dtype, device=emb.device)
        emb = torch.nn.functional.silu(emb)
        emb = self.proj_out(emb)
        return emb


class TextInterpolationModel(nn.Module):
    """Text interpolation model from DiffSynth"""
    def __init__(self, dim_in=256, dim_out=3584, value_emb_length=32, num_heads=28):
        super().__init__()
        self.to_q = ValueEncoder(dim_in=dim_in, dim_out=dim_out, value_emb_length=value_emb_length)
        self.xk_emb = nn.Parameter(torch.randn(1, 1, dim_out))
        self.yk_emb = nn.Parameter(torch.randn(1, 1, dim_out))
        self.xv_emb = nn.Parameter(torch.randn(1, 1, dim_out))
        self.yv_emb = nn.Parameter(torch.randn(1, 1, dim_out))
        self.to_k = nn.Linear(dim_out, dim_out, bias=False)
        self.to_v = nn.Linear(dim_out, dim_out, bias=False)
        self.to_out = nn.Linear(dim_out, dim_out)
        self.num_heads = num_heads

    def forward(self, value, x, y):
        q = self.to_q(value)
        k = self.to_k(torch.concat([x + self.xk_emb, y + self.yk_emb], dim=1))
        v = self.to_v(torch.concat([x + self.xv_emb, y + self.yv_emb], dim=1))
        
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h s d -> b s (h d)')
        out = self.to_out(out)
        return out


class QwenPromptInterpolator:
    """
    Interpolate between two prompts using learned semantic blending
    Based on DiffSynth's qwen-image-interpolate implementation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt_a": ("STRING", {"multiline": True}),
                "prompt_b": ("STRING", {"multiline": True}),
                "interpolation_value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "models/qwen_interpolate.pth"}),
                "train_mode": ("BOOLEAN", {"default": False}),
                "num_training_steps": ("INT", {"default": 1000, "min": 100, "max": 100000}),
                "learning_rate": ("FLOAT", {"default": 1e-5, "min": 1e-6, "max": 1e-3}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "DICT")
    RETURN_NAMES = ("conditioning", "interpolation_info")
    FUNCTION = "interpolate"
    CATEGORY = "QwenImage/Interpolation"
    
    def __init__(self):
        self.model = None
        self.device = comfy.model_management.get_torch_device()
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    def load_or_create_model(self, model_path: str) -> TextInterpolationModel:
        """Load existing model or create new one"""
        if self.model is not None:
            return self.model
        
        self.model = TextInterpolationModel(
            dim_in=256,
            dim_out=3584,  # Qwen2.5-VL embedding dimension
            value_emb_length=32,
            num_heads=28  # Qwen attention heads
        ).to(dtype=self.dtype, device=self.device)
        
        # Try to load existing weights
        if os.path.exists(model_path):
            try:
                state_dict = comfy.utils.load_torch_file(model_path)
                self.model.load_state_dict(state_dict)
                print(f"[QwenInterpolator] Loaded model from {model_path}")
            except Exception as e:
                print(f"[QwenInterpolator] Failed to load model: {e}")
        else:
            print(f"[QwenInterpolator] Created new model (no weights found at {model_path})")
        
        return self.model
    
    def encode_prompt(self, clip, prompt: str) -> torch.Tensor:
        """Encode prompt to embedding"""
        # Use Qwen template
        template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        formatted = template.format(prompt)
        
        tokens = clip.tokenize(formatted)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        # Extract embedding
        if isinstance(conditioning, list) and len(conditioning) > 0:
            embedding = conditioning[0][0]
        else:
            embedding = conditioning
        
        return embedding
    
    def sample_tokens(self, emb: torch.Tensor, p: float) -> torch.Tensor:
        """Sample tokens for training (from DiffSynth)"""
        if p <= 0:
            return torch.zeros(emb.shape[0], 0, emb.shape[2], device=emb.device, dtype=emb.dtype)
        perm = torch.randperm(emb.shape[1])[:max(1, int(emb.shape[1] * p))]
        return emb[:, perm]
    
    def loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Chamfer distance loss for training"""
        s, l = x.shape[1], y.shape[1]
        if s == 0 or l == 0:
            return torch.tensor(0.0, device=x.device)
        
        x = repeat(x, "b s d -> b s l d", l=l)
        y = repeat(y, "b l d -> b s l d", s=s)
        d = torch.square(x - y).mean(dim=-1)
        loss_x = d.min(dim=1).values.mean()
        loss_y = d.min(dim=2).values.mean()
        return loss_x + loss_y
    
    def train_step(self, model: TextInterpolationModel, x: torch.Tensor, y: torch.Tensor,
                  optimizer: torch.optim.Optimizer) -> float:
        """Single training step"""
        optimizer.zero_grad()
        
        # Random interpolation value
        value = torch.rand((1,), dtype=self.dtype, device=self.device)
        
        # Forward pass
        out = model(value, x, y)
        
        # Compute loss
        loss = self.loss_fn(out, x) * (1 - value) + self.loss_fn(out, y) * value
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def interpolate(self, clip, prompt_a: str, prompt_b: str, interpolation_value: float,
                   model_path: str = "models/qwen_interpolate.pth",
                   train_mode: bool = False, num_training_steps: int = 1000,
                   learning_rate: float = 1e-5) -> Tuple[list, Dict]:
        """
        Interpolate between two prompts
        """
        interpolation_info = {
            "prompt_a": prompt_a[:50] + "..." if len(prompt_a) > 50 else prompt_a,
            "prompt_b": prompt_b[:50] + "..." if len(prompt_b) > 50 else prompt_b,
            "value": interpolation_value,
            "model_path": model_path
        }
        
        # Encode prompts
        with torch.no_grad():
            emb_a = self.encode_prompt(clip, prompt_a).to(dtype=self.dtype, device=self.device)
            emb_b = self.encode_prompt(clip, prompt_b).to(dtype=self.dtype, device=self.device)
        
        interpolation_info["embedding_shapes"] = {
            "a": list(emb_a.shape),
            "b": list(emb_b.shape)
        }
        
        # Load or create model
        model = self.load_or_create_model(model_path)
        
        # Training mode
        if train_mode:
            print(f"[QwenInterpolator] Training for {num_training_steps} steps...")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            losses = []
            for step in range(num_training_steps):
                loss = self.train_step(model, emb_a, emb_b, optimizer)
                losses.append(loss)
                
                if (step + 1) % 100 == 0:
                    avg_loss = np.mean(losses[-100:])
                    print(f"[QwenInterpolator] Step {step + 1}/{num_training_steps}, Loss: {avg_loss:.6f}")
            
            # Save model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"[QwenInterpolator] Saved model to {model_path}")
            
            interpolation_info["trained"] = True
            interpolation_info["final_loss"] = np.mean(losses[-100:]) if losses else 0
        
        # Interpolation
        with torch.no_grad():
            value_tensor = torch.tensor([interpolation_value], dtype=self.dtype, device=self.device)
            interpolated = model(value_tensor, emb_a, emb_b)
        
        interpolation_info["interpolated_shape"] = list(interpolated.shape)
        
        # Create conditioning
        conditioning = [[interpolated, {}]]
        
        return (conditioning, interpolation_info)


class QwenPromptInterpolatorBatch:
    """
    Batch interpolation for creating smooth transitions
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt_a": ("STRING", {"multiline": True}),
                "prompt_b": ("STRING", {"multiline": True}),
                "num_steps": ("INT", {"default": 5, "min": 2, "max": 100}),
                "curve": (["linear", "ease_in", "ease_out", "ease_in_out"], {"default": "linear"}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "models/qwen_interpolate.pth"}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning_batch",)
    FUNCTION = "interpolate_batch"
    CATEGORY = "QwenImage/Interpolation"
    
    def apply_curve(self, t: float, curve: str) -> float:
        """Apply easing curve to interpolation value"""
        if curve == "ease_in":
            return t * t
        elif curve == "ease_out":
            return 1 - (1 - t) * (1 - t)
        elif curve == "ease_in_out":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - 2 * (1 - t) * (1 - t)
        else:  # linear
            return t
    
    def interpolate_batch(self, clip, prompt_a: str, prompt_b: str, num_steps: int,
                         curve: str = "linear", model_path: str = "models/qwen_interpolate.pth"):
        """
        Create batch of interpolated conditionings
        """
        interpolator = QwenPromptInterpolator()
        
        # Generate interpolation values
        values = []
        for i in range(num_steps):
            t = i / (num_steps - 1)
            t = self.apply_curve(t, curve)
            values.append(t)
        
        # Generate conditionings
        batch_conditioning = []
        for value in values:
            cond, _ = interpolator.interpolate(clip, prompt_a, prompt_b, value, model_path)
            batch_conditioning.extend(cond)
        
        return (batch_conditioning,)


NODE_CLASS_MAPPINGS = {
    "QwenPromptInterpolator": QwenPromptInterpolator,
    "QwenPromptInterpolatorBatch": QwenPromptInterpolatorBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenPromptInterpolator": "Qwen Prompt Interpolator",
    "QwenPromptInterpolatorBatch": "Qwen Prompt Interpolator Batch",
}