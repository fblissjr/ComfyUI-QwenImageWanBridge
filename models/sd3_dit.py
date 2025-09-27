# partially from https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/models/sd3_dit.py
"""
Minimal SD3 DiT components needed for Qwen Image DiT.
Extracted from DiffSynth-Studio to avoid importing the entire library.
"""

import torch
import torch.nn as nn
import numpy as np


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones((dim,)))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(input_dtype)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        return hidden_states


# Minimal timestep embedding implementation
class TimestepEmbeddings(torch.nn.Module):
    def __init__(self, dim_in, dim_out, computation_device=None, diffusers_compatible_format=False, scale=1, align_dtype_to_timestep=False):
        super().__init__()
        self.dim_in = dim_in
        self.scale = scale

        # Simple sinusoidal embeddings (simplified from TemporalTimesteps)
        self.time_proj = None  # We'll compute inline for simplicity

        # Use the DiffSynth format
        if diffusers_compatible_format:
            self.timestep_embedder = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.SiLU(),
                nn.Linear(dim_out, dim_out)
            )
        else:
            self.timestep_embedder = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.SiLU(),
                nn.Linear(dim_out, dim_out)
            )

    def forward(self, timestep, dtype):
        # Create sinusoidal embeddings (simplified)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        # Scale timesteps
        timestep = timestep * self.scale

        # Create position embeddings
        half_dim = self.dim_in // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(-emb * torch.arange(half_dim, device=timestep.device))
        emb = timestep.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.dim_in % 2 == 1:  # odd dim
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

        # Convert to target dtype and apply MLP
        time_emb = emb.to(dtype)
        time_emb = self.timestep_embedder(time_emb)
        return time_emb
