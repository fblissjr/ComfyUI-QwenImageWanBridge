# partially from https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/models/flux_dit.py
"""
Minimal Flux DiT components needed for Qwen Image DiT.
Extracted from DiffSynth-Studio to avoid importing the entire library.
"""

import torch
import torch.nn as nn


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization from Flux.
    Normalizes with adaptive scale and bias from conditioning.
    """

    def __init__(self, dim, single=True):
        super().__init__()
        self.single = single
        self.linear = nn.Linear(dim, dim * (2 if single else 6))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x, emb=None):
        if emb is not None:
            emb = self.linear(emb)
            if self.single:
                scale, bias = emb.chunk(2, dim=-1)
                x = self.norm(x) * (1 + scale) + bias
            else:
                # For non-single mode (used in some blocks)
                scale_msa, bias_msa, scale_mlp, bias_mlp, scale_out, bias_out = emb.chunk(6, dim=-1)
                # Return the parameters for use in the block
                return x, scale_msa, bias_msa, scale_mlp, bias_mlp, scale_out, bias_out
        else:
            x = self.norm(x)
        return x
