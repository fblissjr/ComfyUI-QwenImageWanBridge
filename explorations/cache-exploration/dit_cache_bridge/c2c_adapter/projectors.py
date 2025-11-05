"""
KV Cache Projectors

Implements various strategies for projecting and fusing KV caches
between different models.

Based on C2C paper projector designs, adapted for DiT models.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class BaseProjector(nn.Module):
    """
    Base class for KV cache projectors.

    All projectors transform source KV caches to match target model's format
    and optionally fuse them with target's own KV caches.
    """

    def __init__(
        self,
        source_heads: int,
        target_heads: int,
        source_head_dim: int,
        target_head_dim: int,
    ):
        super().__init__()
        self.source_heads = source_heads
        self.target_heads = target_heads
        self.source_head_dim = source_head_dim
        self.target_head_dim = target_head_dim

    def forward(
        self,
        source_kv: torch.Tensor,
        target_kv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project and fuse KV caches.

        Args:
            source_kv: Source model KV cache (B, H_s, N, D_s)
            target_kv: Target model KV cache (B, H_t, N, D_t)

        Returns:
            Fused KV cache (B, H_t, N, D_t)
        """
        raise NotImplementedError


class IdentityProjector(BaseProjector):
    """
    Identity projector for architecturally identical models.

    Simply returns the source KV cache when models have matching
    head counts and dimensions. Perfect for Wan2.1 ↔ ChronoEdit.
    """

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__(num_heads, num_heads, head_dim, head_dim)

    def forward(
        self,
        source_kv: torch.Tensor,
        target_kv: torch.Tensor,
    ) -> torch.Tensor:
        """Simply return source KV (models are identical)."""
        assert source_kv.shape == target_kv.shape, \
            f"Shape mismatch: {source_kv.shape} vs {target_kv.shape}"
        return source_kv


class WeightedFusionProjector(BaseProjector):
    """
    Simple weighted fusion of source and target KV caches.

    Combines caches using a learnable or fixed weight alpha:
        output = alpha * source + (1 - alpha) * target
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        alpha: float = 0.5,
        learnable: bool = False,
    ):
        super().__init__(num_heads, num_heads, head_dim, head_dim)

        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))

    def forward(
        self,
        source_kv: torch.Tensor,
        target_kv: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted fusion of source and target."""
        assert source_kv.shape == target_kv.shape
        return self.alpha * source_kv + (1 - self.alpha) * target_kv


class LearnedGatingProjector(BaseProjector):
    """
    Learned gating projector with context-dependent fusion.

    Uses a small MLP to compute per-head gates based on the target KV cache,
    then selectively fuses source information:
        output = target + gate * (source - target)

    Ideal for Wan2.1 ↔ ChronoEdit where architectures match but we want
    learned, context-dependent fusion.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        hidden_dim: int = 256,
        gate_granularity: str = "head",  # "head", "token", or "value"
    ):
        super().__init__(num_heads, num_heads, head_dim, head_dim)

        self.gate_granularity = gate_granularity

        # Input: concatenated source + target per head
        input_dim = 2 * head_dim

        # Output depends on granularity
        if gate_granularity == "head":
            output_dim = 1  # One gate per head
        elif gate_granularity == "token":
            output_dim = 1  # Computed per token
        elif gate_granularity == "value":
            output_dim = head_dim  # One gate per value dimension
        else:
            raise ValueError(f"Unknown granularity: {gate_granularity}")

        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        source_kv: torch.Tensor,
        target_kv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Context-dependent gating fusion.

        Args:
            source_kv: (B, H, N, D)
            target_kv: (B, H, N, D)

        Returns:
            fused_kv: (B, H, N, D)
        """
        assert source_kv.shape == target_kv.shape
        B, H, N, D = source_kv.shape

        # Concatenate source and target along feature dim
        combined = torch.cat([source_kv, target_kv], dim=-1)  # (B, H, N, 2D)

        if self.gate_granularity == "head":
            # Compute one gate per head (average over tokens)
            combined_mean = combined.mean(dim=2)  # (B, H, 2D)
            gates = self.gate_network(combined_mean)  # (B, H, 1)
            gates = gates.unsqueeze(2).unsqueeze(3)  # (B, H, 1, 1)

        elif self.gate_granularity == "token":
            # Compute one gate per token (average over heads)
            combined_mean = combined.mean(dim=1)  # (B, N, 2D)
            gates = self.gate_network(combined_mean)  # (B, N, 1)
            gates = gates.unsqueeze(1).unsqueeze(3)  # (B, 1, N, 1)

        elif self.gate_granularity == "value":
            # Compute per-value gates
            gates = self.gate_network(combined)  # (B, H, N, D)

        # Apply gates: target + gate * (source - target)
        diff = source_kv - target_kv
        fused = target_kv + gates * diff

        return fused


class FullProjector(BaseProjector):
    """
    Full projection for mismatched architectures.

    Handles different head counts and dimensions by:
    1. Flattening multi-head structure: (B, H, N, D) → (B, N, H*D)
    2. Projecting through MLP: (B, N, H_s*D_s) → (B, N, H_t*D_t)
    3. Reshaping to target format: (B, N, H_t*D_t) → (B, H_t, N, D_t)
    4. Gated fusion with target KV

    Required for Qwen2.5-VL → Qwen3-VL (28 heads → 32 heads).
    """

    def __init__(
        self,
        source_heads: int,
        target_heads: int,
        source_head_dim: int,
        target_head_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 3,
    ):
        super().__init__(source_heads, target_heads, source_head_dim, target_head_dim)

        source_total_dim = source_heads * source_head_dim
        target_total_dim = target_heads * target_head_dim

        # Projection MLP
        layers = []
        in_dim = source_total_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, target_total_dim))

        self.projection = nn.Sequential(*layers)

        # Gating network (uses target KV to compute gate)
        self.gate_network = nn.Sequential(
            nn.Linear(target_total_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        source_kv: torch.Tensor,
        target_kv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full projection and gated fusion.

        Args:
            source_kv: (B, H_s, N, D_s)
            target_kv: (B, H_t, N, D_t)

        Returns:
            fused_kv: (B, H_t, N, D_t)
        """
        B, H_s, N, D_s = source_kv.shape
        _, H_t, _, D_t = target_kv.shape

        # Flatten source multi-head structure
        source_flat = source_kv.transpose(1, 2).reshape(B, N, H_s * D_s)

        # Project to target dimensions
        projected = self.projection(source_flat)  # (B, N, H_t * D_t)

        # Reshape to multi-head format
        projected = projected.reshape(B, N, H_t, D_t).transpose(1, 2)  # (B, H_t, N, D_t)

        # Compute gate based on target KV
        target_flat = target_kv.transpose(1, 2).reshape(B, N, H_t * D_t)
        gates = self.gate_network(target_flat)  # (B, N, 1)
        gates = gates.unsqueeze(1).unsqueeze(3)  # (B, 1, N, 1)

        # Gated fusion
        fused = target_kv + gates * projected

        return fused


def create_projector(
    source_config: dict,
    target_config: dict,
    projector_type: str = "auto",
    **kwargs
) -> BaseProjector:
    """
    Factory function to create appropriate projector.

    Args:
        source_config: {"num_heads": int, "head_dim": int}
        target_config: {"num_heads": int, "head_dim": int}
        projector_type: "auto", "identity", "weighted", "gating", or "full"
        **kwargs: Additional arguments for specific projector types

    Returns:
        Appropriate projector instance
    """
    source_heads = source_config["num_heads"]
    target_heads = target_config["num_heads"]
    source_dim = source_config["head_dim"]
    target_dim = target_config["head_dim"]

    # Auto-select projector type
    if projector_type == "auto":
        if source_heads == target_heads and source_dim == target_dim:
            projector_type = "gating"  # Use learned gating for identical archs
        else:
            projector_type = "full"  # Use full projection for mismatched

    # Create projector
    if projector_type == "identity":
        assert source_heads == target_heads and source_dim == target_dim
        return IdentityProjector(source_heads, source_dim)

    elif projector_type == "weighted":
        assert source_heads == target_heads and source_dim == target_dim
        return WeightedFusionProjector(
            source_heads,
            source_dim,
            alpha=kwargs.get("alpha", 0.5),
            learnable=kwargs.get("learnable", False),
        )

    elif projector_type == "gating":
        assert source_heads == target_heads and source_dim == target_dim
        return LearnedGatingProjector(
            source_heads,
            source_dim,
            hidden_dim=kwargs.get("hidden_dim", 256),
            gate_granularity=kwargs.get("gate_granularity", "head"),
        )

    elif projector_type == "full":
        return FullProjector(
            source_heads,
            target_heads,
            source_dim,
            target_dim,
            hidden_dim=kwargs.get("hidden_dim", 1024),
            num_layers=kwargs.get("num_layers", 3),
        )

    else:
        raise ValueError(f"Unknown projector type: {projector_type}")
