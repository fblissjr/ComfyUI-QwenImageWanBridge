"""
Vision Cache Bridge/Projector

Project Qwen3-VL's superior vision understanding into Qwen2.5-VL format.

Key operations:
1. Project DeepStack multi-level features (9B → 7B dimensions)
2. Fuse multi-level features intelligently
3. Output Qwen2.5-compatible vision cache

Design goals:
- Lightweight: <10M trainable parameters
- Fast: Minimal inference overhead
- Effective: Preserve Qwen3's superior capabilities
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from cache_extraction import VisionCache


@dataclass
class BridgeConfig:
    """
    Configuration for VisionCacheBridge.
    """
    # Source (Qwen3) dimensions
    qwen3_hidden_dim: int = 4096  # 8B model
    qwen3_num_layers: int = 32

    # Target (Qwen2.5) dimensions
    qwen25_hidden_dim: int = 3584  # 7B model
    qwen25_num_layers: int = 28

    # Bridge architecture
    fusion_hidden_dim: int = 7168  # 2x qwen25_hidden_dim
    num_attention_heads: int = 8
    dropout: float = 0.1

    # Fusion strategy
    use_attention_fusion: bool = True  # Use attention vs simple concat
    use_residual: bool = True  # Add residual connection to Qwen2.5 baseline
    layer_wise_fusion: bool = False  # Different fusion per layer vs shared

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01


class DeepStackProjector(nn.Module):
    """
    Project Qwen3's DeepStack multi-level features to Qwen2.5 dimensions.

    For each level (early, mid, late):
    - Input: (batch, num_vision_tokens, qwen3_hidden_dim)
    - Output: (batch, num_vision_tokens, qwen25_hidden_dim)
    """

    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.config = config

        # Separate projector for each DeepStack level
        self.early_projector = nn.Sequential(
            nn.Linear(config.qwen3_hidden_dim, config.qwen25_hidden_dim),
            nn.LayerNorm(config.qwen25_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.mid_projector = nn.Sequential(
            nn.Linear(config.qwen3_hidden_dim, config.qwen25_hidden_dim),
            nn.LayerNorm(config.qwen25_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.late_projector = nn.Sequential(
            nn.Linear(config.qwen3_hidden_dim, config.qwen25_hidden_dim),
            nn.LayerNorm(config.qwen25_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        early_features: torch.Tensor,
        mid_features: torch.Tensor,
        late_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project all DeepStack levels to Qwen2.5 dimensions.

        Args:
            early_features: (B, N, qwen3_hidden_dim)
            mid_features: (B, N, qwen3_hidden_dim)
            late_features: (B, N, qwen3_hidden_dim)

        Returns:
            Tuple of projected features, each (B, N, qwen25_hidden_dim)
        """
        early_proj = self.early_projector(early_features)
        mid_proj = self.mid_projector(mid_features)
        late_proj = self.late_projector(late_features)

        return early_proj, mid_proj, late_proj


class MultiLevelFusion(nn.Module):
    """
    Fuse multi-level projected features into single representation.

    Strategy options:
    1. Simple concatenation → MLP
    2. Attention-based fusion (query: baseline, key/value: multi-level)
    3. Learned gating per level
    """

    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.config = config

        if config.use_attention_fusion:
            # Attention-based fusion
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=config.qwen25_hidden_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
                batch_first=True,
            )

            # Project concatenated levels to query space
            self.level_projection = nn.Linear(
                config.qwen25_hidden_dim * 3,  # early + mid + late
                config.qwen25_hidden_dim
            )

        else:
            # MLP-based fusion
            self.fusion_network = nn.Sequential(
                nn.Linear(
                    config.qwen25_hidden_dim * 3,  # Concatenated levels
                    config.fusion_hidden_dim
                ),
                nn.LayerNorm(config.fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.fusion_hidden_dim, config.qwen25_hidden_dim),
            )

        # Level importance gates (learned weights for each level)
        self.level_gates = nn.Parameter(torch.ones(3) / 3)  # Equal init

    def forward(
        self,
        early_proj: torch.Tensor,
        mid_proj: torch.Tensor,
        late_proj: torch.Tensor,
        baseline_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fuse multi-level features.

        Args:
            early_proj: (B, N, D)
            mid_proj: (B, N, D)
            late_proj: (B, N, D)
            baseline_features: Optional Qwen2.5 baseline (B, N, D)

        Returns:
            fused_features: (B, N, D)
            attention_weights: Optional attention weights for analysis
        """
        # Apply learned gates to each level
        gates = torch.softmax(self.level_gates, dim=0)
        gated_early = gates[0] * early_proj
        gated_mid = gates[1] * mid_proj
        gated_late = gates[2] * late_proj

        if self.config.use_attention_fusion and baseline_features is not None:
            # Concatenate multi-level features
            multi_level = torch.cat([gated_early, gated_mid, gated_late], dim=-1)

            # Project to query space
            multi_level_projected = self.level_projection(multi_level)

            # Attention fusion: Query = baseline, Key/Value = multi-level
            fused_features, attention_weights = self.fusion_attention(
                query=baseline_features,
                key=multi_level_projected,
                value=multi_level_projected,
            )

            return fused_features, attention_weights

        else:
            # Simple MLP fusion
            multi_level = torch.cat([gated_early, gated_mid, gated_late], dim=-1)
            fused_features = self.fusion_network(multi_level)

            return fused_features, None


class VisionCacheBridge(nn.Module):
    """
    Main bridge: Inject Qwen3-VL's vision understanding into Qwen2.5-VL format.

    Pipeline:
    1. Extract DeepStack features from Qwen3
    2. Project each level: 9B → 7B dimensions
    3. Fuse multi-level features intelligently
    4. Optionally combine with Qwen2.5 baseline (residual)
    5. Output Qwen2.5-compatible vision cache

    This enables:
    - OCR-aware editing (32 langs)
    - Better spatial reasoning
    - Fine detail preservation
    - Multi-image consistency
    """

    def __init__(self, config: Optional[BridgeConfig] = None):
        super().__init__()

        if config is None:
            config = BridgeConfig()

        self.config = config

        # DeepStack projector
        self.projector = DeepStackProjector(config)

        # Multi-level fusion
        self.fusion = MultiLevelFusion(config)

        # Residual blending (how much baseline vs enhanced)
        if config.use_residual:
            self.residual_weight = nn.Parameter(torch.tensor(0.5))  # Learnable blend

        print(f"VisionCacheBridge initialized")
        print(f"  Qwen3: {config.qwen3_hidden_dim}D → Qwen2.5: {config.qwen25_hidden_dim}D")
        print(f"  Attention fusion: {config.use_attention_fusion}")
        print(f"  Residual connection: {config.use_residual}")
        print(f"  Total parameters: {self.count_parameters():,}")

    def forward(
        self,
        qwen3_cache: VisionCache,
        qwen25_cache: VisionCache,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Bridge Qwen3 vision cache to Qwen2.5 format.

        Args:
            qwen3_cache: DeepStack multi-level features from Qwen3
            qwen25_cache: Baseline features from Qwen2.5
            return_attention_weights: Whether to return attention maps

        Returns:
            enhanced_features: Qwen2.5-compatible cache with Qwen3's understanding
            attention_weights: Optional attention weights for analysis
        """
        # Extract features
        early = qwen3_cache.early_features  # (B, N, qwen3_dim)
        mid = qwen3_cache.mid_features
        late = qwen3_cache.late_features
        baseline = qwen25_cache.vision_features  # (B, N, qwen25_dim)

        # Project DeepStack levels
        early_proj, mid_proj, late_proj = self.projector(early, mid, late)

        # Fuse multi-level features
        fused_features, attention_weights = self.fusion(
            early_proj, mid_proj, late_proj, baseline
        )

        # Optionally blend with baseline (residual connection)
        if self.config.use_residual:
            # Learned weighted blend
            alpha = torch.sigmoid(self.residual_weight)
            enhanced_features = alpha * fused_features + (1 - alpha) * baseline
        else:
            enhanced_features = fused_features

        if return_attention_weights:
            return enhanced_features, attention_weights
        return enhanced_features, None

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        """Save bridge weights."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
        }, path)
        print(f"Bridge saved to {path}")

    def load(self, path: str):
        """Load bridge weights."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        print(f"Bridge loaded from {path}")

    @classmethod
    def from_pretrained(cls, path: str) -> 'VisionCacheBridge':
        """Load pre-trained bridge."""
        checkpoint = torch.load(path)
        bridge = cls(config=checkpoint['config'])
        bridge.load_state_dict(checkpoint['state_dict'])
        print(f"Bridge loaded from {path}")
        return bridge


def create_default_bridge(
    qwen3_hidden: int = 4096,
    qwen25_hidden: int = 3584,
    use_attention: bool = True,
) -> VisionCacheBridge:
    """
    Factory function to create bridge with sensible defaults.

    Args:
        qwen3_hidden: Qwen3-VL hidden dimension (9B: 4096)
        qwen25_hidden: Qwen2.5-VL hidden dimension (7B: 3584)
        use_attention: Use attention-based fusion vs MLP

    Returns:
        Initialized VisionCacheBridge
    """
    config = BridgeConfig(
        qwen3_hidden_dim=qwen3_hidden,
        qwen25_hidden_dim=qwen25_hidden,
        use_attention_fusion=use_attention,
        use_residual=True,
    )

    return VisionCacheBridge(config)
