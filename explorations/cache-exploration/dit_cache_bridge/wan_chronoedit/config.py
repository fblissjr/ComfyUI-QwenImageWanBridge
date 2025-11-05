"""
Configuration for Wan2.1 ↔ ChronoEdit Bridging
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class WanChronoEditConfig:
    """
    Configuration for Wan ↔ ChronoEdit cache bridging.

    Since both models have identical architectures, most parameters
    relate to the fusion strategy rather than dimension transformation.
    """

    # Model specifications (identical for both)
    num_layers: int = 40
    num_heads: int = 40
    head_dim: int = 128
    hidden_dim: int = 5120

    # Projector configuration
    projector_type: Literal["identity", "weighted", "gating"] = "gating"
    projector_hidden_dim: int = 256
    gate_granularity: Literal["head", "token", "value"] = "head"

    # Weighted fusion (if projector_type == "weighted")
    fusion_alpha: float = 0.5
    learnable_alpha: bool = True

    # Layer mapping (1:1 since architectures match)
    layer_mapping_strategy: Literal["identity", "custom"] = "identity"
    custom_layer_mapping: Optional[dict] = None

    # Inference configuration
    batch_size: int = 1
    device: str = "cuda"

    # Fusion strategy
    fusion_direction: Literal["wan_to_chrono", "chrono_to_wan", "bidirectional"] = "bidirectional"

    # Training configuration
    trainable_components: list = field(default_factory=lambda: ["projector"])
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    def __post_init__(self):
        """Validate configuration."""
        if self.projector_type == "identity" and self.fusion_direction == "bidirectional":
            raise ValueError("Identity projector doesn't make sense for bidirectional fusion")

        if self.custom_layer_mapping is not None:
            self.layer_mapping_strategy = "custom"


    def get_wan_config(self) -> dict:
        """Get Wan2.1 model configuration."""
        return {
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
        }

    def get_chronoedit_config(self) -> dict:
        """Get ChronoEdit model configuration (same as Wan)."""
        return self.get_wan_config()
