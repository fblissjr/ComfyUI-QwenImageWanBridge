"""
Wan2.1 ↔ ChronoEdit Bridge Implementation

Main bridge class for combining Wan2.1 and ChronoEdit KV caches.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from c2c_adapter.projectors import create_projector, BaseProjector
from c2c_adapter.cache_utils import (
    DynamicCache,
    extract_kv_cache,
    fuse_caches,
    build_layer_mapping,
)
from .config import WanChronoEditConfig


class WanChronoEditBridge(nn.Module):
    """
    Bridge between Wan2.1 and ChronoEdit models.

    This class handles:
    1. Loading both models
    2. Running parallel prefill to extract KV caches
    3. Fusing caches using learned projector
    4. Running generation with fused cache

    Since architectures are identical, this is the simplest bridging scenario.
    """

    def __init__(
        self,
        config: WanChronoEditConfig,
        wan_model: Optional[nn.Module] = None,
        chronoedit_model: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.config = config

        # Store model references (not as nn.Module members to avoid training)
        self.wan_model = wan_model
        self.chronoedit_model = chronoedit_model

        # Create projector for KV cache fusion
        self.key_projector = self._create_projector()
        self.value_projector = self._create_projector()

        # Build layer mapping (identity for Wan ↔ ChronoEdit)
        self.layer_mapping = self._build_layer_mapping()

    def _create_projector(self) -> BaseProjector:
        """Create projector for KV cache fusion."""
        model_config = {
            "num_heads": self.config.num_heads,
            "head_dim": self.config.head_dim,
        }

        projector_kwargs = {
            "hidden_dim": self.config.projector_hidden_dim,
        }

        if self.config.projector_type == "gating":
            projector_kwargs["gate_granularity"] = self.config.gate_granularity
        elif self.config.projector_type == "weighted":
            projector_kwargs["alpha"] = self.config.fusion_alpha
            projector_kwargs["learnable"] = self.config.learnable_alpha

        return create_projector(
            source_config=model_config,
            target_config=model_config,
            projector_type=self.config.projector_type,
            **projector_kwargs
        )

    def _build_layer_mapping(self) -> Dict[int, int]:
        """Build layer mapping between models."""
        if self.config.layer_mapping_strategy == "identity":
            # 1:1 mapping since architectures match
            return {i: i for i in range(self.config.num_layers)}
        elif self.config.layer_mapping_strategy == "custom":
            return self.config.custom_layer_mapping
        else:
            raise ValueError(f"Unknown mapping strategy: {self.config.layer_mapping_strategy}")

    def extract_wan_cache(self, inputs: Dict) -> DynamicCache:
        """
        Extract KV cache from Wan2.1 model.

        Args:
            inputs: Input dictionary for Wan model

        Returns:
            KV cache from Wan2.1
        """
        if self.wan_model is None:
            raise ValueError("Wan model not loaded. Provide wan_model in __init__.")

        return extract_kv_cache(self.wan_model, inputs)

    def extract_chronoedit_cache(self, inputs: Dict) -> DynamicCache:
        """
        Extract KV cache from ChronoEdit model.

        Args:
            inputs: Input dictionary for ChronoEdit model

        Returns:
            KV cache from ChronoEdit
        """
        if self.chronoedit_model is None:
            raise ValueError("ChronoEdit model not loaded. Provide chronoedit_model in __init__.")

        return extract_kv_cache(self.chronoedit_model, inputs)

    def fuse_caches(
        self,
        wan_cache: DynamicCache,
        chronoedit_cache: DynamicCache,
    ) -> DynamicCache:
        """
        Fuse Wan and ChronoEdit KV caches.

        Args:
            wan_cache: KV cache from Wan2.1
            chronoedit_cache: KV cache from ChronoEdit

        Returns:
            Fused KV cache
        """
        # Validate cache dimensions
        assert len(wan_cache) == len(chronoedit_cache), \
            f"Layer count mismatch: {len(wan_cache)} vs {len(chronoedit_cache)}"

        assert wan_cache.shape == chronoedit_cache.shape, \
            f"Cache shape mismatch: {wan_cache.shape} vs {chronoedit_cache.shape}"

        # Fuse keys and values separately
        fused_keys = []
        fused_values = []

        for layer_idx in range(len(wan_cache)):
            wan_key, wan_value = wan_cache[layer_idx]
            chrono_key, chrono_value = chronoedit_cache[layer_idx]

            # Determine fusion direction
            if self.config.fusion_direction == "wan_to_chrono":
                # Wan is source, ChronoEdit is target
                source_key, target_key = wan_key, chrono_key
                source_value, target_value = wan_value, chrono_value
            elif self.config.fusion_direction == "chrono_to_wan":
                # ChronoEdit is source, Wan is target
                source_key, target_key = chrono_key, wan_key
                source_value, target_value = chrono_value, wan_value
            else:  # bidirectional
                # Average fusion (or use projector to decide)
                source_key = (wan_key + chrono_key) / 2
                target_key = chrono_key
                source_value = (wan_value + chrono_value) / 2
                target_value = chrono_value

            # Project and fuse
            fused_key = self.key_projector(source_key, target_key)
            fused_value = self.value_projector(source_value, target_value)

            fused_keys.append(fused_key)
            fused_values.append(fused_value)

        return DynamicCache(fused_keys, fused_values)

    def forward(
        self,
        wan_inputs: Dict,
        chronoedit_inputs: Dict,
        target_model: str = "wan",
    ) -> Tuple[DynamicCache, Dict]:
        """
        Full forward pass: extract caches, fuse, and prepare for generation.

        Args:
            wan_inputs: Inputs for Wan2.1 model
            chronoedit_inputs: Inputs for ChronoEdit model
            target_model: Which model to use for generation ("wan" or "chronoedit")

        Returns:
            Tuple of (fused_cache, generation_inputs)
        """
        # Extract KV caches from both models
        wan_cache = self.extract_wan_cache(wan_inputs)
        chronoedit_cache = self.extract_chronoedit_cache(chronoedit_inputs)

        # Fuse caches
        fused_cache = self.fuse_caches(wan_cache, chronoedit_cache)

        # Prepare generation inputs for target model
        if target_model == "wan":
            generation_inputs = wan_inputs.copy()
            generation_inputs["past_key_values"] = list(zip(fused_cache.key_cache, fused_cache.value_cache))
        elif target_model == "chronoedit":
            generation_inputs = chronoedit_inputs.copy()
            generation_inputs["past_key_values"] = list(zip(fused_cache.key_cache, fused_cache.value_cache))
        else:
            raise ValueError(f"Unknown target model: {target_model}")

        return fused_cache, generation_inputs

    def generate_with_fusion(
        self,
        wan_inputs: Dict,
        chronoedit_inputs: Dict,
        target_model: str = "wan",
        **generation_kwargs
    ) -> torch.Tensor:
        """
        End-to-end generation with cache fusion.

        Args:
            wan_inputs: Inputs for Wan2.1
            chronoedit_inputs: Inputs for ChronoEdit
            target_model: Model to use for generation
            **generation_kwargs: Additional kwargs for generation

        Returns:
            Generated output tensor
        """
        # Get fused cache and generation inputs
        fused_cache, generation_inputs = self.forward(
            wan_inputs,
            chronoedit_inputs,
            target_model
        )

        # Add generation kwargs
        generation_inputs.update(generation_kwargs)

        # Generate with target model
        if target_model == "wan":
            if self.wan_model is None:
                raise ValueError("Wan model not loaded")
            with torch.no_grad():
                outputs = self.wan_model.generate(**generation_inputs)
        elif target_model == "chronoedit":
            if self.chronoedit_model is None:
                raise ValueError("ChronoEdit model not loaded")
            with torch.no_grad():
                outputs = self.chronoedit_model.generate(**generation_inputs)

        return outputs

    def train_projector(
        self,
        dataloader,
        num_epochs: int = 10,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Train projector on paired Wan/ChronoEdit outputs.

        Args:
            dataloader: DataLoader yielding (wan_inputs, chronoedit_inputs, targets)
            num_epochs: Number of training epochs
            optimizer: Optimizer (default: AdamW)
        """
        # Set projectors to training mode
        self.key_projector.train()
        self.value_projector.train()

        # Create optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                list(self.key_projector.parameters()) + list(self.value_projector.parameters()),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                wan_inputs, chronoedit_inputs, targets = batch

                # Extract and fuse caches
                fused_cache, _ = self.forward(wan_inputs, chronoedit_inputs)

                # Compute loss (MSE between fused cache and target cache)
                # In practice, target could be:
                # 1. Ground truth ChronoEdit cache
                # 2. Generated video quality loss
                # 3. Temporal consistency loss
                target_cache = targets["cache"]
                loss = 0.0

                for layer_idx in range(len(fused_cache)):
                    fused_key, fused_value = fused_cache[layer_idx]
                    target_key, target_value = target_cache[layer_idx]

                    loss += nn.functional.mse_loss(fused_key, target_key)
                    loss += nn.functional.mse_loss(fused_value, target_value)

                loss = loss / len(fused_cache)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    def save_projectors(self, path: str):
        """Save trained projectors."""
        torch.save({
            "key_projector": self.key_projector.state_dict(),
            "value_projector": self.value_projector.state_dict(),
            "config": self.config,
        }, path)
        print(f"Projectors saved to {path}")

    def load_projectors(self, path: str):
        """Load trained projectors."""
        checkpoint = torch.load(path)
        self.key_projector.load_state_dict(checkpoint["key_projector"])
        self.value_projector.load_state_dict(checkpoint["value_projector"])
        print(f"Projectors loaded from {path}")
