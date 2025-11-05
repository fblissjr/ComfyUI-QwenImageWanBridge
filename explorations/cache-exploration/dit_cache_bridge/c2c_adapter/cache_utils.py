"""
KV Cache Utilities

Helper functions for extracting, manipulating, and applying KV caches
in transformer models.
"""

import torch
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class DynamicCache:
    """
    Dynamic KV cache structure compatible with HuggingFace transformers.

    Stores key and value caches as lists of tensors, one per layer.
    """
    key_cache: List[torch.Tensor]  # Per-layer keys: (B, H, N, D)
    value_cache: List[torch.Tensor]  # Per-layer values: (B, H, N, D)

    def __len__(self) -> int:
        """Number of layers in cache."""
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get key and value for specific layer."""
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def to(self, device: Union[str, torch.device]) -> 'DynamicCache':
        """Move cache to device."""
        self.key_cache = [k.to(device) for k in self.key_cache]
        self.value_cache = [v.to(device) for v in self.value_cache]
        return self

    def clone(self) -> 'DynamicCache':
        """Create a deep copy of the cache."""
        return DynamicCache(
            key_cache=[k.clone() for k in self.key_cache],
            value_cache=[v.clone() for v in self.value_cache],
        )

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Shape of cache tensors (B, H, N, D)."""
        if len(self.key_cache) == 0:
            return (0, 0, 0, 0)
        return self.key_cache[0].shape

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get sequence length from specific layer."""
        return self.key_cache[layer_idx].shape[2]


def extract_kv_cache(
    model: torch.nn.Module,
    inputs: dict,
    layer_indices: Optional[List[int]] = None,
) -> DynamicCache:
    """
    Extract KV cache from a model's forward pass.

    Args:
        model: PyTorch model (should support output_hidden_states=True)
        inputs: Input dictionary for model forward pass
        layer_indices: Specific layers to extract (None = all layers)

    Returns:
        DynamicCache with extracted KV tensors
    """
    # Enable KV cache extraction if model supports it
    model_kwargs = {
        **inputs,
        "output_hidden_states": True,
        "use_cache": True,
    }

    # Forward pass
    with torch.no_grad():
        outputs = model(**model_kwargs)

    # Extract cache from outputs
    # Different models store cache differently:
    # - HuggingFace: outputs.past_key_values
    # - Custom: outputs.cache or outputs.kv_cache

    if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
        cache = outputs.past_key_values
    elif hasattr(outputs, 'cache'):
        cache = outputs.cache
    elif hasattr(outputs, 'kv_cache'):
        cache = outputs.kv_cache
    else:
        raise ValueError("Model does not output KV cache. Check model implementation.")

    # Convert to DynamicCache format
    if isinstance(cache, DynamicCache):
        result_cache = cache
    elif isinstance(cache, (list, tuple)):
        # HuggingFace format: list of (key, value) tuples
        key_cache = [layer_cache[0] for layer_cache in cache]
        value_cache = [layer_cache[1] for layer_cache in cache]
        result_cache = DynamicCache(key_cache, value_cache)
    else:
        raise ValueError(f"Unknown cache format: {type(cache)}")

    # Filter to specific layers if requested
    if layer_indices is not None:
        result_cache = DynamicCache(
            key_cache=[result_cache.key_cache[i] for i in layer_indices],
            value_cache=[result_cache.value_cache[i] for i in layer_indices],
        )

    return result_cache


def apply_kv_cache(
    model: torch.nn.Module,
    cache: DynamicCache,
    inputs: dict,
) -> torch.Tensor:
    """
    Run model generation with a pre-filled KV cache.

    Args:
        model: PyTorch model
        cache: Pre-filled KV cache
        inputs: Input dictionary for generation

    Returns:
        Model outputs using the provided cache
    """
    # Convert cache to model's expected format
    if hasattr(model, 'cache_format'):
        # Custom model with specific cache format
        model_cache = model.convert_cache(cache)
    else:
        # Standard HuggingFace format: list of (key, value) tuples
        model_cache = list(zip(cache.key_cache, cache.value_cache))

    # Run model with cache
    model_kwargs = {
        **inputs,
        "past_key_values": model_cache,
        "use_cache": True,
    }

    outputs = model(**model_kwargs)
    return outputs


def fuse_caches(
    source_cache: DynamicCache,
    target_cache: DynamicCache,
    projector: torch.nn.Module,
    layer_mapping: Optional[dict] = None,
) -> DynamicCache:
    """
    Fuse two KV caches using a projector.

    Args:
        source_cache: Cache from source model
        target_cache: Cache from target model
        projector: Projector module to transform/fuse caches
        layer_mapping: Map source layers → target layers (None = 1:1)

    Returns:
        Fused cache in target model's format
    """
    # Default 1:1 layer mapping
    if layer_mapping is None:
        assert len(source_cache) == len(target_cache), \
            f"Cache layer mismatch: {len(source_cache)} vs {len(target_cache)}"
        layer_mapping = {i: i for i in range(len(source_cache))}

    # Prepare fused cache
    fused_keys = []
    fused_values = []

    for target_idx in range(len(target_cache)):
        # Get corresponding source layer
        if target_idx in layer_mapping:
            source_idx = layer_mapping[target_idx]
            source_key = source_cache.key_cache[source_idx]
            source_value = source_cache.value_cache[source_idx]
        else:
            # No source mapping, use target only
            fused_keys.append(target_cache.key_cache[target_idx])
            fused_values.append(target_cache.value_cache[target_idx])
            continue

        target_key = target_cache.key_cache[target_idx]
        target_value = target_cache.value_cache[target_idx]

        # Project and fuse
        fused_key = projector(source_key, target_key)
        fused_value = projector(source_value, target_value)

        fused_keys.append(fused_key)
        fused_values.append(fused_value)

    return DynamicCache(fused_keys, fused_values)


def interpolate_cache_sequence(
    cache: DynamicCache,
    target_length: int,
    method: str = "linear",
) -> DynamicCache:
    """
    Interpolate KV cache to different sequence length.

    Useful when source and target models process different numbers of tokens.

    Args:
        cache: Input cache
        target_length: Desired sequence length
        method: Interpolation method ("linear", "nearest")

    Returns:
        Cache with interpolated sequence dimension
    """
    interpolated_keys = []
    interpolated_values = []

    for key, value in zip(cache.key_cache, cache.value_cache):
        B, H, N, D = key.shape

        if N == target_length:
            # No interpolation needed
            interpolated_keys.append(key)
            interpolated_values.append(value)
            continue

        # Reshape for interpolation: (B, H, N, D) → (B*H, D, N)
        key_reshaped = key.transpose(2, 3).reshape(B * H, D, N)
        value_reshaped = value.transpose(2, 3).reshape(B * H, D, N)

        # Interpolate sequence dimension
        key_interp = torch.nn.functional.interpolate(
            key_reshaped,
            size=target_length,
            mode=method,
            align_corners=False if method == "linear" else None,
        )
        value_interp = torch.nn.functional.interpolate(
            value_reshaped,
            size=target_length,
            mode=method,
            align_corners=False if method == "linear" else None,
        )

        # Reshape back: (B*H, D, N') → (B, H, N', D)
        key_interp = key_interp.reshape(B, H, D, target_length).transpose(2, 3)
        value_interp = value_interp.reshape(B, H, D, target_length).transpose(2, 3)

        interpolated_keys.append(key_interp)
        interpolated_values.append(value_interp)

    return DynamicCache(interpolated_keys, interpolated_values)


def build_layer_mapping(
    source_layers: int,
    target_layers: int,
    strategy: str = "position",
) -> dict:
    """
    Build layer mapping between source and target models.

    Args:
        source_layers: Number of layers in source model
        target_layers: Number of layers in target model
        strategy: Mapping strategy:
            - "position": Map by relative position (0→0, middle→middle, end→end)
            - "uniform": Uniformly sample source layers
            - "duplicate": Duplicate source layers to match target

    Returns:
        Dictionary mapping target_layer_idx → source_layer_idx
    """
    if strategy == "position":
        # Position-based interpolation (C2C default)
        source_positions = [i / (source_layers - 1) for i in range(source_layers)]
        target_positions = [j / (target_layers - 1) for j in range(target_layers)]

        mapping = {}
        for target_idx, target_pos in enumerate(target_positions):
            # Find closest source layer by position
            distances = [abs(target_pos - src_pos) for src_pos in source_positions]
            source_idx = distances.index(min(distances))
            mapping[target_idx] = source_idx

    elif strategy == "uniform":
        # Uniformly sample source layers
        indices = torch.linspace(0, source_layers - 1, target_layers).long()
        mapping = {i: idx.item() for i, idx in enumerate(indices)}

    elif strategy == "duplicate":
        # Duplicate source layers (repeat if target > source)
        mapping = {}
        for target_idx in range(target_layers):
            source_idx = (target_idx * source_layers) // target_layers
            mapping[target_idx] = min(source_idx, source_layers - 1)

    else:
        raise ValueError(f"Unknown mapping strategy: {strategy}")

    return mapping
