"""
Vision Cache Extraction Module

Extract vision-related KV caches and embeddings from Qwen models.

Key challenges:
1. Qwen3-VL has DeepStack: Multi-level ViT features injected at multiple layers
2. Qwen2.5-VL has single-level vision injection
3. Different model sizes: 9B vs 7B
4. Need to extract the right features for image editing (not just generation)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from PIL import Image


@dataclass
class VisionCache:
    """
    Structured vision cache containing extracted features.

    For Qwen3-VL (DeepStack):
        - early_features: Low-level (edges, textures) from layers 0-10
        - mid_features: Mid-level (objects, structure) from layers 10-20
        - late_features: High-level (semantics, context) from layers 20-32
        - vision_hidden_states: All vision token hidden states

    For Qwen2.5-VL (Standard):
        - vision_features: Single-level vision embeddings
        - vision_hidden_states: Vision token hidden states

    Common:
        - kv_cache: Full KV cache (past_key_values)
        - attention_mask: Attention mask for vision tokens
        - image_grid_thw: Image grid dimensions (time, height, width)
    """
    # Multi-level features (Qwen3 DeepStack)
    early_features: Optional[torch.Tensor] = None
    mid_features: Optional[torch.Tensor] = None
    late_features: Optional[torch.Tensor] = None

    # Single-level features (Qwen2.5)
    vision_features: Optional[torch.Tensor] = None

    # Common
    vision_hidden_states: Optional[List[torch.Tensor]] = None
    kv_cache: Optional[Tuple] = None
    attention_mask: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None

    # Metadata
    model_type: str = "unknown"  # "qwen3-vl" or "qwen25-vl"
    num_layers: int = 0
    hidden_dim: int = 0
    num_vision_tokens: int = 0


class VisionCacheExtractor:
    """
    Extract vision caches from Qwen3-VL and Qwen2.5-VL models.

    Usage:
        extractor = VisionCacheExtractor()

        # Extract from Qwen3-VL (better vision understanding)
        qwen3_cache = extractor.extract_qwen3_vision(
            image=image,
            text_prompt="Edit this image"
        )

        # Extract from Qwen2.5-VL (baseline in Image-Edit)
        qwen25_cache = extractor.extract_qwen25_vision(
            image=image,
            text_prompt="Edit this image"
        )

        # Bridge them for enhanced editing
        enhanced_cache = bridge(qwen3_cache, qwen25_cache)
    """

    def __init__(
        self,
        qwen3_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        qwen25_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        load_in_8bit: bool = False,
    ):
        """
        Initialize cache extractor with both models.

        Args:
            qwen3_model_name: HuggingFace model ID for Qwen3-VL
            qwen25_model_name: HuggingFace model ID for Qwen2.5-VL
            device: Device to load models on
            load_in_8bit: Use 8-bit quantization to save memory
        """
        self.device = device
        self.load_in_8bit = load_in_8bit

        # Models will be loaded lazily
        self._qwen3_model = None
        self._qwen3_processor = None
        self._qwen25_model = None
        self._qwen25_processor = None

        self.qwen3_model_name = qwen3_model_name
        self.qwen25_model_name = qwen25_model_name

        print(f"VisionCacheExtractor initialized")
        print(f"  Qwen3-VL: {qwen3_model_name}")
        print(f"  Qwen2.5-VL: {qwen25_model_name}")
        print(f"  Device: {device}")
        print(f"  8-bit quantization: {load_in_8bit}")

    @property
    def qwen3_model(self):
        """Lazy load Qwen3-VL model."""
        if self._qwen3_model is None:
            print("Loading Qwen3-VL model...")
            self._qwen3_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.qwen3_model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device if not self.load_in_8bit else "auto",
                load_in_8bit=self.load_in_8bit,
            )
            self._qwen3_model.eval()
            print(f"  Model loaded: {self.qwen3_model_name}")
            print(f"  Layers: {self._qwen3_model.config.num_hidden_layers}")
            print(f"  Hidden dim: {self._qwen3_model.config.hidden_size}")
        return self._qwen3_model

    @property
    def qwen3_processor(self):
        """Lazy load Qwen3-VL processor."""
        if self._qwen3_processor is None:
            print("Loading Qwen3-VL processor...")
            self._qwen3_processor = AutoProcessor.from_pretrained(
                self.qwen3_model_name
            )
        return self._qwen3_processor

    @property
    def qwen25_model(self):
        """Lazy load Qwen2.5-VL model."""
        if self._qwen25_model is None:
            print("Loading Qwen2.5-VL model...")
            self._qwen25_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.qwen25_model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device if not self.load_in_8bit else "auto",
                load_in_8bit=self.load_in_8bit,
            )
            self._qwen25_model.eval()
            print(f"  Model loaded: {self.qwen25_model_name}")
            print(f"  Layers: {self._qwen25_model.config.num_hidden_layers}")
            print(f"  Hidden dim: {self._qwen25_model.config.hidden_size}")
        return self._qwen25_model

    @property
    def qwen25_processor(self):
        """Lazy load Qwen2.5-VL processor."""
        if self._qwen25_processor is None:
            print("Loading Qwen2.5-VL processor...")
            self._qwen25_processor = AutoProcessor.from_pretrained(
                self.qwen25_model_name
            )
        return self._qwen25_processor

    def extract_qwen3_vision(
        self,
        image: Union[Image.Image, str],
        text_prompt: Optional[str] = None,
        extract_layers: Optional[List[int]] = None,
    ) -> VisionCache:
        """
        Extract vision cache from Qwen3-VL with DeepStack multi-level features.

        Qwen3-VL innovations:
        - DeepStack: Multi-level ViT features for fine details
        - Interleaved-MRoPE: Enhanced positional encoding
        - 256K context for multi-image understanding
        - 32-language OCR capability

        Args:
            image: Input image (PIL Image or path)
            text_prompt: Optional text prompt for context
            extract_layers: Specific layers to extract (None = all layers)

        Returns:
            VisionCache with multi-level DeepStack features
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Default prompt if none provided
        if text_prompt is None:
            text_prompt = "Describe this image in detail."

        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        text = self.qwen3_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.qwen3_processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features with hooks
        print(f"Extracting Qwen3-VL vision cache...")

        with torch.no_grad():
            outputs = self.qwen3_model(
                **inputs,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=True,
                return_dict=True,
            )

        # Extract vision tokens from hidden states
        # Vision tokens are the first tokens in the sequence
        hidden_states = outputs.hidden_states  # Tuple of (num_layers, batch, seq, hidden)
        num_layers = len(hidden_states)

        # Identify vision token positions
        # In Qwen3-VL, vision tokens are prepended to text tokens
        image_grid_thw = inputs.get("image_grid_thw", None)
        if image_grid_thw is not None:
            # Calculate number of vision tokens
            # Format: (temporal, height, width) grid
            num_vision_tokens = image_grid_thw[0, 0] * image_grid_thw[0, 1] * image_grid_thw[0, 2]
        else:
            # Fallback: Estimate from attention mask
            num_vision_tokens = self._estimate_vision_tokens(inputs["attention_mask"])

        print(f"  Detected {num_vision_tokens} vision tokens")
        print(f"  Model layers: {num_layers}")
        print(f"  Hidden dim: {hidden_states[0].shape[-1]}")

        # Extract vision features from each layer
        vision_hidden_states = []
        for layer_idx, layer_hidden in enumerate(hidden_states):
            # Extract just the vision tokens
            vision_tokens = layer_hidden[:, :num_vision_tokens, :]  # (batch, vision_seq, hidden)
            vision_hidden_states.append(vision_tokens)

        # DeepStack: Extract multi-level features
        # Early layers: Low-level features (edges, textures)
        early_layers = range(0, num_layers // 3)
        early_features = torch.stack([
            vision_hidden_states[i] for i in early_layers
        ]).mean(dim=0)  # Average across early layers

        # Middle layers: Mid-level features (objects, structure)
        mid_layers = range(num_layers // 3, 2 * num_layers // 3)
        mid_features = torch.stack([
            vision_hidden_states[i] for i in mid_layers
        ]).mean(dim=0)

        # Late layers: High-level features (semantics, context)
        late_layers = range(2 * num_layers // 3, num_layers)
        late_features = torch.stack([
            vision_hidden_states[i] for i in late_layers
        ]).mean(dim=0)

        print(f"  Early features: {early_features.shape}")
        print(f"  Mid features: {mid_features.shape}")
        print(f"  Late features: {late_features.shape}")

        # Create VisionCache
        cache = VisionCache(
            early_features=early_features,
            mid_features=mid_features,
            late_features=late_features,
            vision_hidden_states=vision_hidden_states,
            kv_cache=outputs.past_key_values,
            attention_mask=inputs["attention_mask"],
            image_grid_thw=image_grid_thw,
            model_type="qwen3-vl",
            num_layers=num_layers,
            hidden_dim=hidden_states[0].shape[-1],
            num_vision_tokens=num_vision_tokens.item() if torch.is_tensor(num_vision_tokens) else num_vision_tokens,
        )

        print(f"✓ Qwen3-VL cache extracted successfully")
        return cache

    def extract_qwen25_vision(
        self,
        image: Union[Image.Image, str],
        text_prompt: Optional[str] = None,
    ) -> VisionCache:
        """
        Extract vision cache from Qwen2.5-VL (standard single-level).

        This is the baseline used in Qwen-Image-Edit currently.

        Args:
            image: Input image (PIL Image or path)
            text_prompt: Optional text prompt for context

        Returns:
            VisionCache with single-level features
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Default prompt if none provided
        if text_prompt is None:
            text_prompt = "Describe this image in detail."

        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        text = self.qwen25_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.qwen25_processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        print(f"Extracting Qwen2.5-VL vision cache...")

        with torch.no_grad():
            outputs = self.qwen25_model(
                **inputs,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=True,
                return_dict=True,
            )

        # Extract vision tokens
        hidden_states = outputs.hidden_states
        num_layers = len(hidden_states)

        # Identify vision token positions
        image_grid_thw = inputs.get("image_grid_thw", None)
        if image_grid_thw is not None:
            num_vision_tokens = image_grid_thw[0, 0] * image_grid_thw[0, 1] * image_grid_thw[0, 2]
        else:
            num_vision_tokens = self._estimate_vision_tokens(inputs["attention_mask"])

        print(f"  Detected {num_vision_tokens} vision tokens")
        print(f"  Model layers: {num_layers}")
        print(f"  Hidden dim: {hidden_states[0].shape[-1]}")

        # Extract vision hidden states
        vision_hidden_states = []
        for layer_hidden in hidden_states:
            vision_tokens = layer_hidden[:, :num_vision_tokens, :]
            vision_hidden_states.append(vision_tokens)

        # For Qwen2.5, use last layer as primary features
        vision_features = vision_hidden_states[-1]

        print(f"  Vision features: {vision_features.shape}")

        # Create VisionCache
        cache = VisionCache(
            vision_features=vision_features,
            vision_hidden_states=vision_hidden_states,
            kv_cache=outputs.past_key_values,
            attention_mask=inputs["attention_mask"],
            image_grid_thw=image_grid_thw,
            model_type="qwen25-vl",
            num_layers=num_layers,
            hidden_dim=hidden_states[0].shape[-1],
            num_vision_tokens=num_vision_tokens.item() if torch.is_tensor(num_vision_tokens) else num_vision_tokens,
        )

        print(f"✓ Qwen2.5-VL cache extracted successfully")
        return cache

    def _estimate_vision_tokens(self, attention_mask: torch.Tensor) -> int:
        """
        Estimate number of vision tokens from attention mask.

        Vision tokens are typically at the beginning of the sequence.
        """
        # Simple heuristic: Vision tokens are contiguous at start
        # This is a fallback if image_grid_thw is not available

        # For now, return a reasonable default
        # This should be improved based on actual model behavior
        return 256  # Typical vision token count for 256x256 image

    def compare_caches(
        self,
        qwen3_cache: VisionCache,
        qwen25_cache: VisionCache,
    ) -> Dict:
        """
        Compare Qwen3 and Qwen2.5 vision caches.

        Useful for understanding differences and debugging bridge.

        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            "qwen3": {
                "model_type": qwen3_cache.model_type,
                "num_layers": qwen3_cache.num_layers,
                "hidden_dim": qwen3_cache.hidden_dim,
                "num_vision_tokens": qwen3_cache.num_vision_tokens,
                "has_multi_level": qwen3_cache.early_features is not None,
            },
            "qwen25": {
                "model_type": qwen25_cache.model_type,
                "num_layers": qwen25_cache.num_layers,
                "hidden_dim": qwen25_cache.hidden_dim,
                "num_vision_tokens": qwen25_cache.num_vision_tokens,
                "has_multi_level": qwen25_cache.early_features is not None,
            },
            "compatibility": {
                "vision_token_count_match": (
                    qwen3_cache.num_vision_tokens == qwen25_cache.num_vision_tokens
                ),
                "dimension_mismatch": (
                    qwen3_cache.hidden_dim != qwen25_cache.hidden_dim
                ),
                "dimension_ratio": (
                    qwen3_cache.hidden_dim / qwen25_cache.hidden_dim
                    if qwen25_cache.hidden_dim > 0 else 0
                ),
            }
        }

        print("\n" + "="*60)
        print("Cache Comparison")
        print("="*60)
        print(f"\nQwen3-VL:")
        print(f"  Layers: {comparison['qwen3']['num_layers']}")
        print(f"  Hidden dim: {comparison['qwen3']['hidden_dim']}")
        print(f"  Vision tokens: {comparison['qwen3']['num_vision_tokens']}")
        print(f"  Multi-level (DeepStack): {comparison['qwen3']['has_multi_level']}")

        print(f"\nQwen2.5-VL:")
        print(f"  Layers: {comparison['qwen25']['num_layers']}")
        print(f"  Hidden dim: {comparison['qwen25']['hidden_dim']}")
        print(f"  Vision tokens: {comparison['qwen25']['num_vision_tokens']}")
        print(f"  Multi-level: {comparison['qwen25']['has_multi_level']}")

        print(f"\nCompatibility:")
        print(f"  Vision token count match: {comparison['compatibility']['vision_token_count_match']}")
        print(f"  Dimension mismatch: {comparison['compatibility']['dimension_mismatch']}")
        print(f"  Dimension ratio: {comparison['compatibility']['dimension_ratio']:.2f}")
        print("="*60 + "\n")

        return comparison
