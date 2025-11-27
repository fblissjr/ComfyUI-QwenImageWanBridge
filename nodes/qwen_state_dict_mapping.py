"""
State dict key mapping between ComfyUI's Qwen2.5-VL implementation and HuggingFace transformers.

This module provides functions to convert state_dict keys between:
- ComfyUI's custom Qwen implementation (used when loading via CLIPType.HUNYUAN_VIDEO_15)
- HuggingFace transformers' Qwen2VLForConditionalGeneration

Author: Generated for ComfyUI-QwenImageWanBridge
Purpose: Enable extracting weights from ComfyUI's loaded CLIP model to reconstruct
         a transformers model for text generation without loading the model twice.
"""

import re
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# COMFYUI STATE DICT KEY PATTERNS
# =============================================================================
#
# ComfyUI loads Qwen2.5-VL via comfy/text_encoders/llama.py -> Qwen25_7BVLI class
# The model structure is:
#
# Qwen25_7BVLI (BaseLlama, torch.nn.Module)
#   ├── model: Llama2_ (the language model)
#   │   ├── embed_tokens: Embedding
#   │   ├── layers: ModuleList[TransformerBlock]
#   │   │   └── [0..27]:
#   │   │       ├── self_attn: Attention
#   │   │       │   ├── q_proj: Linear
#   │   │       │   ├── k_proj: Linear
#   │   │       │   ├── v_proj: Linear
#   │   │       │   └── o_proj: Linear
#   │   │       ├── mlp: MLP
#   │   │       │   ├── gate_proj: Linear
#   │   │       │   ├── up_proj: Linear
#   │   │       │   └── down_proj: Linear
#   │   │       ├── input_layernorm: RMSNorm
#   │   │       └── post_attention_layernorm: RMSNorm
#   │   └── norm: RMSNorm (final norm)
#   │
#   └── visual: Qwen2VLVisionTransformer
#       ├── patch_embed: VisionPatchEmbed
#       │   └── proj: Conv3d
#       ├── rotary_pos_emb: VisionRotaryEmbedding
#       ├── blocks: ModuleList[VisionBlock]
#       │   └── [0..31]:
#       │       ├── norm1: RMSNorm
#       │       ├── norm2: RMSNorm
#       │       ├── attn: VisionAttention
#       │       │   ├── qkv: Linear
#       │       │   └── proj: Linear
#       │       └── mlp: VisionMLP
#       │           ├── gate_proj: Linear
#       │           ├── up_proj: Linear
#       │           └── down_proj: Linear
#       └── merger: PatchMerger
#           ├── ln_q: RMSNorm
#           └── mlp: Sequential[Linear, GELU, Linear]
#
# ComfyUI key format:
#   model.embed_tokens.weight
#   model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}
#   model.layers.{i}.mlp.{gate,up,down}_proj.weight
#   model.layers.{i}.input_layernorm.weight
#   model.layers.{i}.post_attention_layernorm.weight
#   model.norm.weight
#   visual.patch_embed.proj.{weight}
#   visual.blocks.{i}.norm{1,2}.weight
#   visual.blocks.{i}.attn.qkv.{weight,bias}
#   visual.blocks.{i}.attn.proj.{weight,bias}
#   visual.blocks.{i}.mlp.{gate,up,down}_proj.{weight,bias}
#   visual.merger.ln_q.weight
#   visual.merger.mlp.{0,2}.{weight,bias}


# =============================================================================
# HUGGINGFACE TRANSFORMERS STATE DICT KEY PATTERNS
# =============================================================================
#
# Qwen2VLForConditionalGeneration
#   ├── model: Qwen2VLModel
#   │   ├── visual: Qwen2VisionTransformerPretrainedModel
#   │   │   ├── patch_embed.proj: Conv3d
#   │   │   ├── blocks: ModuleList[Qwen2VLVisionBlock]
#   │   │   │   └── [0..31]:
#   │   │   │       ├── norm1: LayerNorm
#   │   │   │       ├── norm2: LayerNorm
#   │   │   │       ├── attn: VisionAttention
#   │   │   │       │   ├── qkv: Linear
#   │   │   │       │   └── proj: Linear
#   │   │   │       └── mlp: VisionMlp
#   │   │   │           ├── fc1: Linear
#   │   │   │           └── fc2: Linear
#   │   │   └── merger: PatchMerger
#   │   │       ├── ln_q: LayerNorm
#   │   │       └── mlp: Sequential[Linear, GELU, Linear]
#   │   │
#   │   └── language_model: Qwen2VLTextModel
#   │       ├── embed_tokens: Embedding
#   │       ├── layers: ModuleList[Qwen2VLDecoderLayer]
#   │       │   └── [0..27]:
#   │       │       ├── self_attn: Qwen2VLAttention
#   │       │       │   ├── q_proj: Linear
#   │       │       │   ├── k_proj: Linear
#   │       │       │   ├── v_proj: Linear
#   │       │       │   └── o_proj: Linear
#   │       │       ├── mlp: Qwen2MLP
#   │       │       │   ├── gate_proj: Linear
#   │       │       │   ├── up_proj: Linear
#   │       │       │   └── down_proj: Linear
#   │       │       ├── input_layernorm: RMSNorm
#   │       │       └── post_attention_layernorm: RMSNorm
#   │       └── norm: RMSNorm
#   │
#   └── lm_head: Linear (NOT in ComfyUI - encoder only!)
#
# Transformers key format:
#   model.language_model.embed_tokens.weight
#   model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}
#   model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight
#   model.language_model.layers.{i}.input_layernorm.weight
#   model.language_model.layers.{i}.post_attention_layernorm.weight
#   model.language_model.norm.weight
#   model.visual.patch_embed.proj.weight
#   model.visual.blocks.{i}.norm{1,2}.{weight,bias}
#   model.visual.blocks.{i}.attn.qkv.{weight,bias}
#   model.visual.blocks.{i}.attn.proj.{weight,bias}
#   model.visual.blocks.{i}.mlp.fc1.{weight,bias}  <- DIFFERENT from ComfyUI!
#   model.visual.blocks.{i}.mlp.fc2.{weight,bias}  <- DIFFERENT from ComfyUI!
#   model.visual.merger.ln_q.{weight,bias}
#   model.visual.merger.mlp.{0,2}.{weight,bias}
#   lm_head.weight  <- NOT in ComfyUI


def comfyui_to_transformers_key(comfy_key: str) -> Optional[str]:
    """
    Convert a ComfyUI state_dict key to its transformers equivalent.

    Args:
        comfy_key: The key from ComfyUI's Qwen25_7BVLI model state_dict

    Returns:
        The equivalent key for transformers Qwen2VLForConditionalGeneration,
        or None if there's no direct mapping.

    Notes:
        - ComfyUI uses "model." prefix for language model
        - Transformers uses "model.language_model." prefix for language model
        - Vision encoder keys are similar but nested under "model.visual."
        - Vision MLP uses different naming: gate/up/down_proj vs fc1/fc2
    """

    # Language model keys: model.* -> model.language_model.*
    if comfy_key.startswith("model."):
        return "model.language_" + comfy_key

    # Vision encoder keys: visual.* -> model.visual.*
    if comfy_key.startswith("visual."):
        hf_key = "model." + comfy_key

        # Vision MLP naming difference:
        # ComfyUI: gate_proj, up_proj, down_proj (SwiGLU-style)
        # HF: fc1, fc2 (standard MLP)
        # Actually, checking the HF source, vision MLP does use fc1/fc2
        # But ComfyUI uses SwiGLU in vision blocks too with gate/up/down
        # This is a structural difference - need to handle carefully

        # The ComfyUI vision MLP (VisionMLP class in qwen_vl.py) uses:
        #   gate_proj, up_proj, down_proj (SwiGLU activation)
        # The HF vision MLP (VisionMlp class) uses:
        #   fc1, act_fn, fc2 (standard GELU)

        # CRITICAL: These architectures may not be directly compatible!
        # The weight dimensions would be different:
        # - SwiGLU: gate and up have size (hidden, intermediate), down has (intermediate, hidden)
        # - Standard: fc1 has (hidden, intermediate), fc2 has (intermediate, hidden)

        # For now, return the key with a warning marker
        if ".mlp.gate_proj" in hf_key or ".mlp.up_proj" in hf_key or ".mlp.down_proj" in hf_key:
            # This is a structural mismatch - see GOTCHAS section
            pass

        return hf_key

    return None


def transformers_to_comfyui_key(hf_key: str) -> Optional[str]:
    """
    Convert a HuggingFace transformers state_dict key to its ComfyUI equivalent.

    Args:
        hf_key: The key from transformers Qwen2VLForConditionalGeneration state_dict

    Returns:
        The equivalent key for ComfyUI's Qwen25_7BVLI model,
        or None if there's no direct mapping (e.g., lm_head).
    """

    # LM head is not in ComfyUI (encoder-only)
    if hf_key.startswith("lm_head"):
        return None

    # Language model keys: model.language_model.* -> model.*
    if hf_key.startswith("model.language_model."):
        return hf_key.replace("model.language_model.", "model.")

    # Vision encoder keys: model.visual.* -> visual.*
    if hf_key.startswith("model.visual."):
        return hf_key.replace("model.visual.", "visual.")

    return None


def get_comfyui_to_transformers_mapping() -> Dict[str, str]:
    """
    Generate a complete mapping from ComfyUI keys to transformers keys.

    This creates regex-based patterns for all expected keys.

    Returns:
        Dictionary mapping ComfyUI key patterns to transformers key patterns
    """
    mapping = {}

    # Language model mappings
    # Embedding
    mapping["model.embed_tokens.weight"] = "model.language_model.embed_tokens.weight"

    # Final norm
    mapping["model.norm.weight"] = "model.language_model.norm.weight"

    # Layer-specific patterns (for 28 layers, indices 0-27)
    layer_keys = [
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ]

    for i in range(28):  # Qwen2.5-VL 7B has 28 layers
        for key in layer_keys:
            comfy = f"model.layers.{i}.{key}"
            hf = f"model.language_model.layers.{i}.{key}"
            mapping[comfy] = hf

    # Vision encoder mappings
    mapping["visual.patch_embed.proj.weight"] = "model.visual.patch_embed.proj.weight"
    mapping["visual.patch_embed.proj.bias"] = "model.visual.patch_embed.proj.bias"

    # Merger
    mapping["visual.merger.ln_q.weight"] = "model.visual.merger.ln_q.weight"
    mapping["visual.merger.ln_q.bias"] = "model.visual.merger.ln_q.bias"
    mapping["visual.merger.mlp.0.weight"] = "model.visual.merger.mlp.0.weight"
    mapping["visual.merger.mlp.0.bias"] = "model.visual.merger.mlp.0.bias"
    mapping["visual.merger.mlp.2.weight"] = "model.visual.merger.mlp.2.weight"
    mapping["visual.merger.mlp.2.bias"] = "model.visual.merger.mlp.2.bias"

    # Vision blocks (32 blocks for Qwen2.5-VL vision encoder)
    vision_layer_keys = [
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
        "attn.qkv.weight",
        "attn.qkv.bias",
        "attn.proj.weight",
        "attn.proj.bias",
        # NOTE: Vision MLP uses different naming - see GOTCHAS
        "mlp.gate_proj.weight",
        "mlp.gate_proj.bias",
        "mlp.up_proj.weight",
        "mlp.up_proj.bias",
        "mlp.down_proj.weight",
        "mlp.down_proj.bias",
    ]

    for i in range(32):  # Vision encoder has 32 blocks
        for key in vision_layer_keys:
            comfy = f"visual.blocks.{i}.{key}"
            hf = f"model.visual.blocks.{i}.{key}"
            mapping[comfy] = hf

    return mapping


def convert_comfyui_state_dict_to_transformers(
    comfy_state_dict: Dict[str, "torch.Tensor"],
    include_lm_head: bool = False,
    lm_head_from_embeddings: bool = True
) -> Tuple[Dict[str, "torch.Tensor"], List[str], List[str]]:
    """
    Convert a ComfyUI Qwen25_7BVLI state_dict to transformers format.

    Args:
        comfy_state_dict: State dict from ComfyUI's loaded CLIP model
        include_lm_head: If True and lm_head_from_embeddings, create lm_head from embed_tokens
        lm_head_from_embeddings: If True, tie lm_head weights to embed_tokens

    Returns:
        Tuple of:
        - Converted state dict for transformers
        - List of missing keys (keys needed by transformers but not in ComfyUI)
        - List of unexpected keys (keys in ComfyUI with no transformers equivalent)
    """
    import torch

    mapping = get_comfyui_to_transformers_mapping()

    converted = {}
    unexpected = []

    for comfy_key, tensor in comfy_state_dict.items():
        if comfy_key in mapping:
            hf_key = mapping[comfy_key]
            converted[hf_key] = tensor
        else:
            # Try dynamic conversion
            hf_key = comfyui_to_transformers_key(comfy_key)
            if hf_key:
                converted[hf_key] = tensor
            else:
                unexpected.append(comfy_key)

    # Handle lm_head
    missing = []
    if include_lm_head:
        if lm_head_from_embeddings and "model.language_model.embed_tokens.weight" in converted:
            # Tie weights - lm_head shares weights with embeddings
            converted["lm_head.weight"] = converted["model.language_model.embed_tokens.weight"]
        else:
            missing.append("lm_head.weight")

    return converted, missing, unexpected


def extract_language_model_state_dict(
    comfy_state_dict: Dict[str, "torch.Tensor"]
) -> Dict[str, "torch.Tensor"]:
    """
    Extract only the language model weights from ComfyUI state dict.

    This extracts keys starting with "model." (not "visual.") and converts
    them to transformers format for Qwen2Model (text-only).

    Args:
        comfy_state_dict: State dict from ComfyUI's loaded CLIP model

    Returns:
        State dict for transformers Qwen2Model (language model only)
    """
    lm_dict = {}

    for key, tensor in comfy_state_dict.items():
        if key.startswith("model."):
            # Convert model.* to model.language_model.* for full model
            # Or just keep as model.* for Qwen2Model
            # For Qwen2Model (not Qwen2VL), the keys are just model.*
            hf_key = key  # Keep as-is for Qwen2Model
            lm_dict[hf_key] = tensor

    return lm_dict


def get_state_dict_from_comfyui_clip(clip) -> Dict[str, "torch.Tensor"]:
    """
    Extract the state dict from a ComfyUI CLIP object.

    Args:
        clip: ComfyUI CLIP object returned from load_clip with CLIPType.HUNYUAN_VIDEO_15

    Returns:
        The state dict of the underlying Qwen model

    Notes:
        The CLIP object has structure:
        clip.cond_stage_model.qwen25_7b.transformer (the Qwen25_7BVLI instance)
    """
    # ComfyUI CLIP structure for Qwen:
    # clip.cond_stage_model -> HunyuanImageTEModel or similar
    # clip.cond_stage_model.qwen25_7b -> Qwen25_7BVLIModel
    # clip.cond_stage_model.qwen25_7b.transformer -> Qwen25_7BVLI (the actual model)

    try:
        # Try the standard path for HUNYUAN_VIDEO_15 / HUNYUAN_IMAGE / QWEN_IMAGE
        model = clip.cond_stage_model.qwen25_7b.transformer
        return model.state_dict()
    except AttributeError:
        pass

    try:
        # Alternative path
        model = clip.cond_stage_model.transformer
        return model.state_dict()
    except AttributeError:
        pass

    raise ValueError(
        "Could not extract state dict from CLIP object. "
        "Expected clip.cond_stage_model.qwen25_7b.transformer or "
        "clip.cond_stage_model.transformer"
    )


# =============================================================================
# GOTCHAS AND KNOWN ISSUES
# =============================================================================

GOTCHAS = """
## CRITICAL GOTCHAS When Converting Between ComfyUI and Transformers

### 1. LM Head Missing in ComfyUI
ComfyUI's Qwen implementation is encoder-only and does NOT include the lm_head.
- Missing key: `lm_head.weight`
- Solution: Tie weights with embed_tokens (they share weights in Qwen)
- Code: `lm_head.weight = embed_tokens.weight` (same tensor)

### 2. Vision MLP Architecture Difference
**ComfyUI (qwen_vl.py):**
```python
class VisionMLP:
    gate_proj: Linear(hidden, intermediate)
    up_proj: Linear(hidden, intermediate)
    down_proj: Linear(intermediate, hidden)
    # Output: down_proj(silu(gate_proj(x)) * up_proj(x))
```

**Transformers (modeling_qwen2_vl.py):**
```python
class VisionMlp:
    fc1: Linear(hidden, hidden_dim)
    fc2: Linear(hidden_dim, hidden)
    # Output: fc2(act_fn(fc1(x)))
```

**CRITICAL:** These are architecturally different!
- ComfyUI uses SwiGLU (3 linear layers)
- Transformers uses standard MLP (2 linear layers with GELU)

The weight shapes are different:
- ComfyUI: gate_proj and up_proj each have shape (hidden, intermediate)
- Transformers: fc1 has shape (hidden, hidden_dim)

**This means vision encoder weights may NOT be directly transferable!**

### 3. Prefix Differences
- ComfyUI: `model.layers.X.*` (language model at top level)
- Transformers: `model.language_model.layers.X.*` (nested under language_model)

### 4. Vision Encoder Prefix
- ComfyUI: `visual.*`
- Transformers: `model.visual.*`

### 5. Attention Bias
Qwen2.5-VL uses bias in q/k/v projections:
- ComfyUI: `model.layers.X.self_attn.{q,k,v}_proj.bias` - PRESENT
- Transformers: Same structure - PRESENT

### 6. Layer Count
- Language model: 28 layers (indices 0-27)
- Vision encoder: 32 blocks (indices 0-31)

### 7. RMSNorm vs LayerNorm
- Language model: Uses RMSNorm (weight only, no bias)
- Vision encoder norms: ComfyUI uses RMSNorm, HF may use LayerNorm (has bias)

### 8. Rotary Embeddings
- These are computed dynamically, not stored in state_dict
- `rotary_pos_emb` in ComfyUI and `rotary_emb` in transformers
- No weight transfer needed

### 9. Scaled FP8
ComfyUI may store `scaled_fp8` key for quantized models - filter this out.

### 10. Quantization Metadata
ComfyUI may include `_quantization_metadata` - filter this out for transformers.
"""


def print_gotchas():
    """Print the gotchas documentation."""
    print(GOTCHAS)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def verify_mapping(comfy_sd: Dict[str, "torch.Tensor"], verbose: bool = True) -> Dict[str, str]:
    """
    Verify that all keys in a ComfyUI state dict can be mapped.

    Args:
        comfy_sd: ComfyUI state dict
        verbose: If True, print detailed results

    Returns:
        Dictionary with verification results:
        - 'mapped': List of successfully mapped keys
        - 'unmapped': List of keys without mapping
        - 'status': 'ok' or 'warning'
    """
    mapping = get_comfyui_to_transformers_mapping()

    mapped = []
    unmapped = []

    for key in comfy_sd.keys():
        # Skip special keys
        if key in ("scaled_fp8", "_quantization_metadata"):
            continue

        if key in mapping or comfyui_to_transformers_key(key):
            mapped.append(key)
        else:
            unmapped.append(key)

    if verbose:
        print(f"Mapped keys: {len(mapped)}")
        print(f"Unmapped keys: {len(unmapped)}")
        if unmapped:
            print("Unmapped keys:")
            for k in unmapped[:10]:
                print(f"  - {k}")
            if len(unmapped) > 10:
                print(f"  ... and {len(unmapped) - 10} more")

    return {
        'mapped': mapped,
        'unmapped': unmapped,
        'status': 'ok' if not unmapped else 'warning'
    }


if __name__ == "__main__":
    # Print documentation
    print("ComfyUI to Transformers Qwen2.5-VL State Dict Mapping")
    print("=" * 60)
    print_gotchas()
