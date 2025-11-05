# DiT Cache Bridging: KV Cache Transfer for Diffusion Transformers

This project implements KV cache bridging techniques for Diffusion Transformer (DiT) models, enabling knowledge transfer between different transformer-based models during inference.

Based on the C2C (Cache-to-Cache) paper, adapted specifically for DiT architectures.

## Overview

**What is KV Cache Bridging?**

KV cache bridging allows transferring learned knowledge between different transformer models by projecting and fusing their Key-Value caches during generation. This enables:
- Combining strengths of different models (e.g., quality + consistency)
- Upgrading model capabilities without full re-training
- Cost-optimized inference with adaptive model selection

```python
from wan_chronoedit import WanChronoEditBridge, WanChronoEditConfig

# 1. Create configuration
config = WanChronoEditConfig(
    projector_type="gating",  # Learned gating
    fusion_direction="bidirectional",  # Combine both models
)

# 2. Load models (replace with actual model loading)
wan_model = load_wan_model()  # Your Wan2.1 loading code
chronoedit_model = load_chronoedit_model()  # Your ChronoEdit loading code

# 3. Create bridge
bridge = WanChronoEditBridge(
    config=config,
    wan_model=wan_model,
    chronoedit_model=chronoedit_model,
)

# 4. Generate with fused caches
output = bridge.generate_with_fusion(
    wan_inputs={"video_latents": ..., "text_embeddings": ...},
    chronoedit_inputs={"video_latents": ..., "text_embeddings": ...},
    target_model="wan",
    num_inference_steps=50,
)
```

### Example Script

```bash
# Run Wan-ChronoEdit examples
cd wan_chronoedit
python example.py
```

## 📁 Project Structure

```
dit_cache_bridge/
├── README.md                          # This file
├── __init__.py                        # Package initialization
│
├── c2c_adapter/                       # Core C2C components
│   ├── __init__.py
│   ├── projectors.py                  # KV cache projectors
│   └── cache_utils.py                 # Cache manipulation utilities
│
├── wan_chronoedit/                    # Wan2.1 ↔ ChronoEdit bridging
│   ├── __init__.py
│   ├── config.py                      # Configuration
│   ├── bridge.py                      # Main bridge implementation
│   └── example.py                     # Usage examples
│
├── qwen_versions/                     # Qwen2.5-VL → Qwen3-VL (planned)
│   └── ...
│
├── configs/                           # Model configurations
│   └── ...
│
├── tests/                             # Unit tests
│   └── ...
│
└── utils/                             # Shared utilities
    └── ...
```
