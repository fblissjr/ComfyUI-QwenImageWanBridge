# DiT Cache Bridging: KV Cache Transfer for Diffusion Transformers

This project implements KV cache bridging techniques for Diffusion Transformer (DiT) models, enabling knowledge transfer between different transformer-based models during inference.

Based on the C2C (Cache-to-Cache) paper, adapted specifically for DiT architectures.

## 📋 Overview

**What is KV Cache Bridging?**

KV cache bridging allows transferring learned knowledge between different transformer models by projecting and fusing their Key-Value caches during generation. This enables:
- Combining strengths of different models (e.g., quality + consistency)
- Upgrading model capabilities without full re-training
- Cost-optimized inference with adaptive model selection

**Supported Scenarios:**

| Scenario | Feasibility | Status | Priority |
|----------|-------------|--------|----------|
| **Wan2.1 ↔ ChronoEdit** | 9/10 ⭐⭐⭐ | ✅ Implemented | **HIGHEST** 🚀 |
| **Replace UMT5 w/ Causal LM** | 7/10 | 🔄 Planned | **MEDIUM-HIGH** ⭐ |
| **Qwen2.5-VL → Qwen3-VL** | 6/10 | 🔄 Planned | **MEDIUM** 🔶 |
| **Qwen Image ↔ Wan** | 2/10 | ❌ Not Recommended | **DO NOT PURSUE** |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository_url>
cd dit_cache_bridge

# Install dependencies
pip install torch transformers accelerate

# Optional: Install model-specific dependencies
pip install diffusers  # For diffusion models
```

### Basic Usage: Wan2.1 ↔ ChronoEdit

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

## 🎯 Scenario 1: Wan2.1 ↔ ChronoEdit (Highest Priority)

### Why This Works

ChronoEdit is built **directly on top of Wan2.1-I2V-14B** with identical architecture:
- ✅ Same layer count: 40 layers
- ✅ Same dimensions: 5120 hidden, 40 heads, 128 head_dim
- ✅ Same text encoder: UMT5-XXL
- ✅ Only difference: Training objectives (generation vs editing)

**Result:** Near-perfect architectural compatibility → Minimal projection overhead!

### Key Benefits

1. **Quality + Consistency Fusion**
   - Wan2.1: Excellent visual quality, diverse generation
   - ChronoEdit: Superior temporal consistency, edit precision
   - Fused: Best of both worlds

2. **Two-Stage Generation**
   - Fast prototyping with Wan2.1
   - Refinement with ChronoEdit temporal reasoning

3. **Physics-Aware Generation**
   - ChronoEdit's temporal patterns → Wan2.1
   - Enables physically plausible video generation

### Configuration Options

```python
WanChronoEditConfig(
    # Projector type
    projector_type="gating",  # "identity", "weighted", or "gating"

    # Gating configuration (if projector_type="gating")
    projector_hidden_dim=256,
    gate_granularity="head",  # "head", "token", or "value"

    # Weighted fusion (if projector_type="weighted")
    fusion_alpha=0.5,  # Balance between models
    learnable_alpha=True,  # Learn optimal weight

    # Fusion direction
    fusion_direction="bidirectional",  # "wan_to_chrono", "chrono_to_wan", or "bidirectional"

    # Training
    learning_rate=1e-4,
    weight_decay=0.01,
)
```

### Training the Projector

```python
# 1. Prepare dataset
# Collect paired outputs from Wan2.1 and ChronoEdit

# 2. Train projector
bridge.train_projector(
    dataloader=train_dataloader,
    num_epochs=10,
)

# 3. Save trained weights
bridge.save_projectors("wan_chrono_projector.pt")

# 4. Load for inference
bridge.load_projectors("wan_chrono_projector.pt")
```

**Training Requirements:**
- Dataset: ~1K-10K video-text pairs
- Training time: Hours to days (not weeks!)
- Trainable params: ~100K-1M (projector only, models frozen)
- Hardware: Single GPU sufficient

### Use Cases

**Use Case 1: Sunset Video with Smooth Transitions**
```python
config = WanChronoEditConfig(
    projector_type="gating",
    fusion_direction="bidirectional",
)

output = bridge.generate_with_fusion(
    prompt="Beautiful sunset over ocean with smooth color transitions",
    ...
)
# Result: Wan's visual quality + ChronoEdit's temporal smoothness
```

**Use Case 2: Fast Iteration → Quality Refinement**
```python
# Stage 1: Quick Wan2.1 generation (low steps)
wan_output = wan_model.generate(prompt, num_steps=20)

# Stage 2: Bridge to ChronoEdit for refinement
config = WanChronoEditConfig(fusion_direction="wan_to_chrono")
refined = bridge.generate_with_fusion(..., num_steps=10)
```

**Use Case 3: Physics-Aware Generation**
```python
config = WanChronoEditConfig(
    fusion_direction="chrono_to_wan",  # ChronoEdit provides physics
    gate_granularity="token",  # Fine-grained control
)

output = bridge.generate_with_fusion(
    prompt="Ball bouncing down stairs with realistic physics",
    ...
)
```

## 🔧 Core Components

### Projectors

Four projector types for different scenarios:

**1. IdentityProjector**
- For identical architectures
- No transformation, just pass-through
- Minimal overhead

**2. WeightedFusionProjector**
```python
output = alpha * source + (1 - alpha) * target
```
- Simple weighted combination
- Alpha can be learned or fixed

**3. LearnedGatingProjector** (Recommended for Wan↔ChronoEdit)
```python
gate = gate_network(target_kv)
output = target + gate * (source - target)
```
- Context-dependent fusion
- Learns when to use source vs target
- Granularity: per-head, per-token, or per-value

**4. FullProjector**
- For mismatched architectures
- Projects head counts and dimensions
- Required for Qwen2.5-VL → Qwen3-VL

### Cache Utilities

**DynamicCache**: Standard KV cache structure
```python
cache = DynamicCache(
    key_cache=[...],    # List of (B, H, N, D) tensors
    value_cache=[...],  # One per layer
)
```

**extract_kv_cache**: Extract from model forward pass
```python
cache = extract_kv_cache(model, inputs)
```

**fuse_caches**: Combine two caches with projector
```python
fused = fuse_caches(source_cache, target_cache, projector)
```

## 📊 Performance Expectations

### Wan2.1 ↔ ChronoEdit

| Metric | Wan2.1 Only | ChronoEdit Only | Fused (Expected) |
|--------|-------------|-----------------|------------------|
| Visual Quality | 9/10 | 7/10 | **9/10** |
| Temporal Consistency | 7/10 | 9/10 | **9/10** |
| Generation Speed | 1x | 1x | **1.1x** (minimal overhead) |

**Computational Overhead:**
- Prefill: 2x (run both models)
- Decode: 1x (single model)
- Projector: <1% overhead
- **Overall:** ~10-20% slower for significant quality gains

## 🛠️ Development Roadmap

### Phase 1: Wan2.1 ↔ ChronoEdit ✅ (Current)

**Status: Implemented**

- [x] Core projector implementations
- [x] Cache utilities
- [x] Wan-ChronoEdit bridge class
- [x] Configuration system
- [x] Example scripts
- [ ] Unit tests
- [ ] Integration with actual Wan/ChronoEdit models
- [ ] Training pipeline
- [ ] Benchmark suite

**Timeline: 2-4 weeks for full implementation**

### Phase 2: Replace UMT5 with Causal LM 🔄 (Planned)

**Goal:** Enable LLM→DiT text cache bridging

- [ ] Architecture modifications for Wan
- [ ] Qwen2.5-0.5B integration as text encoder
- [ ] Adapter training pipeline
- [ ] Text understanding benchmarks
- [ ] Optional: Full Wan fine-tuning

**Timeline: 1-3 months**

### Phase 3: Qwen2.5-VL → Qwen3-VL 🔄 (Planned)

**Goal:** Version migration with cache transfer

- [ ] Head projection layer (28→32 heads)
- [ ] DeepStack multi-layer handling
- [ ] Training on paired Qwen outputs
- [ ] Long-video understanding tests

**Timeline: 2-4 months**

## 📚 Documentation

### Full Analysis Document

See [`dit_cache_bridging_analysis.md`](../dit_cache_bridging_analysis.md) for comprehensive technical analysis including:
- C2C mechanism explanation
- Architecture compatibility analysis
- Detailed feasibility assessments
- Implementation strategies
- Training requirements

### API Reference

(Coming soon)

### Training Guide

(Coming soon)

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run Wan-ChronoEdit examples
python wan_chronoedit/example.py
```

## 🤝 Contributing

This is a research prototype. Contributions welcome!

Areas for contribution:
- Integration with actual model implementations
- Training pipelines and datasets
- Benchmark suites
- Additional bridging scenarios
- Performance optimizations

## 📄 License

[Add license information]

## 🙏 Acknowledgments

- **C2C Paper:** Original cache bridging technique
- **Wan2.1:** Base video generation model
- **ChronoEdit:** Temporal video editing framework
- **HuggingFace Transformers:** Model infrastructure

## 📞 Contact

[Add contact information]

## 🗺️ Roadmap Summary

**Immediate (Weeks):**
- ✅ Core implementation
- 🔄 Testing and validation
- 🔄 Integration with actual models
- 🔄 Training pipeline

**Short-term (Months):**
- UMT5 replacement exploration
- Qwen2.5→3 bridging
- Performance optimization
- Documentation completion

**Long-term (6+ months):**
- Additional DiT model support
- Multi-model fusion (>2 models)
- Real-time optimization
- Production deployment

---

**Status:** Research Prototype v0.1.0

**Last Updated:** 2025-11-05

**Next Milestone:** Testing and model integration
