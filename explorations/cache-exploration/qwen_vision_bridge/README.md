# Qwen3-VL → Qwen-Image-Edit Vision Bridge

**Inject Qwen3-VL's superior vision understanding into Qwen-Image-Edit for enhanced capabilities.**

## Goal

Enable new downstream capabilities in image editing by bridging Qwen3-VL's advanced vision understanding to Qwen-Image-Edit's generation pipeline.

### **Pursuing Capabilities:**

1. **OCR-Aware Editing** (32 languages vs limited)
   - Detect and modify text in images correctly
   - Multi-lingual text editing
   - Preserve text formatting during edits

2. **Enhanced Spatial Reasoning**
   - Complex scene rearrangements ("move object from left to right")
   - Better object positioning and placement
   - Improved depth understanding

3. **Fine Detail Preservation**
   - DeepStack multi-level features capture details at multiple scales
   - Better texture and pattern preservation
   - Maintain small objects during large edits

4. **Multi-Image Consistency**
   - 256K context vs standard 32K
   - Better understanding of image relationships
   - Consistent editing across image sequences

---

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│  Input: Image + Edit Prompt     │
└────────────┬────────────────────┘
             │
     ┌───────▼────────┐
     │  Qwen3-VL 9B   │  ← Superior Vision Understanding
     │  - DeepStack   │     • OCR (32 langs)
     │  - Spatial     │     • Spatial reasoning
     │  - 256K ctx    │     • Fine details
     └───────┬────────┘
             │ Extract multi-level features
             │ (early, mid, late)
             │
     ┌───────▼────────┐
     │ Vision Bridge  │  ← Our Contribution (~5M params)
     │  - Project     │     • Qwen3 (9B) → Qwen2.5 (7B) dimensions
     │  - Fuse        │     • Multi-level fusion
     │  - Enhance     │     • Attention-based blending
     └───────┬────────┘
             │
     ┌───────▼────────┐
     │ Qwen2.5-VL 7B  │  ← Baseline (current in Image-Edit)
     │  (augmented)   │
     └───────┬────────┘
             │ Enhanced vision cache
             │
     ┌───────▼────────┐
     │ Qwen-Image-Edit│  ← Image Generation (20B DiT)
     │  DiT Generator │     • VAE + DiT unchanged
     └───────┬────────┘
             │
             ▼
    Better Edited Image
    (with Qwen3's capabilities!)
```

---

## Structure

### **Cache Extraction** (`cache_extraction.py`)

Extract vision caches from both Qwen3-VL and Qwen2.5-VL.

```python
from cache_extraction import VisionCacheExtractor

extractor = VisionCacheExtractor(
    device="cuda",
    load_in_8bit=True  # For memory efficiency
)

# Extract Qwen3's DeepStack multi-level features
qwen3_cache = extractor.extract_qwen3_vision(
    image=image,
    text_prompt="Describe this image"
)
# Returns: early, mid, late features + full cache

# Extract Qwen2.5 baseline features
qwen25_cache = extractor.extract_qwen25_vision(
    image=image,
    text_prompt="Describe this image"
)
# Returns: single-level features + cache

# Compare architectures
comparison = extractor.compare_caches(qwen3_cache, qwen25_cache)
```

**Key Features:**
- Lazy model loading (only load when needed)
- 8-bit quantization support (save memory)
- DeepStack feature extraction (Qwen3 multi-level)
- Automatic dimension detection

### **Vision Bridge** (`vision_bridge.py`)

Project and fuse Qwen3's features into Qwen2.5 format.

```python
from vision_bridge import VisionCacheBridge, BridgeConfig

# Create bridge with default config
bridge = VisionCacheBridge()

# Or customize
config = BridgeConfig(
    qwen3_hidden_dim=4096,  # 9B model
    qwen25_hidden_dim=3584,  # 7B model
    use_attention_fusion=True,  # Attention vs MLP
    use_residual=True,  # Blend with baseline
)
bridge = VisionCacheBridge(config)

# Bridge caches
enhanced_features, attention_weights = bridge(
    qwen3_cache=qwen3_cache,
    qwen25_cache=qwen25_cache,
    return_attention_weights=True,
)
# Output: Qwen2.5-compatible features with Qwen3's understanding
```

**Architecture Details:**
- **DeepStackProjector**: Projects each level (early, mid, late) independently
- **MultiLevelFusion**: Fuses multi-level features via attention or MLP
- **Residual Blending**: Learned weight between enhanced and baseline
- **~5M parameters**: Lightweight and fast

### **Pipeline** (`enhanced_pipeline.py`)

Integrate bridge with Qwen-Image-Edit.

```python
from enhanced_pipeline import EnhancedQwenImageEdit

# Create enhanced pipeline
model = EnhancedQwenImageEdit(
    device="cuda",
    load_in_8bit=True,  # Save memory
)

# Generate with Qwen3 vision enhancement
output = model.generate(
    image=input_image,
    prompt="Change the text 'Hello' to 'Goodbye'",
    use_qwen3_vision=True,  # Enable enhancement
)

# Compare baseline vs enhanced
baseline, enhanced, attention_maps = model.compare(
    image=input_image,
    prompt="Change the text 'Hello' to 'Goodbye'",
    save_comparison="comparison.png"
)

# Benchmark on test set
results = model.benchmark(
    test_images=[img1, img2, img3],
    test_prompts=["prompt1", "prompt2", "prompt3"],
    output_dir="benchmark_results"
)
```

### **Module 4: Evaluation** (`evaluation.py`) ✅ COMPLETE

Test if bridge actually enables new capabilities.

```python
from evaluation import CapabilityEvaluator

evaluator = CapabilityEvaluator()

# Test OCR capabilities
ocr_results = evaluator.evaluate_ocr_capability()
# Measures: Can it detect and edit text correctly?

# Test spatial reasoning
spatial_results = evaluator.evaluate_spatial_reasoning()
# Measures: Can it reposition objects accurately?

# Test detail preservation
detail_results = evaluator.evaluate_detail_preservation()
# Measures: Are textures and fine details maintained?

# Run complete evaluation
all_results = evaluator.evaluate_all()
# Returns: Dictionary with results for all categories
```

### **Extract Vision Caches**

```python
python example_extraction.py
```

This will:
1. Load Qwen3-VL (9B) and Qwen2.5-VL (7B)
2. Extract vision caches from both models
3. Compare cache structures
4. Analyze DeepStack multi-level features
5. Estimate bridge parameter count

**Expected Output:**
```
Loading Qwen3-VL model...
  Model loaded: Qwen/Qwen3-VL-8B-Instruct
  Layers: 32
  Hidden dim: 4096

Extracting Qwen3-VL vision cache...
  Detected 256 vision tokens
  Early features: torch.Size([1, 256, 4096])
  Mid features: torch.Size([1, 256, 4096])
  Late features: torch.Size([1, 256, 4096])
Qwen3-VL cache extracted successfully

Cache Comparison:
  Vision token count match: True
  Dimension mismatch: True
  Dimension ratio: 1.14

Total bridge parameters: ~5.2M
```

### **Test Bridge (Zero-Shot)**

```python
from cache_extraction import VisionCacheExtractor
from vision_bridge import create_default_bridge
from PIL import Image

# Load models and extractor
extractor = VisionCacheExtractor(device="cuda", load_in_8bit=True)
bridge = create_default_bridge()

# Load test image
image = Image.open("test_image.jpg")

# Extract caches
qwen3_cache = extractor.extract_qwen3_vision(image)
qwen25_cache = extractor.extract_qwen25_vision(image)

# Bridge (zero-shot, no training!)
enhanced_features, attention_weights = bridge(
    qwen3_cache, qwen25_cache, return_attention_weights=True
)

print(f"Enhanced features: {enhanced_features.shape}")
print(f"Attention weights: {attention_weights.shape if attention_weights else 'N/A'}")
```

**Optimization Notes:**
```python
# Use 8-bit quantization
extractor = VisionCacheExtractor(load_in_8bit=True)

# Unload models after extraction to save memory
del extractor._qwen3_model
torch.cuda.empty_cache()
```

---

## Improvements to look for (but maybe none)

Based on Qwen3-VL's documented improvements over Qwen2.5-VL:

### **OCR Tasks:**
- **Text detection accuracy**: +20-30% (especially multi-lingual)
- **Text edit accuracy**: Enables capabilities that don't work with Qwen2.5

### **Spatial Reasoning:**
- **Object positioning**: +15-25% accuracy
- **Scene understanding**: Better spatial relationships

### **Detail Preservation:**
- **Texture similarity**: +10-15% in non-edited regions
- **Small object preservation**: Fewer artifacts on fine details

### **Speed:**
- **Bridge overhead**: <5% additional latency
- **Memory overhead**: ~1-2 GB for cached features

---

## Training the Bridge (TBD)

```python
from vision_bridge import VisionCacheBridge, BridgeConfig

# Create bridge
config = BridgeConfig(learning_rate=1e-4, weight_decay=0.01)
bridge = VisionCacheBridge(config)

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    for batch in dataloader:
        image, target = batch

        # Extract caches
        qwen3_cache = extractor.extract_qwen3_vision(image)
        qwen25_cache = extractor.extract_qwen25_vision(image)

        # Bridge
        enhanced = bridge(qwen3_cache, qwen25_cache)

        # Compute loss (e.g., MSE with target, or task-specific loss)
        loss = compute_loss(enhanced, target)

        # Backward
        loss.backward()
        optimizer.step()

# Save trained bridge
bridge.save("bridge_weights.pt")
```

**Training data requirements:**
- Dataset: 1K-5K image-text pairs
- Tasks: OCR, spatial edits, detail preservation

### **Module-Specific Examples**

**Cache Extraction** (`example_extraction.py`):
1. Basic cache extraction and comparison
2. DeepStack multi-level feature analysis
3. Dimension mismatch analysis for bridge design
4. Memory usage estimation

**Pipeline Usage** (in code):
```python
from enhanced_pipeline import EnhancedQwenImageEdit

# Initialize
pipeline = EnhancedQwenImageEdit(load_in_8bit=True)

# Generate with enhancement
output = pipeline.generate(
    image="input.jpg",
    prompt="Change text to 'Hello World'",
    use_qwen3_vision=True
)

# Compare baseline vs enhanced
baseline, enhanced, attention = pipeline.compare(
    image="input.jpg",
    prompt="Change text to 'Hello World'",
    save_comparison="result.png"
)
```

---

## References

- **Qwen3-VL**: Enhanced vision-language model with DeepStack
- **Qwen2.5-VL**: Current vision encoder in Qwen-Image-Edit
- **Qwen-Image-Edit**: Image editing with VLM + DiT architecture
- **C2C Paper**: Cache-to-Cache bridging technique
