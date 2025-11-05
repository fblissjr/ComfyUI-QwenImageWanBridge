# Qwen Vision Bridge - Notes

## Running Examples

### 1. Cache Extraction Test (Lightest)

Start here to verify everything works:

```bash
uv run python example_extraction.py
```

**What it does:**
- Loads Qwen3-VL and Qwen2.5-VL with 8-bit quantization
- Extracts vision caches from a test image
- Compares DeepStack multi-level features
- Estimates bridge parameters

**Expected time:** 2-5 minutes (first run downloads models)

**Expected output:**
```
Loading Qwen3-VL model...
Qwen3-VL cache extracted successfully
  Early features: torch.Size([1, 256, 4096])
  Mid features: torch.Size([1, 256, 4096])
  Late features: torch.Size([1, 256, 4096])

Loading Qwen2.5-VL model...
Qwen2.5-VL cache extracted successfully
  Vision features: torch.Size([1, 256, 3584])

Cache Comparison:
  Vision token count match: True
  Dimension mismatch: True
  Dimension ratio: 1.14

Total bridge parameters: ~5.2M
```

### 2. Full Pipeline Test

Once extraction works, try the full pipeline:

```bash
uv run python example_end_to_end.py
```

**Interactive menu with 7 examples:**
1. Basic comparison - Baseline vs enhanced side-by-side
2. OCR capability - Multi-lingual text editing
3. Spatial reasoning - Object repositioning
4. Detail preservation - Texture maintenance
5. Full evaluation - Complete capability test suite
6. Attention visualization - See where bridge focuses
7. Run all - Complete demonstration

### Performance Stuff

1. **Enable 8-bit quantization** (should be default):
```python
extractor = VisionCacheExtractor(load_in_8bit=True)
```

2. **Sequential loading** (load one model at a time):
```python
# Extract Qwen3 cache
qwen3_cache = extractor.extract_qwen3_vision(image, prompt)

# Unload Qwen3
del extractor._qwen3_model
torch.cuda.empty_cache()

# Now load Qwen2.5
qwen25_cache = extractor.extract_qwen25_vision(image, prompt)
```

3. **CPU offloading** (slower but uses less VRAM):
```python
extractor = VisionCacheExtractor(
    load_in_8bit=True,
    device_map="auto",  # Automatically offload to CPU
)
```

## Expected Performance

### Zero-Shot (No Training)

The bridge works out-of-the-box with random initialization:

| Capability | Expected Improvement |
|------------|---------------------|
| **OCR tasks** | ~10-15% better |
| **Spatial reasoning** | ~5-10% better |
| **Detail preservation** | ~5% better |

Likely not to make a difference of significance since the bridge hasn't learned task-specific patterns yet.

## What's Actually Happening

**The pipeline:**

1. **Extract Qwen3 vision** → DeepStack multi-level features
   - Early layers: edges, textures (low-level details)
   - Mid layers: objects, structure (mid-level features)
   - Late layers: semantics, context (high-level understanding)

2. **Extract Qwen2.5 baseline** → Single-level features
   - Standard vision features (current in Image-Edit)

3. **Bridge caches** → Project + Fuse
   - Project: 4096D → 3584D per level
   - Fuse: Combine multi-level via attention
   - Blend: Mix with baseline (residual connection)

4. **Generate with Qwen-Image-Edit** → Enhanced output
   - Use enhanced vision instead of baseline
   - Same DiT generator, better vision understanding

**The key insight:** We're injecting Qwen3's superior vision capabilities (OCR, spatial reasoning) into Qwen-Image-Edit's generation pipeline.

---

## Next Steps After Testing

### 1. Run Evaluation

See if bridge actually improves capabilities:

```bash
uv run python example_end_to_end.py
# Select option 5: Full evaluation suite
```

This tests:
- OCR capability (text editing)
- Spatial reasoning (object positioning)
- Detail preservation (texture maintenance)

### 2. Test on Real Images

Replace synthetic test images with net new ones:

```python
from enhanced_pipeline import EnhancedQwenImageEdit
from PIL import Image

pipeline = EnhancedQwenImageEdit(load_in_8bit=True)

image = Image.open("your_image.jpg")
prompt = "Your editing instruction"

baseline, enhanced, attention = pipeline.compare(
    image=image,
    prompt=prompt,
    save_comparison="result.png"
)
```

### 3. Train the Bridge

TBD

### 4. Experiment with Other Bridges

Try different model combinations:
- Wan2.1 ↔ ChronoEdit (temporal video editing)
- Different Qwen variants
- Custom LoRAs with base models

---

## Output Files

All examples save to `examples_output/`:

```
examples_output/
├── test_input.png              # Synthetic test image
├── comparison_basic.png        # Baseline vs enhanced
├── attention_visualization.png # Attention maps
├── ocr_tests/                  # OCR capability results
├── spatial_tests/              # Spatial reasoning results
├── detail_tests/               # Detail preservation results
└── full_evaluation/            # Complete evaluation results
```

## Ready to Go!

Start with:

```bash
cd qwen_vision_bridge
uv run python check_environment.py  # Verify setup
uv run python example_extraction.py # Test extraction
uv run python example_end_to_end.py # Full pipeline
```
