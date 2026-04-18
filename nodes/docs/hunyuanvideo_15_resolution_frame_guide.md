# HunyuanVideo 1.5 Resolution & Frame Guide

Complete technical reference for resolutions, aspect ratios, frame counts, and recommended settings based on the official HunyuanVideo 1.5 codebase.

---

## Base Resolution Configurations

HunyuanVideo 1.5 uses a **bucketing system** where resolutions are dynamically adjusted based on aspect ratio while maintaining consistent total pixel count per resolution tier.

### Resolution Tiers

| Tier | Base Size | Stride | Typical 16:9 Example | Total Pixels (approx) |
|------|-----------|--------|---------------------|----------------------|
| 360p | 480       | 16     | 640×360             | 230K                 |
| 480p | 640       | 16     | 848×480             | 407K                 |
| 720p | 960       | 16     | 1280×720            | 922K                 |
| 1080p| 1440      | 16     | 1920×1080           | 2.07M                |

**Base Size:** The foundation dimension for bucket generation
**Stride:** Resolution must be divisible by 16 (VAE compression factor of 8, latent alignment)

---

## Supported Aspect Ratios

The bucketing system supports **any aspect ratio up to 4:1** (or 1:4), dynamically calculating the closest valid resolution.

### Common Aspect Ratios

| Aspect Ratio | 480p Example    | 720p Example    | 1080p Example   | Use Case          |
|--------------|-----------------|-----------------|-----------------|-------------------|
| 16:9         | 848×480         | 1280×720        | 1920×1080       | Cinematic (default)|
| 9:16         | 480×848         | 720×1280        | 1080×1920       | Portrait/mobile   |
| 1:1          | 672×672         | 960×960         | 1440×1440       | Square/social     |
| 4:3          | 640×480         | 960×720         | 1440×1080       | Standard video    |
| 21:9         | 1120×480        | 1680×720        | 2520×1080       | Ultra-wide cinema |
| 3:4          | 480×640         | 720×960         | 1080×1440       | Portrait photo    |

**Note:** Actual dimensions are rounded to nearest multiple of 16 (stride requirement).

### Aspect Ratio Limits

- **Maximum ratio:** 4:1 (e.g., 2560×640 for 480p)
- **Minimum ratio:** 1:4 (e.g., 640×2560 for 480p)
- Ratios beyond this range are clamped to 4:1 or 1:4

---

## Video Length (Frame Count)

### Default Configuration

**Default video_length:** `121 frames`

This provides **~5 seconds** of video at 24 FPS (typical cinematic frame rate).

### Frame Count Recommendations

| Frames | Duration @ 24fps | Duration @ 30fps | Use Case              | VRAM Impact |
|--------|------------------|------------------|-----------------------|-------------|
| 25     | ~1.0s            | ~0.8s            | Very short clips      | Low         |
| 49     | ~2.0s            | ~1.6s            | Short animations      | Low-Med     |
| 73     | ~3.0s            | ~2.4s            | Medium clips          | Medium      |
| 97     | ~4.0s            | ~3.2s            | Standard clips        | Med-High    |
| 121    | ~5.0s            | ~4.0s            | **Default (recommended)** | High    |
| 145    | ~6.0s            | ~4.8s            | Longer clips          | Very High   |
| 169    | ~7.0s            | ~5.6s            | Extended clips        | Very High   |

**Notes:**
- Frame count must be **odd numbers** (1, 3, 5, 7, ..., 119, 121, 123, ...)
- Longer videos = exponentially higher VRAM requirements
- Temporal compression ratio: 8x (121 frames → 16 latent frames)

### Frame Count Formula

```
latent_frames = (video_length - 1) // 8 + 1
```

For 121 frames:
```
latent_frames = (121 - 1) // 8 + 1 = 16
```

---

## Model Versions & Configurations

**Note:** This node pack currently supports T2V (text-to-video) only. I2V (image-to-video) is not implemented.

### Standard Models

| Version | Resolution | Task | Flow Shift | Guidance Scale | Steps | Description |
|---------|------------|------|------------|----------------|-------|-------------|
| 480p_t2v | 480p | T2V | 5.0 | 6.0 | 50 | Text-to-video standard |
| 720p_t2v | 720p | T2V | 9.0 | 6.0 | 50 | High-res T2V standard |

### Distilled Models (Fast)

| Version | Resolution | Task | Flow Shift | Guidance Scale | Steps | Description |
|---------|------------|------|------------|----------------|-------|-------------|
| 480p_t2v_distilled | 480p | T2V | 5.0 | 1.0 | 25-30 | Fast T2V |
| 720p_t2v_distilled | 720p | T2V | 9.0 | 1.0 | 25-30 | Fast high-res T2V |

### Sparse Attention Models

| Version | Resolution | Task | Flow Shift | GPU Requirement | Description |
|---------|------------|------|------------|-----------------|-------------|
| 720p_t2v_distilled_sparse | 720p | T2V | 7.0 | H100 only | Sparse attention T2V |

**Note:** Sparse attention models require NVIDIA H100 GPU and `flex-block-attn` library.

---

## Super-Resolution Upsampling

HunyuanVideo 1.5 includes built-in super-resolution for upsampling after base generation.

### SR Configurations

| SR Version | Base Resolution | Target Resolution | Flow Shift | Steps | Guidance Scale |
|------------|----------------|-------------------|------------|-------|----------------|
| 720p_sr_distilled | 480p | 720p | 2.0 | 6 | 1.0 |
| 1080p_sr_distilled | 720p | 1080p | 2.0 | 8 | 1.0 |

### SR Resolution Mapping

```
480p_t2v → 720p_sr_distilled → 720p output
480p_i2v → 720p_sr_distilled → 720p output
720p_t2v → 1080p_sr_distilled → 1080p output
720p_i2v → 1080p_sr_distilled → 1080p output
```

### SR Workflow Example

```
1. Generate 480p video (121 frames)
2. Apply 720p_sr_distilled upsampler
3. Output: 720p video (121 frames)

Total: 480p generation (50 steps) + 720p SR (6 steps) = 56 steps
```

---

## VRAM Requirements

### Estimated VRAM by Configuration

#### 480p Standard (121 frames, 50 steps)

| Task | VRAM (with tiling) | VRAM (no tiling) | Batch Size |
|------|--------------------|------------------|------------|
| T2V  | 8-10 GB            | 12-14 GB         | 1          |
| I2V  | 10-12 GB           | 14-16 GB         | 1          |

#### 720p Standard (121 frames, 50 steps)

| Task | VRAM (with tiling) | VRAM (no tiling) | Batch Size |
|------|--------------------|------------------|------------|
| T2V  | 14-16 GB           | 20-24 GB         | 1          |
| I2V  | 16-18 GB           | 24-28 GB         | 1          |

#### Distilled Models (121 frames, 25-30 steps)

| Resolution | Task | VRAM (with tiling) | VRAM (no tiling) |
|------------|------|--------------------|------------------|
| 480p       | T2V/I2V | 6-8 GB      | 10-12 GB         |
| 720p       | T2V/I2V | 12-14 GB    | 18-22 GB         |

### VRAM Optimization Settings

**VAE Tiling (< 23GB VRAM):**
- `sample_size`: 160
- `tile_overlap_factor`: 0.2
- `dtype`: float16

**No Tiling (≥ 23GB VRAM):**
- `sample_size`: 256
- `tile_overlap_factor`: 0.25
- `dtype`: float32

---

## Recommended Configurations

### For Testing/Preview (Low VRAM)

```
Resolution: 480p
Model: 480p_t2v_distilled or 480p_i2v_distilled
Frames: 73 (3 seconds)
Steps: 25
Guidance Scale: 1.0
Flow Shift: 5.0
VRAM: ~6-8GB
```

### For Production (Balanced)

```
Resolution: 480p → 720p (with SR)
Model: 480p_t2v + 720p_sr_distilled
Frames: 121 (5 seconds)
Steps: 50 (generation) + 6 (SR)
Guidance Scale: 6.0 (generation), 1.0 (SR)
Flow Shift: 5.0 (generation), 2.0 (SR)
VRAM: ~10-14GB
```

### For High Quality (High VRAM)

```
Resolution: 720p → 1080p (with SR)
Model: 720p_t2v + 1080p_sr_distilled
Frames: 121 (5 seconds)
Steps: 50 (generation) + 8 (SR)
Guidance Scale: 6.0 (generation), 1.0 (SR)
Flow Shift: 9.0 (T2V) or 7.0 (I2V)
VRAM: ~18-28GB
```

### For Ultra Quality (24GB+ VRAM)

```
Resolution: 720p native (no SR)
Model: 720p_t2v or 720p_i2v
Frames: 145-169 (6-7 seconds)
Steps: 50
Guidance Scale: 6.0
Flow Shift: 9.0 (T2V) or 7.0 (I2V)
VRAM: ~24-32GB
VAE Tiling: Disabled
```

---

## Resolution Selection Guide

### By Use Case

**Social Media (TikTok, Instagram Reels):**
- Aspect Ratio: 9:16 (portrait)
- Resolution: 480p (540×960) or 720p (810×1440)
- Frames: 73-97 (3-4 seconds)

**YouTube Shorts:**
- Aspect Ratio: 9:16 (portrait)
- Resolution: 720p (810×1440) with 1080p SR
- Frames: 121 (5 seconds)

**Cinematic/Film:**
- Aspect Ratio: 16:9 or 21:9
- Resolution: 720p or 1080p
- Frames: 121-145 (5-6 seconds)

**Product Demo:**
- Aspect Ratio: 1:1 or 4:3
- Resolution: 480p or 720p
- Frames: 49-73 (2-3 seconds)

**Experimental/Art:**
- Aspect Ratio: Custom (within 4:1 limit)
- Resolution: 480p distilled (fast iteration)
- Frames: 25-49 (1-2 seconds)

---

## Parameter Tuning Guide

### Flow Shift

Controls the diffusion schedule dynamics.

| Resolution | T2V | I2V | Effect |
|------------|-----|-----|--------|
| 480p       | 5.0 | 5.0 | Standard quality |
| 720p       | 9.0 | 7.0 | Higher resolution requires more shift |

**Too low:** May produce artifacts, inconsistent motion
**Too high:** Over-smoothed, loss of detail

### Guidance Scale

Controls prompt adherence vs. creativity.

| Model Type | Recommended | Effect |
|------------|-------------|--------|
| Standard   | 6.0         | Balanced prompt adherence |
| Distilled  | 1.0         | Distilled models don't need high CFG |

**Higher (7-10):** Stronger prompt adherence, may reduce naturalness
**Lower (3-5):** More creative, may deviate from prompt

### Steps

Number of diffusion sampling steps.

| Model Type | Minimum | Recommended | Maximum | Quality Gain |
|------------|---------|-------------|---------|--------------|
| Standard   | 30      | 50          | 80      | Diminishing returns after 50 |
| Distilled  | 20      | 25-30       | 40      | Optimized for fewer steps |
| SR         | 4       | 6-8         | 12      | Upsampling quality |

**Fewer steps:** Faster, potential quality loss
**More steps:** Slower, marginal quality improvement after recommended

---

## Technical Constraints

### Hard Limits

- **Minimum dimension:** 16 pixels (stride constraint)
- **Dimension divisibility:** Must be multiple of 16
- **Maximum aspect ratio:** 4:1 or 1:4
- **Temporal compression:** 8x (8 video frames = 1 latent frame)
- **Minimum video length:** 1 frame (technically, but 25+ recommended)
- **Frame count parity:** Must be odd (1, 3, 5, 7, ...)

### Soft Limits (Recommended)

- **Maximum 480p dimensions:** ~1600×900 (4:1 ratio limit)
- **Maximum 720p dimensions:** ~2400×1350 (4:1 ratio limit)
- **Practical video length:** 25-169 frames (1-7 seconds)
- **Production video length:** 121 frames (5 seconds, default)

---

## Workflow Integration

### ComfyUI Workflow

When using our HunyuanVideo encoder nodes with ComfyUI's native sampler:

1. **HunyuanVideoCLIPLoader** - Loads Qwen2.5-VL + optional byT5
2. **HunyuanVideoTextEncoder** - Encodes prompt with 39 templates
   - Outputs: `positive`, `negative` (both connect to sampler)
   - Optional: `template_preset` dropdown, `additional_instructions`
3. **EmptyLatentVideo** (native) - Create latent with:
   - `width`: Based on resolution tier (see table above)
   - `height`: Based on resolution tier
   - `length`: 121 (default) or custom frame count
   - `batch_size`: 1 (multi-batch not supported)
4. **KSampler** (native) - Generate
5. **VAEDecode** (native) - Decode frames

### Resolution Selection Logic

```python
# Pseudocode for resolution selection
target_resolution = "720p"
base_size = 960  # From resolution tier table
stride = 16
aspect_ratio = 16/9

# Calculate dimensions maintaining aspect ratio
num_patches = (base_size / stride) ** 2
height_patches = sqrt(num_patches / aspect_ratio)
width_patches = height_patches * aspect_ratio

height = round(height_patches) * stride
width = round(width_patches) * stride
# Result: 1280×720 for 16:9 @ 720p
```

---

## Quick Reference Chart

### Resolution → Typical Dimensions (16:9)

| Resolution | Width | Height | Megapixels | Use Case |
|------------|-------|--------|------------|----------|
| 360p       | 640   | 360    | 0.23 MP    | Preview/test |
| 480p       | 848   | 480    | 0.41 MP    | Standard quality |
| 720p       | 1280  | 720    | 0.92 MP    | HD quality |
| 1080p      | 1920  | 1080   | 2.07 MP    | Full HD |

### Model Type → Best Configuration

| Goal | Model | Resolution | Frames | Steps | VRAM |
|------|-------|------------|--------|-------|------|
| Fast iteration | Distilled | 480p | 73 | 25 | 6-8 GB |
| Balanced | Standard | 480p+SR | 121 | 50+6 | 10-14 GB |
| High quality | Standard | 720p+SR | 121 | 50+8 | 18-28 GB |
| Maximum quality | Standard | 720p | 145 | 50 | 24-32 GB |

---

## Additional Notes

### About Distilled Models

Distilled models use **knowledge distillation** to achieve similar quality with:
- 2x fewer steps (25 vs 50)
- Lower guidance scale (1.0 vs 6.0)
- Slightly faster inference (~40% time reduction)
- Minimal quality degradation

Best for: Iteration, testing, real-time applications

### About Super-Resolution

SR models are **distilled upsampling models** that:
- Upscale 480p→720p or 720p→1080p
- Use 6-8 steps (vs 50 for generation)
- Add ~10-15% total time to workflow
- Improve detail and reduce artifacts
- More VRAM efficient than native high-res generation

Best for: Production output, quality improvement without huge VRAM cost

### About Sparse Attention

Sparse attention (H100 only) enables:
- 2x-3x faster inference
- Same quality as standard models
- Requires `flex-block-attn` library
- Only works on H100 GPUs

Best for: H100 users wanting maximum speed

---

**Last Updated:** 2025-11-27
**Version:** 1.1
**Based on:** HunyuanVideo 1.5 official codebase
