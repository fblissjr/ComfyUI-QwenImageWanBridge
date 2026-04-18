# Wan2.1-VAE-upscale2x Integration Guide

Comprehensive guide for using the Wan2.1-VAE-upscale2x with Qwen-Image-Edit models for maximum quality and performance.

## Overview

The Wan2.1-VAE-upscale2x is a modified Wan VAE that provides 2x upscaling during decode, eliminating quality issues while maintaining the same latent space compatibility with Qwen models.

### Key Features

- **2x upscaling in decoder** - Encode at 1792×1792, decode at 3584×3584
- **Same latent space** - 16-channel Wan format, fully compatible with Qwen-Image-Edit
- **Quality improvements** - Eliminates "wan speckles/polka dots/grain" artifacts
- **Latent degradation training** - Specifically trained to handle diffusion model outputs
- **No VRAM increase during sampling** - Latent size unchanged, upscaling happens at decode

### Technical Specifications

```json
{
  "z_dim": 16,
  "scale_factor_spatial": 8,
  "out_channels": 12,
  "in_channels": 3,
  "base_dim": 96
}
```

**Decoder upscaling:**
- 12-channel output (vs standard 3-channel)
- Pixel shuffle converts 12 channels to 3 channels at 2x resolution
- Native 2x upscaling without external upscaler

## Integration with Qwen-Image-Edit

### Understanding the Two Paths

**IMPORTANT: There are TWO separate image processing paths:**

**Path 1: Vision Encoder (VL Model - Semantic Understanding)**
- Model: Qwen2.5-VL vision encoder
- Resolution: Hardcoded 384×384 area-based scaling
- Purpose: Creates vision tokens for semantic/content understanding
- Output: Vision embeddings (what the image shows/means)
- Not affected by `vae_max_dimension` parameter

**Path 2: VAE Encoder (Generation Model - Pixel Detail)**
- Model: Wan VAE encoder
- Resolution: Controlled by `vae_max_dimension` parameter
- Purpose: Creates reference latents for the diffusion/DiT generation model
- Output: 16-channel latents (pixel-level detail for generation)
- This is what determines your output resolution

**Think of it like this:**
- Vision encoder reads the image semantically ("what is this?")
- VAE encoder provides pixel reference for generation ("copy this detail")
- Both feed into the generation model, different purposes

**Example with 2048×2048 input image:**
```
Input Image (2048×2048)
    │
    ├─→ Vision Encoder Path (VL Model)
    │   └─→ Scaled to 384×384 area (e.g., 384×384)
    │       └─→ Vision tokens (semantic: "a cat on a chair")
    │
    └─→ VAE Encoder Path (Generation Model)
        └─→ Scaled via vae_max_dimension=1792
            └─→ Encoded at 1792×1792
                └─→ Latent 224×224×16 channels
                    └─→ Generation happens here
                        └─→ Decoded to 3584×3584 (2x upscale)
```

Both paths process the SAME input image at DIFFERENT resolutions for DIFFERENT purposes.

### What Changes with Wan2.1-upscale2x

**VAE encoder/decoder (Path 2 only):**
- Encode: Input → 8x downscale → 16-channel latent
- Decode: 16-channel latent → 8x upscale → 2x pixel shuffle → 2x larger output
- Effective output: 16x larger than latent dimensions

**Resolution multiplier:**
- Standard VAE: 8x upscale (encode 1024px → output 1024px)
- Wan2.1-upscale2x: 16x upscale (encode 1024px → output 2048px)

**`vae_max_dimension` parameter:**
- Controls VAE encoding resolution (generation reference)
- Does NOT affect vision encoder (always 384×384)
- Example: vae_max_dimension=1792 → VAE encodes at 1792px → outputs at 3584px
- Vision encoder still processes same image at 384×384 for vision tokens

### What Stays the Same

**Vision encoder (Path 1 - UNCHANGED):**
- Still processes at 384×384 area-based scaling
- Position embeddings trained at this resolution
- Higher resolutions cause artifacts (verified in v2.7.2 testing)
- Vision path provides semantic understanding, not pixel detail
- Not configurable via any parameter

**Latent space:**
- Same 16-channel format
- Same spatial resolution (encode_size / 8)
- Same compatibility with Qwen DiT models
- Same VRAM usage during sampling

**Token processing:**
- Vision tokens: Unchanged
- Token dropping: Unchanged (34 for T2I, 64 for edit/inpaint)
- Template system: Unchanged

## Optimal Resolution Settings

### Recommended Defaults

**Current implementation default: `vae_max_dimension = 2048`**
- Encodes at 2048×2048 (4.2MP)
- Outputs at 4096×4096 (16.8MP) with 2x VAE
- Latent: 256×256×16 channels

**Recommended for Wan2.1-upscale2x: `vae_max_dimension = 1792`**
- Encodes at 1792×1792 (3.2MP)
- Outputs at 3584×3584 (12.8MP) - model's maximum resolution
- Latent: 224×224×16 channels
- Optimal for 12GB+ VRAM

### Resolution Tiers

| Encode Size | Latent Size | Output Size (2x VAE) | VRAM Tier | Use Case |
|-------------|-------------|----------------------|-----------|----------|
| 512×512 | 64×64×16 | 1024×1024 | 6GB | Minimum viable |
| 768×768 | 96×96×16 | 1536×1536 | 8GB | Low VRAM safe |
| 1024×1024 | 128×128×16 | 2048×2048 | 8GB | Baseline quality |
| 1280×1280 | 160×160×16 | 2560×2560 | 10GB | High quality |
| 1536×1536 | 192×192×16 | 3072×3072 | 12GB | Very high quality |
| **1792×1792** | **224×224×16** | **3584×3584** | **12GB+** | **Recommended max** |
| 2048×2048 | 256×256×16 | 4096×4096 | 16GB+ | Experimental |
| 2560×2560 | 320×320×16 | 5120×5120 | 24GB+ | May hit token limits |

### Why 1792×1792 is Optimal

**Hits model maximum:**
- 3584×3584 output = model's trained maximum resolution
- 16,384 token limit: 224×224 latent = 50,176 positions → 6,272 tokens (8×8 patches)
- No wasted capacity

**VRAM efficiency:**
- Latent: 224×224×16 = ~25MB per image (float16)
- Multi-image (3): ~75MB for reference latents
- Manageable on 12GB GPUs

**Quality sweet spot:**
- 3.2MP input preserves detail without excessive upscaling
- 12.8MP output matches model capabilities
- 2x decoder upscaling is high quality (better than external upscalers)

**32px alignment friendly:**
- 1792 = 56 × 32 (perfect VAE alignment)
- 224 = 28 × 8 (latent dimension)
- No dimension adjustment needed

## Node Parameter Recommendations

### QwenVLTextEncoder

**What does `vae_max_dimension` control?**
- Controls VAE encoding resolution (for generation/edit model)
- Does NOT control vision encoder (always 384×384)
- Determines final output resolution via 2x upscaling
- Higher = more pixel detail for generation, more VRAM

**Recommended settings with Wan2.1-upscale2x:**
```python
vae_max_dimension: 1792  # VAE encodes at 1792px → outputs 3584px
                         # Vision encoder still at 384×384 (semantic understanding)
```

**VRAM-constrained settings:**
```python
# 8GB VRAM
vae_max_dimension: 1024  # VAE: 1024px encode → 2048px output
                         # Vision: 384×384 (unchanged)

# 10GB VRAM
vae_max_dimension: 1280  # VAE: 1280px encode → 2560px output
                         # Vision: 384×384 (unchanged)

# 24GB+ VRAM (experimental)
vae_max_dimension: 2048  # VAE: 2048px encode → 4096px output (may hit limits)
                         # Vision: 384×384 (unchanged)
```

### QwenImageBatch

**Recommended settings for multi-image:**
```python
vae_max_dimension: 1536  # Slightly lower for multi-image safety
batch_alignment: "match_smallest"  # Safer VRAM usage
```

**Why lower for multi-image:**
- 3 images at 1792px = 75MB latents + 3× generation cost
- 3 images at 1536px = 54MB latents (28% less VRAM)
- Still outputs at 3072×3072 per image (9.4MP vs 12.8MP)

### QwenVLTextEncoderAdvanced

**Recommended hero/reference weights:**
```python
vae_max_dimension: 1536  # Base dimension
resolution_mode: "hero_first"
hero_weight: 1.17  # 1536 × 1.17 ≈ 1792 (hero at max)
reference_weight: 0.67  # 1536 × 0.67 ≈ 1024 (refs at baseline)
```

**Result:**
- Hero image: 1792×1792 encode → 3584×3584 output
- References: 1024×1024 encode → 2048×2048 output
- Balanced VRAM usage with quality prioritization

## Performance Characteristics

### VRAM Usage Breakdown

**Single image at 1792×1792:**
- Vision processing: 384×384×3 = 0.4MB
- VAE encode: 1792×1792×3 = 9.2MB
- Reference latent: 224×224×16 = 25MB
- Sampling latent: 224×224×16 = 25MB
- Model weights: 4-8GB (depends on quantization)
- **Total: ~60MB + model weights**

**Multi-image (3) at 1792×1792:**
- Vision processing: 3 × 0.4MB = 1.2MB
- VAE encode: 3 × 9.2MB = 27.6MB
- Reference latents: 3 × 25MB = 75MB
- Sampling latent: 224×224×16 = 25MB (single output)
- **Total: ~130MB + model weights**

### Speed Considerations

**Encoding (one-time per image):**
- 1024×1024: ~0.5s on RTX 3090
- 1792×1792: ~1.2s on RTX 3090
- 2048×2048: ~1.8s on RTX 3090

**Decoding (one-time at end):**
- 2x VAE adds ~10-15% decode time vs standard VAE
- Pixel shuffle is fast (GPU-accelerated)
- Still faster than separate upscaler pass

**Sampling (main bottleneck):**
- Latent size drives sampling cost
- 224×224 latent: ~15-30s for 20 steps (fp8 Qwen)
- Independent of 2x upscaling (happens after sampling)

## Workflow Integration

### Text-to-Image

```
QwenVLCLIPLoader → QwenTemplateBuilder → QwenVLTextEncoder
                                              ↓ (vae_max_dimension: 1792)
QwenVLEmptyLatent (1792×1792) → KSampler → VAEDecode → SaveImage
                                              ↓         (3584×3584 output)
                                         Wan2.1-VAE
```

**Settings:**
- Empty latent: 1792×1792 (will output 3584×3584)
- Steps: 20-30
- CFG: 7.0-9.0

### Single Image Edit

```
LoadImage (any size) → QwenVLTextEncoder (vae_max_dimension: 1792)
                              ↓
QwenVLEmptyLatent → KSampler (denoise: 0.5-0.7) → VAEDecode → SaveImage
                                                    (2x output)
```

**Automatic scaling:**
- Input: 2560×1440 → Encoder scales to 1792×1008 → Output: 3584×2016
- Input: 1024×768 → Encoder scales to 1344×1024 → Output: 2688×2048
- Always preserves aspect ratio with 32px alignment

### Multi-Image Edit

```
LoadImage ─┐
LoadImage ─┼─> QwenImageBatch (vae_max_dimension: 1536) → QwenVLTextEncoder
LoadImage ─┘         ↓                                            ↓
                batch_alignment                               (vision: 384×384)
                                                                   ↓
                                        QwenVLEmptyLatent → KSampler → VAEDecode
                                                                       (3072×3072 each)
```

**Multi-image sizing:**
- Lower vae_max_dimension (1536 vs 1792) for VRAM safety
- Still get 3072×3072 outputs per image
- Vision encoder unchanged at 384×384 area

## Quality Optimization Tips

### Maximizing Output Quality

1. **Use 1792px encode for single images**
   - Hits model's 3584px maximum
   - No wasted capacity
   - Best quality/VRAM ratio

2. **Adjust for multi-image**
   - 1536px for 2-3 images (3072px output)
   - 1280px for 4+ images (2560px output)
   - Prevents VRAM issues

3. **Vision encoder stays at 384×384**
   - Don't try to increase it
   - Model's position embeddings trained here
   - Higher resolutions cause artifacts

4. **Use hero weighting for mixed importance**
   - Advanced encoder: hero_weight=1.17 on main subject
   - References at lower resolution save VRAM
   - Output quality still good on hero

### Known Limitations

**Wan2.1-upscale2x trained on real images only:**
- May struggle with anime/manga styles
- Lineart and text rendering may be inconsistent
- Color shifts possible (usually addressable)
- Over-sharpness in some cases (can blur in post)

**Test with your content type:**
- Real photos: Excellent results expected
- Digital art: Good results expected
- Anime/stylized: May need testing and adjustments
- Lineart: May need fallback to standard VAE

### Color Correction

If you experience color shifts (noted in VAE docs):

1. **Use color correction node after decode**
   - Slight adjustments usually sufficient
   - HSV or RGB curves work well

2. **Adjust in prompt**
   - "Preserve original colors"
   - "Maintain color accuracy"

3. **Reference latent strength**
   - Edit mode: Try denoise 0.4-0.6 (vs 0.5-0.7)
   - Stronger reference preserves colors

## Migration from Standard VAE

### No Code Changes Required

**Current implementation works as-is:**
- Same 16-channel latent format
- Same model forward pass
- Same conditioning system
- Just swap VAE file in ComfyUI

### Workflow Adjustments

**Optional optimization:**
1. Lower `vae_max_dimension` from 2048 to 1792
2. Get same effective output (3584px vs 4096px)
3. Save ~25% VRAM

**If keeping 2048:**
- Outputs 4096×4096 (may exceed model optimal range)
- Uses more VRAM than necessary
- May hit token limits on some workflows
- Still works, just not optimal

### A/B Testing Recommendations

**Compare at these settings:**

**Test 1: Standard Wan VAE vs Wan2.1-upscale2x (same encode size)**
- Both at 2048px encode
- Standard outputs 2048px
- Upscale2x outputs 4096px
- Check: Quality, speckles, artifacts

**Test 2: Optimal settings (same output size)**
- Standard: 3584px encode → 3584px output
- Upscale2x: 1792px encode → 3584px output
- Check: Quality, VRAM, speed, artifacts

**Test 3: Multi-image stress test**
- 3 images, standard: 1792px each
- 3 images, upscale2x: 1536px encode (3072px output)
- Check: VRAM usage, quality, consistency

## Troubleshooting

### CUDA Out of Memory

**Symptoms:**
- "CUDA out of memory" during encoding or sampling
- System freezes during generation

**Solutions:**
1. Lower `vae_max_dimension`: 1792 → 1536 → 1280 → 1024
2. Use `batch_alignment: "match_smallest"` for multi-image
3. Reduce number of simultaneous images
4. Use advanced encoder with lower reference_weight

### Output Quality Issues

**Anime/stylized content looks wrong:**
- VAE trained on real images only
- May need to use standard Wan VAE for anime
- Test both VAEs, keep whichever works better

**Over-sharpening:**
- Noted limitation of this VAE
- Add blur node after decode (1-2px Gaussian)
- Adjust in image editor if needed

**Color shifts:**
- Use color correction node
- Lower denoise strength (0.4-0.5 instead of 0.7)
- Adjust prompt ("preserve colors", "accurate color reproduction")

### Dimension Mismatches

**"Reference latent size mismatch" errors:**
- Should be auto-handled by model wrapper
- If error persists, check batch_alignment setting
- Ensure all images go through same encoder

**Aspect ratio distortion in multi-image:**
- Use `batch_alignment: "match_smallest"` or "match_first"
- Pre-scale images to same aspect ratio before batching
- Check resolution_tradeoffs.md for detailed strategies

## Technical Deep Dive

### Decoder Architecture

**Standard Wan VAE:**
```
Latent [B, 16, H/8, W/8] → Decoder → Image [B, 3, H, W]
```

**Wan2.1-upscale2x:**
```
Latent [B, 16, H/8, W/8] → Decoder → [B, 12, H, W] → PixelShuffle → [B, 3, H*2, W*2]
```

**PixelShuffle explanation:**
- 12 channels rearranged into 3 channels at 2x spatial resolution
- Each 2×2 block from 12 channels becomes 1 pixel in 3 channels
- No interpolation, just reorganization
- Efficient and high quality

### Latent Degradation Training

**Problem with standard VAE training:**
- Trained on encode(image) → decode(latent)
- Real diffusion produces degraded latents (noise residual, artifacts)
- VAE decoder not optimized for degraded inputs

**Wan2.1-upscale2x solution:**
- Proxy convolution network simulates diffusion degradation
- Training includes degraded latents, not just clean encodes
- Decoder learns to handle real generation artifacts
- Result: Better quality on actual generations (not just reconstructions)

### Compatibility Details

**Works with any 16-channel Wan latent model:**
- Qwen-Image-Edit (all versions)
- Qwen-Image-Edit-2509
- Qwen-Image-Edit-Nunchaku (quantized)
- Future Qwen models using Wan latents

**Does not work with:**
- Standard 4-channel SDXL/Flux VAEs
- Video models (Wan2.1 uses temporal_downsample)
- Different latent dimension models

### Training Configuration Reference

From model card:
```
Batch size: 4
Total steps: 300,000
Training time: ~40 hours on RTX 5090
Losses: L1 + LPIPS + Frequency Distribution + patchGAN
```

**Why this matters:**
- Extensive training ensures quality
- Frequency Distribution Loss preserves high-frequency details
- patchGAN prevents artifacts better than standard adversarial
- Real images only (limitation but also strength for photo editing)

## Recommended Tooltip Updates

### For Node Developers

If implementing `vae_max_dimension` parameter, recommended tooltip:

```python
"vae_max_dimension": ("INT", {
    "default": 1792,  # Updated for Wan2.1-upscale2x
    "min": 512,
    "max": 3584,
    "step": 64,
    "tooltip": (
        "VAE encoder max dimension.\n\n"
        "With Wan2.1-VAE-upscale2x (2x decoder):\n"
        "  • 1024 - Safe for 8GB VRAM → 2048px output\n"
        "  • 1536 - High quality (12GB VRAM) → 3072px output\n"
        "  • 1792 - Recommended max (12GB+ VRAM) → 3584px output\n"
        "  • 2048 - Experimental (16GB+ VRAM) → 4096px output\n\n"
        "With standard Wan VAE (1x):\n"
        "  • Use values directly (no 2x multiplication)\n\n"
        "Note: Vision encoder remains 384×384 area (model limitation).\n"
        "Higher VAE resolutions improve detail without affecting vision."
    )
})
```

## Summary

### Key Takeaways

1. **Wan2.1-upscale2x provides 2x upscaling for free** - Same VRAM during sampling
2. **Lower encode size, same quality** - 1792px encode = 3584px output
3. **Vision encoder unchanged** - Still 384×384 area (model limitation)
4. **Optimal default: 1792px** - Hits model maximum efficiently
5. **Multi-image: Use 1536px** - VRAM safety with great quality
6. **Real photos work best** - VAE trained on real images only

### Quick Start

1. Install Wan2.1-VAE-upscale2x via ComfyUI-VAE-Utils
2. Keep existing workflows (no code changes needed)
3. Optionally adjust `vae_max_dimension` to 1792 for optimization
4. Test with your content type (anime may need standard VAE)
5. Enjoy 2x resolution outputs with better quality

### Further Reading

- [Resolution Tradeoffs Guide](resolution_tradeoffs.md)
- [QwenImageBatch Documentation](QwenImageBatch.md)
- [Advanced Encoder Guide](QwenVLTextEncoderAdvanced.md)
- [Wan2.1-VAE-upscale2x Model Card](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x)
