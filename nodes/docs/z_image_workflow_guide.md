# Z-Image Workflow Guide

Complete guide for text-to-image generation using Z-Image Turbo in ComfyUI.

---

## Overview

Z-Image is Alibaba's 6B parameter text-to-image model using:
- **Text Encoder**: Qwen3-4B (2560 dim embeddings)
- **Architecture**: S3-DiT (Single-Stream DiT)
- **Training**: Decoupled DMD distillation (CFG baked in)
- **Inference**: 8-9 steps, CFG=1, euler sampler

### Why Our Nodes?

After analysis, we found ComfyUI and diffusers produce **identical templates** by default. Our nodes provide:
- System prompt presets for experimentation
- Optional `add_think_block` parameter for testing
- Debug mode for troubleshooting

---

## Prerequisites

### Required Models

**Text Encoder (in `models/text_encoders/`):**
- `qwen_3_4b.safetensors` - Qwen3-4B weights

**DiT Model (in `models/diffusion_models/`):**
- `z_image_turbo_bf16.safetensors` - Z-Image Turbo model

**VAE (in `models/vae/`):**
- `ae.safetensors` - Z-Image autoencoder (16-channel)

### Model Downloads

- **Z-Image Turbo**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- **Qwen3-4B**: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

### Node Categories

Our custom nodes appear in:
- `ZImage/Encoding` - Text encoders

ComfyUI native nodes you'll also use:
- `loaders` - CLIPLoader, UNETLoader, VAELoader
- `sampling` - KSampler
- `latent` - EmptySD3LatentImage, VAEDecode

---

## Quick Start

### Minimal Change from Official Workflow

The official workflow uses:
```
CLIPLoader (lumina2) -> CLIPTextEncode -> KSampler
```

Our nodes:
```
CLIPLoader (lumina2) -> ZImageTextEncoderSimple -> KSampler
```

Default behavior matches diffusers exactly. Use `add_think_block=True` to experiment.

---

## Text-to-Image Workflow

### Node Chain

```
┌──────────────────────────┐
│ CLIPLoader               │ Load Qwen3-4B text encoder
│ (loaders)                │ type: lumina2
└────────────┬─────────────┘
             │ CLIP
             v
┌──────────────────────────┐
│ ZImageTextEncoderSimple  │ Encode prompt
│ (ZImage/Encoding)        │ add_think_block: False (default)
└────────────┬─────────────┘
             │ CONDITIONING
             v
┌──────────────────────────┐
│ KSampler                 │ Generate image
│ (sampling)               │ steps: 9, cfg: 1
└────────────┬─────────────┘
             │ LATENT
             v
┌──────────────────────────┐
│ VAEDecode                │ Decode to pixels
│ (latent)                 │
└────────────┬─────────────┘
             │ IMAGE
             v
┌──────────────────────────┐
│ SaveImage                │ Save output
│ (image)                  │
└──────────────────────────┘
```

### Step-by-Step Setup

#### 1. CLIPLoader (ComfyUI Native)
**Category:** `loaders`

**Settings:**
- `clip_name`: `qwen_3_4b.safetensors`
- `type`: `lumina2` (important!)
- `weight_dtype`: `default`

**Output:**
- `CLIP` -> Connect to ZImageTextEncoderSimple

#### 2. ZImageTextEncoderSimple (Our Node)
**Category:** `ZImage/Encoding`

**Settings:**
- `text`: Your prompt
- `add_think_block`: `False` (default, matches diffusers)

**Example prompt:**
```
A serene Japanese garden with a koi pond, cherry blossoms falling gently,
soft morning light filtering through maple trees, highly detailed,
photorealistic
```

**Output:**
- `CONDITIONING` -> Connect to KSampler positive

#### 3. Negative Prompt (Optional)
For negative conditioning, add another ZImageTextEncoderSimple:

**Settings:**
- `text`: What to avoid
- `add_think_block`: `False`

**Example:**
```
blurry, ugly, bad quality, watermark, text, logo
```

**Output:**
- `CONDITIONING` -> Connect to KSampler negative

#### 4. EmptySD3LatentImage (ComfyUI Native)
**Category:** `latent`

**Settings:**
- `width`: 1024
- `height`: 1024
- `batch_size`: 1

**Output:**
- `LATENT` -> Connect to KSampler latent_image

#### 5. UNETLoader (ComfyUI Native)
**Category:** `loaders`

**Settings:**
- `unet_name`: `z_image_turbo_bf16.safetensors`
- `weight_dtype`: `default`

**Output:**
- `MODEL` -> Connect to KSampler model

#### 6. KSampler (ComfyUI Native)
**Category:** `sampling`

**Settings:**
- `seed`: Any integer (or "randomize")
- `steps`: `9` (turbo model)
- `cfg`: `1` (CFG baked into model)
- `sampler_name`: `euler`
- `scheduler`: `simple`
- `denoise`: `1.0`

**Inputs:**
- `model`: From UNETLoader
- `positive`: From ZImageTextEncoderSimple (prompt)
- `negative`: From ZImageTextEncoderSimple (negative) or leave empty
- `latent_image`: From EmptySD3LatentImage

**Output:**
- `LATENT` -> Connect to VAEDecode

#### 7. VAELoader (ComfyUI Native)
**Category:** `loaders`

**Settings:**
- `vae_name`: `ae.safetensors`

**Output:**
- `VAE` -> Connect to VAEDecode vae

#### 8. VAEDecode (ComfyUI Native)
**Category:** `latent`

**Inputs:**
- `samples`: From KSampler
- `vae`: From VAELoader

**Output:**
- `IMAGE` -> Connect to SaveImage

---

## Advanced Workflow: With System Prompts

Use `ZImageTextEncoder` (full version) for system prompt control.

### Node Chain

```
CLIPLoader (lumina2)
        |
        | CLIP
        v
ZImageTextEncoder
  - text: "your prompt"
  - system_prompt_preset: "photorealistic"
  - add_think_block: False (or True for experiments)
  - debug_mode: True
        |
        | CONDITIONING
        v
KSampler (steps: 9, cfg: 1)
        |
        | LATENT
        v
VAEDecode
```

### System Prompt Options

| Preset | Best For |
|--------|----------|
| `none` | Default behavior, matches diffusers |
| `quality` | General high-quality output |
| `photorealistic` | Photography, realistic scenes |
| `artistic` | Creative, stylized images |
| `bilingual` | Images with English/Chinese text |

### Custom System Prompt

Override presets with your own:

```
custom_system_prompt: "Generate a detailed architectural visualization
with accurate perspective, professional lighting, and photorealistic
materials. Focus on clean lines and modern design."
```

---

## Settings Reference

### Z-Image Turbo Sampling

| Parameter | Value | Notes |
|-----------|-------|-------|
| steps | 9 | Very low - turbo distilled model |
| cfg | 1 | No guidance - CFG baked in via DMD |
| sampler | euler | Simple, fast |
| scheduler | simple | Basic scheduling |
| denoise | 1.0 | Full denoising for T2I |

### Resolution

**Native resolution:** 1024x1024

**Supported ratios:**
- 1:1 (1024x1024) - default
- 4:3 (1024x768)
- 3:4 (768x1024)
- 16:9 (1024x576)
- 9:16 (576x1024)

### VRAM Estimates

- 1024x1024 @ 9 steps: ~8-10GB
- Higher resolutions may require more

---

## Prompting Tips

### What Works Well

**Structure your prompts:**
```
[Subject], [Setting], [Lighting], [Style], [Quality modifiers]
```

**Example:**
```
A majestic lion resting on a savanna rock, golden hour sunset,
dramatic rim lighting, wildlife photography, highly detailed, 8k
```

### Include Details

- **Lighting**: "soft diffused light", "dramatic shadows", "golden hour"
- **Style**: "photorealistic", "cinematic", "editorial", "artistic"
- **Quality**: "highly detailed", "sharp focus", "professional"
- **Mood**: "serene", "dramatic", "mysterious", "vibrant"

### Text Rendering

Z-Image can render text. Put text in quotes:
```
A neon sign that says "OPEN 24 HOURS" in a rainy alley at night
```

### Negative Prompts

Keep simple for turbo model:
```
blurry, ugly, bad quality, watermark, text, logo, distorted
```

---

## Troubleshooting

### "Model not found" errors

**Solution:** Verify model paths:
```
ComfyUI/models/
  text_encoders/
    qwen_3_4b.safetensors
  diffusion_models/
    z_image_turbo_bf16.safetensors
  vae/
    ae.safetensors
```

### "Wrong CLIP type" or dimension errors

**Solution:** Ensure CLIPLoader uses `type: lumina2`

### Poor image quality

**Solutions:**
1. Use our encoder (default settings match diffusers)
2. Verify you're using our node, not CLIPTextEncode
3. Check sampling settings (steps=9, cfg=1 for turbo)

### Different results than diffusers

**Known differences:**
- Embedding extraction (we include padding, diffusers filters)
- Tokenizer bundling (minor differences)
- Sampling implementation (ComfyUI vs diffusers)

### Out of Memory (OOM)

**Solutions:**
1. Lower resolution (768x768 instead of 1024x1024)
2. Enable CPU offloading in ComfyUI settings
3. Close other applications

---

## Comparing Results

### A/B Test: With vs Without Thinking Tokens

**Test A (ComfyUI default):**
```
CLIPLoader -> CLIPTextEncode -> KSampler
```

**Test B (Our fix):**
```
CLIPLoader -> ZImageTextEncoderSimple -> KSampler
```

Use same prompt, same seed, compare outputs.

### What to Look For

- Overall coherence and quality
- Prompt adherence
- Fine details and textures
- Artifact reduction

---

## Example Prompts

### Portrait

```
Professional headshot of a young woman with natural makeup,
soft studio lighting, neutral gray background, sharp focus,
editorial photography style
```

### Landscape

```
Dramatic mountain landscape at sunrise, mist in the valley,
snow-capped peaks catching golden light, wide angle view,
landscape photography, highly detailed
```

### Product

```
Luxury watch on a dark marble surface, dramatic side lighting,
product photography, reflection visible, sharp focus, 8k quality
```

### Fantasy

```
Ancient wizard in a mystical library, floating books and glowing runes,
volumetric lighting through dusty windows, fantasy art style,
highly detailed robes and magical effects
```

### Architectural

```
Modern minimalist house with floor-to-ceiling windows,
overlooking a cliff at sunset, interior visible,
architectural visualization, photorealistic rendering
```

---

## Quick Reference

### Workflow Summary

1. Load CLIP with `lumina2` type
2. Use `ZImageTextEncoderSimple` instead of `CLIPTextEncode`
3. Use default `add_think_block=False` (matches diffusers)
4. Sample with `steps=9`, `cfg=1`, `euler` sampler
5. Decode with Z-Image VAE

### Key Settings

| Component | Setting | Value |
|-----------|---------|-------|
| CLIPLoader | type | lumina2 |
| Encoder | add_think_block | False |
| KSampler | steps | 9 |
| KSampler | cfg | 1 |
| KSampler | sampler | euler |
| Latent | size | 1024x1024 |

### File Locations

- **Our nodes**: `nodes/z_image_encoder.py`
- **Templates**: `nodes/templates/z_image_*.md`
- **Example workflow**: `example_workflows/official_workflows/comfy_z_image_turbo_example_workflow.json`

---

## Experiments to Try

These experiments help determine if our fixes actually improve output quality.

### Experiment 1: Think Block A/B Test

**Goal:** Does adding `<think>` block improve or change quality?

**Note:** After analysis, we found ComfyUI and diffusers produce identical templates (no think block). This experiment tests if ADDING a think block helps.

**Setup:**
1. Same prompt, same seed, same settings
2. Run A: `ZImageTextEncoderSimple` with `add_think_block=False` (matches diffusers)
3. Run B: `ZImageTextEncoderSimple` with `add_think_block=True` (experimental)
4. Compare outputs

**What to look for:**
- Overall coherence and detail
- Prompt adherence
- Any quality differences

### Experiment 2: Custom Thinking Content

**Goal:** Does providing reasoning inside `<think>` tags affect output?

**Setup:**
1. Use `ZImageTextEncoder` with `add_think_block=True`
2. Run A: Leave `thinking_content` empty
3. Run B: Provide structured reasoning:

```
thinking_content: "Key elements: [list main subjects]
Composition: [describe layout, focal points]
Lighting: [specify lighting style]
Style: [artistic direction]"
```

4. Compare outputs

**Example thinking content patterns:**

**Analytical:**
```
Subject is a mountain landscape at sunset.
Primary colors: orange, purple, deep blue.
Focal point: snow-capped peak catching last light.
Depth: foreground rocks, midground trees, background peaks.
Mood: serene, majestic, contemplative.
```

**Technical:**
```
Camera: wide angle lens, f/11 for deep DOF.
Lighting: golden hour, sun 15 degrees above horizon.
Exposure: HDR blend for highlight/shadow detail.
Post: warm color grade, slight vignette.
```

**Compositional:**
```
Rule of thirds: horizon on lower third.
Leading lines: river guides eye to mountain.
Framing: trees on left create natural frame.
Balance: large mountain balanced by cloud mass.
```

### Experiment 3: System Prompts

**Goal:** Do system prompts affect embedding quality?

**Setup:**
1. Same prompt across all tests
2. Test each `system_prompt_preset`:
   - `none` (diffusers default)
   - `quality`
   - `photorealistic`
   - `artistic`
   - `bilingual`

**Document:**
- Which preset produces best results for your use case
- Any noticeable style differences

### Experiment 4: Sequence Length

**Goal:** Does prompt length affect quality?

**Setup:**
1. Short prompt (~50 chars)
2. Medium prompt (~200 chars)
3. Long prompt (~500 chars)
4. Very long prompt (~1000 chars, near limit)

**Document:**
- Quality vs length tradeoff
- Point of diminishing returns

### Experiment 5: Negative Prompts

**Goal:** How much do negative prompts help with turbo model?

**Setup:**
1. Run A: No negative prompt
2. Run B: Simple negative: `"blurry, ugly, bad quality"`
3. Run C: Detailed negative: `"blurry, ugly, bad quality, watermark, text, logo, distorted, artifacts, low resolution, pixelated"`

**Note:** Z-Image Turbo uses CFG=1 (baked in), so negative prompts may have less effect than standard models.

### Recording Results

For each experiment, record:
- **Prompt used**
- **Seed** (for reproducibility)
- **Settings** (steps, cfg, sampler)
- **Variation** (A/B/C)
- **Observations** (quality, adherence, artifacts)
- **Winner** (which variation was best)

### Share Your Findings

If you discover something interesting:
1. Document your methodology
2. Include example images if possible
3. Note your hardware (may affect results)
4. Share in GitHub issues or discussions

---

**Last Updated:** 2025-11-27
**Version:** 1.1
