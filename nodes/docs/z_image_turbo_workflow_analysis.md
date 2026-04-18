# Z-Image Turbo Workflow Analysis

## Overview

This document analyzes the official ComfyUI workflow for Z-Image Turbo, a text-to-image model that uses Qwen 3 (4B variant) as its text encoder.

**Source**: `./example_workflows/official_workflows/comfy_z_image_turbo_example_workflow.json`

---

## 1. Node Graph Structure

### Complete Node List

| Node ID | Type | Purpose |
|---------|------|---------|
| 16 | UNETLoader | Load DiT/UNET model |
| 18 | CLIPLoader | Load Qwen 3 4B text encoder |
| 17 | VAELoader | Load VAE decoder |
| 11 | ModelSamplingAuraFlow | Model sampling configuration (BYPASSED) |
| 6 | CLIPTextEncode | Positive prompt encoding |
| 7 | CLIPTextEncode | Negative prompt encoding |
| 13 | EmptySD3LatentImage | Create 16-channel latent canvas |
| 3 | KSampler | Diffusion sampling |
| 8 | VAEDecode | Latent to image decoding |
| 9 | SaveImage | Output image saving |
| 15 | Note | Workflow documentation |

### Data Flow Diagram

```
UNETLoader (z_image_turbo_bf16.safetensors)
    |
    v
ModelSamplingAuraFlow (mode=4, BYPASSED) -----> KSampler
                                                    ^
CLIPLoader (qwen_3_4b.safetensors, lumina2)         |
    |                                               |
    +---> CLIPTextEncode (Positive) ----------------+
    |                                               |
    +---> CLIPTextEncode (Negative) ----------------+
                                                    |
EmptySD3LatentImage (1024x1024) -------------------+
                                                    |
                                                    v
VAELoader (ae.safetensors) ----------------> VAEDecode
                                                    |
                                                    v
                                              SaveImage
```

### Connection Details (Links)

| Link ID | Source Node | Source Slot | Target Node | Target Slot | Data Type |
|---------|-------------|-------------|-------------|-------------|-----------|
| 42 | 16 (UNET) | 0 | 11 (AuraFlow) | 0 | MODEL |
| 47 | 11 (AuraFlow) | 0 | 3 (KSampler) | 0 | MODEL |
| 43 | 18 (CLIP) | 0 | 6 (Pos Encode) | 0 | CLIP |
| 44 | 18 (CLIP) | 0 | 7 (Neg Encode) | 0 | CLIP |
| 4 | 6 (Pos Encode) | 0 | 3 (KSampler) | 1 | CONDITIONING |
| 6 | 7 (Neg Encode) | 0 | 3 (KSampler) | 2 | CONDITIONING |
| 17 | 13 (Empty) | 0 | 3 (KSampler) | 3 | LATENT |
| 51 | 3 (KSampler) | 0 | 8 (VAEDecode) | 0 | LATENT |
| 45 | 17 (VAE) | 0 | 8 (VAEDecode) | 1 | VAE |
| 16 | 8 (VAEDecode) | 0 | 9 (SaveImage) | 0 | IMAGE |

---

## 2. Key Observations

### Model Components

#### Text Encoder: qwen_3_4b.safetensors
- **CLIP Type**: `lumina2`
- **Model**: Qwen 3 4B (NOT Qwen 2.5-VL 7B)
- **Configuration**: Default settings
- This is a newer, smaller Qwen variant compared to the Qwen 2.5-VL 7B used in Qwen-Image-Edit

#### Diffusion Model: z_image_turbo_bf16.safetensors
- **Loader**: UNETLoader
- **Weight Type**: Default (bf16 as indicated by filename)
- **Architecture**: DiT-based (indicated by SD3 latent compatibility)

#### VAE: ae.safetensors
- **Name**: Generic "ae.safetensors" (autoencoder)
- **Architecture**: Flux-derived AutoencoderKL (config shows `_name_or_path: "flux-dev"`)
- **Channel Count**: 16-channel
- **Scaling**: `scaling_factor=0.3611`, `shift_factor=0.1159` (Flux-specific)
- **Note**: Uses EmptySD3LatentImage node for 16-channel latent creation

### ModelSamplingAuraFlow Analysis

```json
{
  "type": "ModelSamplingAuraFlow",
  "mode": 4,
  "widgets_values": [3]
}
```

**Critical Finding**: The node is in `mode: 4`

In ComfyUI, node modes are:
- `0` = Active (normal operation)
- `2` = Muted (output passes through unchanged)
- `4` = Bypassed (node is skipped entirely)

**Why is it bypassed?**
- The Z-Image Turbo model likely has its own sampling configuration baked in
- AuraFlow sampling may have been tested but found unnecessary or detrimental
- The model already uses appropriate sigma scheduling internally
- The workflow author kept it visible for documentation/experimentation but disabled it

The `widgets_values: [3]` would set shift=3 if the node were active, but since it is bypassed, this has no effect.

### KSampler Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| seed | 47447417949230 | Randomized |
| steps | 9 | Very low - indicates "turbo" distilled model |
| cfg | 1 | No classifier-free guidance (turbo characteristic) |
| sampler_name | euler | Simple, fast sampler |
| scheduler | simple | Basic scheduling |
| denoise | 1 | Full denoising (text-to-image) |

**Turbo Model Characteristics**:
- 9 steps is extremely low compared to standard models (20-50 steps)
- CFG=1 means no guidance - the model was distilled to not need it
- This matches other "turbo" or "lightning" models that use consistency distillation

---

## 3. Prompt Structure Analysis

### Positive Prompt (from workflow)

```
cute anime style girl with massive fluffy fennec ears and a big fluffy tail
blonde messy long hair blue eyes wearing a maid outfit with a long black
gold leaf pattern dress and a white apron, it is a postcard held by a hand
in front of a beautiful realistic city at sunset and there is cursive
writing that says "ZImage, Now in ComfyUI"
```

### Prompt Template Discussion

The workflow note (Node 15) states:

> "The 'You are an assistant... <Prompt Start>' text before the actual prompt is the one used in the official example.
>
> The reason it is exposed to the user like this is because the model still works if you modify or remove it."

**Key Insights**:
1. The official Z-Image examples use a system-prompt-like prefix
2. This prefix is NOT required - the model works without it
3. Unlike Qwen-Image-Edit which strictly requires DiffSynth templates, Z-Image is more flexible
4. The workflow exposes raw prompts directly to CLIPTextEncode without special formatting

### Comparison: Z-Image vs Qwen-Image-Edit Prompt Handling

| Aspect | Z-Image Turbo | Qwen-Image-Edit |
|--------|---------------|-----------------|
| System Prompt | Optional, can be removed | Required for proper token dropping |
| Template Format | Simple prefix (if used) | `<|im_start|>system...` format |
| Token Dropping | Not required | Drops 34/64 tokens based on mode |
| Vision Tokens | N/A (text-only) | `<|vision_start|>...<|vision_end|>` |
| CLIP Type | lumina2 | QWEN_IMAGE |

### Negative Prompt

```
blurry ugly bad
```

Simple, minimal negative prompt - typical for turbo models that are less sensitive to negative guidance.

---

## 4. Model Relationships and Architecture

### EmptySD3LatentImage Implications

The use of `EmptySD3LatentImage` with 1024x1024 resolution indicates:

1. **16-Channel Latent Space**: SD3 uses 16 channels like Qwen-Image-Edit
2. **VAE Compatibility**: Z-Image uses a Flux-derived AutoencoderKL (16 latent channels, 8x spatial compression)
3. **Resolution**: 1024x1024 is the native resolution, matching SD3 family
4. **Latent Shape**: `[1, 16, 128, 128]` (1024/8 = 128 spatial dimensions)

### Architecture Comparison

| Component | Z-Image Turbo | Qwen-Image-Edit | SD3 |
|-----------|---------------|-----------------|-----|
| Text Encoder | Qwen 3 4B | Qwen 2.5-VL 7B | CLIP + T5 |
| CLIP Type | lumina2 | QWEN_IMAGE | SD3 |
| DiT Architecture | Yes (presumed) | Yes | Yes |
| VAE Channels | 16 | 16 | 16 |
| Native Resolution | 1024 | 512-2048 | 1024 |
| Distillation | Yes (turbo) | No | No |

### What is "lumina2" CLIP Type?

The `lumina2` CLIP type in ComfyUI refers to:
- A specific text encoder configuration for Lumina-based models
- Qwen models adapted for diffusion model conditioning
- Different tokenization and embedding handling compared to standard CLIP

**ComfyUI CLIP Types Context**:
- `CLIP` - Original OpenAI CLIP
- `SD3` - Stable Diffusion 3 dual encoder
- `FLUX` - Flux model encoder setup
- `lumina2` - Lumina/Z-Image text encoding
- `QWEN_IMAGE` - Qwen-Image-Edit specific

---

## 5. Integration Analysis

### Workflow Simplicity

Z-Image Turbo workflow is notably simpler than Qwen-Image-Edit:

**Z-Image Turbo**:
```
CLIP -> TextEncode -> KSampler -> VAEDecode
```

**Qwen-Image-Edit**:
```
CLIP -> QwenVLTextEncoder (with processor, templates, token dropping) -> KSampler -> VAEDecode
          ^
          |-- Template Builder
          |-- Image Batch
          |-- VAE (for reference latents)
          |-- Mask Processor (for inpainting)
```

### Why No Special Nodes?

Z-Image Turbo uses standard ComfyUI nodes because:
1. **Text-only**: No vision tokens or image conditioning
2. **No Templates**: Model doesn't require specific prompt formatting
3. **Distilled Model**: CFG and complex sampling not needed
4. **Standard Architecture**: SD3-compatible components

### Potential Integration with Qwen-Image-Edit Pipeline

If someone wanted to use Z-Image-style encoding with Qwen-Image-Edit:
- Would need different CLIP type (`lumina2` vs `QWEN_IMAGE`)
- Different model files (4B vs 7B Qwen)
- Incompatible embedding dimensions likely
- Token handling differs significantly

---

## 6. Summary and Recommendations

### Key Takeaways

1. **Z-Image Turbo is a distilled T2I model** using Qwen 3 4B, not Qwen 2.5-VL 7B
2. **9 steps with CFG=1** - characteristic of consistency-distilled models
3. **SD3-compatible latent space** - 16 channels, 1024x1024 native
4. **ModelSamplingAuraFlow is bypassed** - model handles sampling internally
5. **Simple prompt structure** - no mandatory templates or token manipulation
6. **lumina2 CLIP type** - specific to Lumina/Z-Image family

### How This Differs from Qwen-Image-Edit

| Feature | Z-Image Turbo | Qwen-Image-Edit |
|---------|---------------|-----------------|
| Primary Use | Text-to-Image (fast) | Image Editing |
| Steps | 9 | 20-50 typical |
| CFG | 1 | 5-8 typical |
| Image Input | No | Yes (vision encoder) |
| Special Nodes | None needed | QwenVLTextEncoder, etc. |
| Token Processing | Standard | DiffSynth template + dropping |
| Model Size | 4B text encoder | 7B VL encoder |

### File Locations Referenced

- **Workflow**: `./example_workflows/official_workflows/comfy_z_image_turbo_example_workflow.json`
- **Project CLIP Types**: `./nodes/qwen_vl_encoder.py` (line 111: `CLIPType.QWEN_IMAGE`)

---

## Appendix: Raw Node Configuration

### UNETLoader (Node 16)
```json
{
  "type": "UNETLoader",
  "widgets_values": ["z_image_turbo_bf16.safetensors", "default"]
}
```

### CLIPLoader (Node 18)
```json
{
  "type": "CLIPLoader",
  "widgets_values": ["qwen_3_4b.safetensors", "lumina2", "default"]
}
```

### VAELoader (Node 17)
```json
{
  "type": "VAELoader",
  "widgets_values": ["ae.safetensors"]
}
```

### EmptySD3LatentImage (Node 13)
```json
{
  "type": "EmptySD3LatentImage",
  "widgets_values": [1024, 1024, 1]
}
```

### KSampler (Node 3)
```json
{
  "type": "KSampler",
  "widgets_values": [47447417949230, "randomize", 9, 1, "euler", "simple", 1]
}
```
