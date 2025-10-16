# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code and Writing Style Guidelines

- **No emojis** in code, display names, or documentation
- Do **NOT** commit code to git or stage code for git without me explicitly asking and approving it
- Keep all naming and display text professional
- Avoid "Pure", "Enhanced", "Advanced", "Ultimate" type prefixes - use descriptive names instead
- Always avoid redundancy and unnecessary complexity. If you need to make a v2, there needs to be a compelling reason for it instead of simply modifying the code.
- Clean, simple node names that describe what they do
- Keep descriptions minimal and factual

## Organization

- `nodes/` - Production-ready nodes only
- `nodes/archive/` - Legacy and experimental nodes
- `example_workflows/` - Example JSON workflows with comprehensive notes
- `internal/` - Internal documentation and analysis

## Project Overview

ComfyUI nodes for Qwen-Image-Edit model, enabling text-to-image generation and vision-based image editing using Qwen2.5-VL 7B. Bridges DiffSynth-Studio patterns with ComfyUI's node system.

**Key Features:**
- 16-channel VAE latents (vs standard 4-channel)
- Vision token processing with multi-image support
- Template system with token dropping (DiffSynth-compatible)
- Mask-based inpainting with diffusers blending pattern
- Native ComfyUI integration via `CLIPType.QWEN_IMAGE`

**Models:**
- Text/Vision Encoder: `Qwen/Qwen2.5-VL-7B-Instruct` → `models/text_encoders/`
- DiT Model: `qwen-image-edit-2509` (fp8 or Nunchaku quantized) → `models/diffusion_models/`
- VAE: `qwen_image_vae.safetensors` (16-channel) → `models/vae/`

## Implementation Status

### Production Ready
- Text-to-image generation (QwenVLTextEncoder)
- Single/multi-image editing with vision tokens
- Mask-based inpainting (QwenMaskProcessor + QwenInpaintSampler)
- Template system (15+ presets)
- 16-channel VAE support
- Multi-image "Picture X:" labeling (auto, 1-3 images optimal)
- Token dropping (34 for T2I, 64 for I2E)
- RoPE position embedding fix for batch processing

### Experimental
- EliGen entity control (mask-based spatial editing, untested)
- Spatial coordinate tokens (not used by DiffSynth, see `explorations/20251003_diffsynth_spatial_token_analysis.md`)
- Wrapper nodes (transformers/diffusers, incomplete ComfyUI integration)

## Core Architecture

### Data Flow
```
LoadImage → QwenVLTextEncoder → KSampler → VAEDecode
              ↑ (vision+text)      ↑ (latents)
      QwenTemplateBuilder    QwenVLEmptyLatent
```

### Conditioning System
- Text embeddings: Qwen2.5-VL → 3584 dimensions
- Vision tokens: `<|vision_start|><|image_pad|><|vision_end|>` per image
- Multi-image: Auto "Picture X:" labels when `auto_label=True` (2+ images)
- Token dropping: Applied post-encoding to match DiffSynth behavior
- Reference latents: Attached to conditioning dict (16-channel, from VAE)

### Resolution Handling
- **32-pixel alignment** for VAE (required)
- Vision encoder: 384×384 target area
- VAE encoder: 1024×1024 target area
- `calculate_dimensions()` in `nodes/qwen_vl_encoder.py:187`

### Inpainting System
- QwenMaskProcessor: Mask preprocessing (blur, expand, feather) - outputs IMAGE, MASK, preview
- QwenInpaintSampler: Implements `final = (1-mask)*original + mask*generated` (548 lines - consider using KSampler + LatentCompositeMasked instead)
- Encoder inpainting mode: Accepts mask input, auto-resizes to match VAE dimensions
- Template Builder: Includes "inpainting" template mode
- Prompts: Use text encoder or template builder (NOT mask processor)
- See `example_workflows/qwen_edit_2509_mask_inpainting.json`

## Node Categories

### Core (QwenImage/Loaders, Encoding)
- QwenVLCLIPLoader - Load Qwen2.5-VL model
- QwenVLTextEncoder - Main encoder (3 modes: text_to_image, image_edit, inpainting)
  - Inpainting mode: Accepts optional `inpaint_mask`, auto-resizes to VAE dimensions
  - Default prompts cleared (was "A beautiful landscape")
- QwenTemplateBuilder - System prompt templates (includes "inpainting" mode)
  - Default prompts cleared

### Latents (QwenImage/Latents)
- QwenVLEmptyLatent - 16-channel latent creation
- QwenVLImageToLatent - Image to 16-channel latent

### Inpainting (QwenImage/Mask, Sampling)
- QwenMaskProcessor - Mask preprocessing (outputs: IMAGE, MASK, preview, mask_preview)
  - No longer outputs prompt (use text encoder instead)
- QwenInpaintSampler - Diffusers-pattern sampler
  - Alternative: Use KSampler + LatentCompositeMasked for simpler workflow

### Refinement (QwenImage/Refinement)
- QwenLowresFixNode - Two-stage upscale refinement

### Experimental (Archived or Untested)
- QwenEliGenEntityControl - Mask-based spatial (untested)
- QwenSpatialTokenGenerator - Coordinate tokens (deprecated)
- Wrapper nodes (transformers/diffusers, incomplete)

## Workflow Examples

See `example_workflows/` for complete JSON workflows:
- `qwen_edit_2509_single_image_edit.json` - Basic image editing
- `qwen_edit_2509_mask_inpainting.json` - Mask-based inpainting
- `nunchaku_qwen_mask_inpainting.json` - Quantized model variant

### Text-to-Image
```
QwenVLCLIPLoader → QwenTemplateBuilder → QwenVLTextEncoder (mode: text_to_image)
                                              ↓
QwenVLEmptyLatent → KSampler → VAEDecode → SaveImage
```

### Image Editing
```
LoadImage → QwenVLTextEncoder (mode: image_edit, with edit_image input)
              ↓
QwenVLEmptyLatent → KSampler (denoise: 0.5-0.7) → VAEDecode
```

### Mask Inpainting
```
LoadImage → QwenMaskProcessor (outputs: image, mask, preview)
              ↓              ↓
            VAEEncode    (mask input)
                ↓            ↓
         QwenVLTextEncoder (mode: inpainting, mask auto-resized to VAE dims)
                ↓
         QwenInpaintSampler ← mask from QwenMaskProcessor
                ↓
         VAEDecode → SaveImage
```

**Alternative Simplified Workflow:**
```
LoadImage → QwenMaskProcessor → KSampler → LatentCompositeMasked → VAEDecode
              ↓ (mask)                         ↑ (mask)
              └──────────────────────────────────┘
```

**Recommended Settings:**
- Resolution: 1024×1024 or 512×512
- Steps: 8 (Lightning LoRA) or 20-30 (base)
- CFG: 7.0-9.0
- Sampler: euler, euler_ancestral
- Denoise: 1.0 (T2I), 0.5-0.7 (edit), 1.0 (inpaint strength)

## Technical Details

### Vision Token Format
```
Single: <|vision_start|><|image_pad|><|vision_end|>
Multi:  Picture 1: <|vision_start|><|image_pad|><|vision_end|>Picture 2: ...
```

### Template Format (DiffSynth-compatible)
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{vision_tokens}{text}<|im_end|>
<|im_start|>assistant
```

### Token Dropping
- Text-to-image: Drop first 34 embeddings
- Image edit: Drop first 64 embeddings
- Inpainting: Drop first 64 embeddings (same as image_edit)
- Applied AFTER encoding in QwenProcessorV2
- Only applied when system_prompt is provided

### 16-Channel VAE
- Shape: `[B, 16, H/8, W/8]`
- Wan21 format (Qwen-specific)
- Standard ComfyUI VAE nodes work when loaded correctly

## Integration Points

- ComfyUI CLIP system: `CLIPType.QWEN_IMAGE`
- Model paths: `models/text_encoders/`, `models/diffusion_models/`, `models/vae/`
- Dependencies: KJNodes (Image Batch for multi-image)
- Wrapper nodes: transformers, diffusers (optional, experimental)

## Known Issues

1. **Multi-image memory**: 4+ images may OOM (optimal: 1-3)
2. **Wrapper nodes**: Incomplete ComfyUI sampler integration
3. **Spatial tokens**: Not used by DiffSynth (use EliGen/masks instead)
4. **QwenInpaintSampler**: 548-line implementation for simple blend - consider using KSampler + LatentCompositeMasked instead

## Debug Features

- `debug_mode=True` in encoder: UI output with token/dimension details
- `verbose_log=True`: Console tracing of model forward passes
- QwenDebugController: Centralized debug interface