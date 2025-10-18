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
- `nodes/docs/` - Detailed node documentation (README, per-node docs, guides)
- `nodes/archive/` - Legacy and experimental nodes
- `example_workflows/` - Example JSON workflows with comprehensive notes
- `internal/` - Internal documentation and analysis

## Documentation

- `nodes/docs/README.md` - Documentation index
- `nodes/docs/QwenImageBatch.md` - Batch node documentation
- `nodes/docs/QwenVLTextEncoder.md` - Standard encoder documentation
- `nodes/docs/QwenTemplateBuilder.md` - Template builder documentation
- `nodes/docs/resolution_tradeoffs.md` - Comprehensive resolution and scaling guide

## Project Overview

ComfyUI nodes for Qwen-Image-Edit model, enabling text-to-image generation and vision-based image editing using Qwen2.5-VL 7B. Bridges DiffSynth-Studio patterns with ComfyUI's node system.

**Key Features (v2.7.0):**
- **File-based template system** - 9 templates in `nodes/templates/*.md` files (single source of truth)
- **Template Builder → Encoder auto-sync** - Connect BOTH `mode` and `system_prompt` outputs (required)
- QwenImageBatch node (auto-detection, aspect preservation, double-scaling prevention) - [docs](nodes/docs/QwenImageBatch.md)
- multi_image_edit mode (DiffSynth `encode_prompt_edit_multi` alignment)
- Resolution scaling guide - [docs](nodes/docs/resolution_tradeoffs.md)
- 16-channel VAE latents (vs standard 4-channel)
- Vision token processing with multi-image support
- Token dropping (34 for T2I, 64 for I2E/multi/inpainting) - DiffSynth-compatible
- Mask-based inpainting with diffusers blending pattern (experimental, not fully tested)
- Native ComfyUI integration via `CLIPType.QWEN_IMAGE`

**Models:**
- Text/Vision Encoder: `Qwen/Qwen2.5-VL-7B-Instruct` → `models/text_encoders/`
- DiT Model: `qwen-image-edit-2509` (fp8 or Nunchaku quantized) → `models/diffusion_models/`
- VAE: `qwen_image_vae.safetensors` (16-channel) → `models/vae/`

## Implementation Status

### Production Ready
- Text-to-image generation (QwenVLTextEncoder) - [docs](nodes/docs/QwenVLTextEncoder.md)
- Single/multi-image editing with vision tokens
- QwenImageBatch (aspect preservation, auto-detection, double-scaling prevention) - [docs](nodes/docs/QwenImageBatch.md)
- **File-based template system (9 templates)** - [docs](nodes/docs/QwenTemplateBuilder.md)
  - Templates: `default_t2i`, `default_edit`, `multi_image_edit`, `artistic`, `photorealistic`, `minimal_edit`, `technical`, `inpainting`, `raw`
  - Stored in `nodes/templates/*.md` with YAML frontmatter
  - JavaScript UI auto-fills `custom_system` field for editing
- Resolution scaling guide - [docs](nodes/docs/resolution_tradeoffs.md)
- 16-channel VAE support
- Multi-image "Picture X:" labeling (auto, 1-3 images optimal)
- Token dropping (34 for T2I, 64 for I2E/multi/inpainting)
- RoPE position embedding fix for batch processing

### Experimental (Not Fully Tested)
- Mask-based inpainting (QwenMaskProcessor + QwenInpaintSampler) - [docs](nodes/docs/QwenMaskProcessor.md), [docs](nodes/docs/QwenInpaintSampler.md)
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
- Vision encoder: 384×384 target area (always area-based scaling)
- VAE encoder: Configurable via `scaling_mode` parameter
  - `preserve_resolution` (default): Keeps original size with 32px alignment
  - `max_dimension_1024`: Scales largest side to 1024px
  - `area_1024`: Scales to ~1024×1024 area (legacy)
- `calculate_dimensions()` in `nodes/qwen_vl_encoder.py:193`
- Advanced encoder: `scaling_mode` sets base, `resolution_mode` applies weights

### Inpainting System (Simple Blending Approach)

**Our Implementation:**
- QwenMaskProcessor: Mask preprocessing (blur, expand, feather) - outputs IMAGE, MASK, preview
- QwenInpaintSampler: Post-processing blend `final = (1-mask)*original + mask*generated`
- Encoder inpainting mode: Accepts mask input, auto-resizes to match VAE dimensions
- Template Builder: Includes "inpainting" template mode
- Prompts: Use text encoder or template builder (NOT mask processor)

**Approach:** Simple latent blending after generation (post-processing)
- Works with existing qwen-image-edit model
- No DiT modifications required
- ComfyUI-friendly implementation
- Single mask + single prompt per operation

**DiffSynth Alternative (Not Implemented):**
- EliGen uses attention masking INSIDE the DiT (requires model access)
- Multi-entity with isolated attention per region
- QwenEliGenEntityControl node exists but untested
- Would need ComfyUI DiT integration (may not be possible)

See `example_workflows/qwen_edit_2509_mask_inpainting.json`

## Node Categories

### Core (QwenImage/Loaders, Encoding)
- QwenVLCLIPLoader - Load Qwen2.5-VL model
- QwenVLTextEncoder - Main encoder (4 modes: text_to_image, image_edit, multi_image_edit, inpainting)
  - `mode` input is STRING (accepts connection from Template Builder or manual typing)
  - **multi_image_edit**: DiffSynth-compatible multi-reference mode (vision tokens inside prompt)
  - **image_edit**: Single/multi image (vision tokens before prompt, auto_label optional)
  - **inpainting**: Mask-based editing (accepts `inpaint_mask`, auto-resizes to VAE)
- QwenVLTextEncoderAdvanced - Power user encoder (same 4 modes + weighted resolution)
  - Per-image resolution control and weighted importance
- QwenTemplateBuilder - File-based system prompt templates (9 templates)
  - Templates loaded from `nodes/templates/*.md` with YAML frontmatter
  - **REQUIRED**: Connect BOTH `mode` and `system_prompt` outputs to encoder
  - JavaScript UI auto-fills `custom_system` field when selecting presets

### Latents (QwenImage/Latents)
- QwenVLEmptyLatent - 16-channel latent creation
- QwenVLImageToLatent - Image to 16-channel latent

### Utilities (QwenImage/Utilities)
- QwenImageBatch - Aspect-ratio preserving batch node
  - Skips None/empty inputs (no black images)
  - Preserves aspect ratios (no cropping)
  - Applies v2.6.1 scaling modes
  - Up to 10 image inputs
  - Compatible with both standard and advanced encoders
  - Two-stage scaling: batch normalizes, advanced encoder applies weights

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

### Image Editing (Single Image)
```
LoadImage → QwenVLTextEncoder (mode: image_edit, with edit_image input)
              ↓
QwenVLEmptyLatent → KSampler (denoise: 0.5-0.7) → VAEDecode
```

### Multi-Image Editing
```
LoadImage ─┐
LoadImage ─┼─> QwenImageBatch → QwenVLTextEncoder (mode: image_edit)
LoadImage ─┘    (scaling_mode)        ↓
                              QwenVLEmptyLatent → KSampler → VAEDecode
```

### Multi-Image with Advanced Encoder
```
LoadImage ─┐
LoadImage ─┼─> QwenImageBatch ────────> QwenVLTextEncoderAdvanced
LoadImage ─┘    (scaling_mode:          (resolution_mode: hero_first,
                 preserve_resolution      hero_weight: 1.5,
                 or no_scaling)           reference_weight: 0.5)
                                                ↓
                                   QwenVLEmptyLatent → KSampler → VAEDecode
```

**Recommended Settings:**
- **With Standard Encoder**: Use `scaling_mode=preserve_resolution` in QwenImageBatch
- **With Advanced Encoder**:
  - Option 1: `scaling_mode=no_scaling` (let advanced encoder handle all scaling)
  - Option 2: `scaling_mode=preserve_resolution` (batch normalizes, advanced applies weights)

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
- Dependencies: None (QwenImageBatch replaces KJNodes ImageBatchMulti)
- Optional: KJNodes (for other utilities)
- Wrapper nodes: transformers, diffusers (optional, experimental)

## Implementation Decisions

### Inpainting Approach: Simple Blending vs EliGen

**Decision:** Implemented simple post-processing blend (Choice A)

**Rationale:**
- Works with existing qwen-image-edit model (no new downloads)
- No DiT modifications required (ComfyUI may not support this)
- Standard ComfyUI workflow integration
- Simpler to maintain and debug

**Trade-offs:**
- ✅ Works now with current infrastructure
- ✅ No model access required
- ✅ Compatible with all samplers
- ❌ Not full DiffSynth approach (they use attention masking)
- ❌ Single mask/prompt only (no multi-entity)
- ❌ Post-processing, not in-model control

**DiffSynth EliGen (Not Implemented):**
- Requires DiT `process_entity_masks()` method (lines 434-484 in qwen_image_dit.py)
- Modifies attention to isolate entity prompts to masked regions
- Multi-entity support with prevented cross-attention
- Would need ComfyUI DiT wrapper modifications

**Alternative:** QwenEliGenEntityControl node exists but is untested and requires DiT integration.

## Known Issues

1. **Multi-image memory**: 4+ images may OOM (optimal: 1-3, use `max_dimension_1024` for VRAM relief)
2. **Wrapper nodes**: Incomplete ComfyUI sampler integration
3. **Spatial tokens**: Not used by DiffSynth (use EliGen/masks instead)
4. **Inpainting approach**: Post-processing blend, not in-model attention masking like DiffSynth
5. **QwenInpaintSampler**: 548-line implementation for simple blend - consider using KSampler + LatentCompositeMasked instead
6. **Zoom-out on large images**: Fixed in v2.6.1 with `preserve_resolution` default (was `area_1024`)

## Debug Features

- `debug_mode=True` in encoder: UI output with token/dimension details
- `verbose_log=True`: Console tracing of model forward passes
- QwenDebugController: Centralized debug interface