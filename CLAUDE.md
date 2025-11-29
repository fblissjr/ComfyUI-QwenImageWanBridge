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
- `nodes/docs/wan21_vae_upscale2x_guide.md` - Wan2.1-VAE-upscale2x integration and optimization guide
- `nodes/docs/qwen_wan_bridge_guide.md` - Qwen-to-Wan Video bridge for image-to-video workflows
- `nodes/docs/z_image_encoder.md` - Z-Image encoder documentation (thinking token fix)

## Project Overview

ComfyUI nodes for Qwen-Image-Edit model, enabling text-to-image generation and vision-based image editing using Qwen2.5-VL 7B. Bridges DiffSynth-Studio patterns with ComfyUI's node system.

**Key Features (v2.9.10):**
- **Extended template format** - Templates can include `add_think_block`, `thinking_content`, `assistant_content` in YAML frontmatter
- **Structured prompt templates** - `json_structured`, `yaml_structured`, `markdown_structured` with pre-configured thinking
- **JSON key quote filtering** - `strip_key_quotes` toggle on Z-Image nodes prevents JSON keys appearing as text in images - [docs](nodes/docs/z_image_encoder.md)
- **HunyuanVideo 1.5 T2V** - Text-to-video with Qwen2.5-VL encoder (23 video templates)
- **File-based template system** - Templates in `nodes/templates/*.md` files (single source of truth)
- **Template Builder → Encoder** - Single `template_output` connection handles everything
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
- VAE: `qwen_image_vae.safetensors` (16-channel, standard) → `models/vae/`
  - Alternative: `Wan2.1-VAE-upscale2x` (16-channel, 2x decoder upscaling) - [guide](nodes/docs/wan21_vae_upscale2x_guide.md)

## Implementation Status

### Core Features
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

### Deprecated / Experimental / Research Only
- Mask-based inpainting (QwenMaskProcessor + QwenInpaintSampler) - [docs](nodes/docs/QwenMaskProcessor.md), [docs](nodes/docs/QwenInpaintSampler.md)
- EliGen entity control (mask-based spatial editing, untested)
- Spatial coordinate tokens (not used by DiffSynth, see `explorations/20251003_diffsynth_spatial_token_analysis.md`)
- QwenSmartCrop - Automated face cropping with VLM detection (experimental) - [docs](nodes/docs/QwenSmartCrop.md)
- QwenEliGenEntityControl - Mask-based spatial (untested)
- QwenSpatialTokenGenerator - Coordinate tokens (deprecated)

### Archived (nodes/archive/wrapper/)
- Wrapper nodes (transformers/diffusers) - had VRAM leak issues, not production ready
- Custom DiT implementation (`models/qwen_image_dit.py`) - unbounded RoPE cache

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

**Two Separate Paths (IMPORTANT):**
- **Vision Encoder (VL Model)**: 384×384 area-based scaling (hardcoded, not configurable)
  - Purpose: Semantic understanding, creates vision tokens
  - Model: Qwen2.5-VL vision encoder
  - Not affected by `vae_max_dimension` parameter
- **VAE Encoder (Generation Model)**: Configurable via `vae_max_dimension` parameter
  - Purpose: Pixel-level detail, creates reference latents for generation
  - Controls output resolution (encode 1792px → output 1792px with standard VAE, 3584px with 2x VAE)
  - Default: 2048px (current), recommended 1792px for Wan2.1-upscale2x

**VAE Options:**
- **Standard Wan VAE**: 8x spatial scale (encode 1024px → output 1024px)
- **Wan2.1-VAE-upscale2x**: 16x spatial scale (encode 1024px → output 2048px via 2x decoder)
  - Same latent space (16-channel), same VRAM during sampling
  - Upscaling happens at decode, not during generation
  - Eliminates "wan speckles/polka dots" artifacts
  - [Full guide](nodes/docs/wan21_vae_upscale2x_guide.md)

**Alignment:**
- **32-pixel alignment** for VAE (required)
- **28-pixel alignment** for vision encoder
- `calculate_dimensions()` in `nodes/qwen_vl_encoder.py:203` (VAE), `qwen_vl_encoder.py:227` (vision)
- Advanced encoder: `vae_max_dimension` sets base, `resolution_mode` applies weights

### Inpainting System (Simple Blending Approach) - NOT YET WORKING / TESTED

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

**DiffSynth Alternative (Not implemented and no plans to do so):**
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
  - Single `template_output` connection to encoder (handles mode, prompt, system)
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
- PromptKeyFilter - Strip quotes from JSON-style keys in prompts
  - Converts `"subject": "description"` to `subject: "description"`
  - Prevents key names appearing as visible text in generated images
  - Useful when piping structured prompts from LLMs
  - Toggle on/off via `strip_key_quotes` parameter (default: on)
  - Works with any text encoder (Z-Image, Qwen-Image-Edit, HunyuanVideo)

### Wan Video Bridge (QwenWanBridge) - EXPERIMENTAL AND LIKELY NOT USEFUL YET
- QwenToWanFirstFrameLatent - Prepare Qwen output for Wan Video first frame
  - Encodes to 16-channel latent with temporal dimension
  - Applies Wan Video normalization (mean/std per channel)
  - Compatible with DiffSynth-Studio and Diffusers pipelines
  - [Full guide](nodes/docs/qwen_wan_bridge_guide.md)
- QwenToWanLatentSaver - Export first frame latent for external tools
  - Formats: safetensors, pt, npz
  - Includes metadata for compatibility
- QwenToWanImageSaver - Save edited first frame for verification

### Inpainting (QwenImage/Mask, Sampling) - EXPERIMENTAL AND NOT WORKING / TESTED
- QwenMaskProcessor - Mask preprocessing (outputs: IMAGE, MASK, preview, mask_preview)
  - No longer outputs prompt (use text encoder instead)
- QwenInpaintSampler - Diffusers-pattern sampler
  - Alternative: Use KSampler + LatentCompositeMasked for simpler workflow

### Refinement (QwenImage/Refinement)
- QwenLowresFixNode - Two-stage upscale refinement

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

### Headshot Swap (Experimental)
```
LoadImage (portrait) → QwenSmartCrop (auto_fallback, face_headshot)
                           ↓ (tight face crop)
LoadImage (scene) ─────────┼─→ QwenImageBatch → QwenVLTextEncoder (multi_image_edit)
                                                       ↓
                                          QwenVLEmptyLatent → KSampler → VAEDecode
```

**Prompt:** Describe subjects semantically, not by image numbers
- "Replace the face of [person in scene] with the face of [person in crop], while keeping the same pose, clothing, lighting, background from the scene"

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

### Mask Inpainting - NOT YET WORKING / TESTED
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

## HunyuanVideo 1.5 Support (v2.8.1)

### Overview
Text-to-video encoding using Qwen2.5-VL with ComfyUI's native HunyuanVideo sampler/VAE.

### Nodes

#### Loader (HunyuanVideo/Loaders)
- **HunyuanVideoCLIPLoader**: Load Qwen2.5-VL + optional byT5

#### Template Builder (HunyuanVideo/Templates)
- **HunyuanVideoTemplateBuilder**: Build prompts with 23 video templates
  - Output: `template_output` (HUNYUAN_TEMPLATE) connects to encoder's `template_input`
  - Select preset template and customize system prompt

#### Encoder (HunyuanVideo/Encoding)
- **HunyuanVideoTextEncoder**: T2V with dual output (positive, negative, debug)
  - Optional `template_input`: Connect from Template Builder
  - Built-in `template_preset` dropdown (used if template_input not connected)
  - Default negative: "low quality, blurry, distorted, artifacts, watermark, text, logo"
  - Priority: template_input > custom_system_prompt > template_preset

### Workflows

**With Template Builder + KSampler (recommended):**
```
HunyuanVideoCLIPLoader ─────────────────────────┐
                                                ↓
HunyuanVideoTemplateBuilder → template_output → HunyuanVideoTextEncoder → positive → KSampler
                                                                        → negative ↗
```

**Without Template Builder (use encoder dropdown):**
```
HunyuanVideoCLIPLoader → HunyuanVideoTextEncoder → positive → KSampler
                        (set template_preset)    → negative ↗
```

**With SamplerCustomAdvanced (needs CFGGuider):**
```
HunyuanVideoTextEncoder → positive → CFGGuider → GUIDER → SamplerCustomAdvanced
                        → negative ↗
```

### Usage Notes
- **byT5**: Quoted text (e.g., `"你好"`) triggers byT5 multilingual rendering
- **Templates**: 23 video templates in `nodes/templates/hunyuan_video_*.md`
- **Basic encoding**: Use CLIPTextEncode for no-frills encoding without templates

### Documentation
- `nodes/docs/hunyuanvideo_15_workflow_guide.md` - T2V setup
- `nodes/docs/hunyuanvideo_15_resolution_frame_guide.md` - Technical specs
- `nodes/docs/hunyuanvideo_prompting_experiments.md` - Prompting experiments guide
- `example_workflows/hunyuanvideo_15_t2v_example.json` - Working T2V workflow

## Z-Image Support (v2.9.9)

### Overview
Z-Image is Alibaba's 6B parameter text-to-image model using Qwen3-4B as the text encoder. Our nodes implement the correct Qwen3-4B chat template format.

### Qwen3-4B Template Format

**Correct format (from `tokenizer_config.json`):**
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
<think>
{thinking_content}
</think>

{assistant_content}
```

**Our implementation:**
- `user_prompt` - the user's generation request
- `thinking_content` - content INSIDE `<think>...</think>` tags
- `assistant_content` - content AFTER `</think>` tags (what assistant says after thinking)
- `add_think_block` - auto-enabled when `thinking_content` is provided

**Closing tag behavior (v2.9.6):**
- Empty `assistant_content`: No closing `<|im_end|>` (matches diffusers, model is "generating")
- With `assistant_content`: Closes with `<|im_end|>` (complete message)

### Nodes

#### ZImage/Encoding
- **ZImageTextEncoder**: Full-featured encoder with Qwen3-4B template
  - Handles complete first turn (system + user + assistant)
  - `user_prompt` - your generation request
  - `conversation_override` - optional input from ZImageTurnBuilder (overrides all other inputs)
  - `template_preset` - auto-fills system_prompt from templates
  - `system_prompt` - editable after template auto-fill
  - `add_think_block` - add `<think></think>` structure
  - `thinking_content` / `assistant_content` - content for assistant response
  - `raw_prompt` - bypass all formatting, use your own tokens
  - `strip_key_quotes` - remove quotes from JSON keys to prevent them appearing as text
  - **Outputs**: conditioning, formatted_prompt, debug_output, conversation
- **ZImageTextEncoderSimple**: Simplified encoder for quick use / negative prompts
  - Same template/thinking support, no conversation chaining
  - **Outputs**: conditioning, formatted_prompt

#### ZImage/Conversation
- **ZImageTurnBuilder**: Add conversation turns for multi-turn workflows
  - Each node = one turn (user message + assistant response)
  - `previous` (required) - from encoder or another TurnBuilder
  - `clip` (optional) - connect to output conditioning directly
  - `user_prompt` - user's message for this turn
  - `thinking_content` / `assistant_content` - optional assistant response
  - `is_final` - if True (default), last message has no `<|im_end|>`
  - `strip_key_quotes` - remove quotes from JSON keys to prevent them appearing as text
  - **Outputs**: conversation, conditioning (if clip connected), formatted_prompt, debug_output

### Workflow

**Basic (matches diffusers):**
```
CLIPLoader (qwen_3_4b, lumina2) → ZImageTextEncoder → KSampler
                                  user_prompt: "A cat sleeping"
```

**With templates:**
```
CLIPLoader → ZImageTextEncoder → KSampler
             template_preset: photorealistic
             user_prompt: "A cat sleeping"
```

**Multi-turn conversation (with direct encoding):**
```
ZImageTextEncoder → ZImageTurnBuilder (clip connected) → conditioning → KSampler
(first turn)        (additional turn, outputs conditioning directly)
```

**Multi-turn conversation (chain to encoder):**
```
ZImageTextEncoder → ZImageTurnBuilder → ZImageTextEncoder
(first turn)        (additional turn)    conversation_override: (connect)
```

### Key Differences from Qwen-Image-Edit

| Aspect | Z-Image | Qwen-Image-Edit |
|--------|---------|-----------------|
| Text Encoder | Qwen3-4B (instruct, 2560 dim) | Qwen2.5-VL 7B (instruct, 3584 dim) |
| Vision | None (text-only) | Full vision encoder |
| Steps | 8-9 (turbo distilled) | 20-50 |
| CFG | 0 (baked in via DMD) | 5-8 |
| Template | Chat + thinking tokens | DiffSynth chat template |

### Technical Details
- **Decoupled DMD distillation**: CFG is "baked" into the model during training
- **Turbo variant**: 8 NFEs (Number of Function Evaluations)
- **VAE**: Same 16-channel latent space as other Wan models
- **Architecture**: S3-DiT (Single-Stream DiT) with 6B parameters
- **Qwen3-4B specs**: 2560 hidden dim, 36 layers, `hidden_states[-2]` for embeddings

### Templates
Templates in `nodes/templates/z_image/` subfolder (examples):
- `default.md` - No system prompt
- `photorealistic.md` - Natural lighting and textures
- `bilingual_text.md` - English/Chinese text rendering
- `artistic.md` - Creative compositions
- Many more available (see folder for full list)

### Documentation
- [z_image_intro.md](nodes/docs/z_image_intro.md) - "WTF is this?" intro guide with progressive examples
- [z_image_encoder.md](nodes/docs/z_image_encoder.md) - Full technical reference
- [z_image_character_generation.md](nodes/docs/z_image_character_generation.md) - Multi-turn character consistency
- [z_image_turbo_workflow_analysis.md](nodes/docs/z_image_turbo_workflow_analysis.md) - Official workflow analysis

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

## VAE Compatibility and Recommendations

### Standard Wan VAE (qwen_image_vae.safetensors)
- 16-channel, 8x spatial scaling
- Works out of the box, no special configuration
- May have "wan speckles/polka dots" artifacts
- Recommended `vae_max_dimension`: 2048px

### Wan2.1-VAE-upscale2x (Recommended for Quality)
- 16-channel, 16x spatial scaling (2x in decoder)
- Eliminates speckle/grain artifacts
- Same VRAM during sampling (upscaling at decode)
- Trained on real images (may struggle with anime/lineart)
- Recommended `vae_max_dimension`: 1792px (outputs 3584px, model maximum)
- Install via: [ComfyUI-VAE-Utils](https://github.com/spacepxl/ComfyUI-VAE-Utils)
- [Full integration guide](nodes/docs/wan21_vae_upscale2x_guide.md)

**Resolution recommendations with Wan2.1-upscale2x:**
- 8GB VRAM: vae_max_dimension=1024 → 2048px output
- 12GB VRAM: vae_max_dimension=1792 → 3584px output (recommended)
- 16GB+ VRAM: vae_max_dimension=2048 → 4096px output (experimental)
- Multi-image: Lower by ~256px for VRAM safety (e.g., 1536 for 3 images)

## Integration Points

- ComfyUI CLIP system: `CLIPType.QWEN_IMAGE`
- Model paths: `models/text_encoders/`, `models/diffusion_models/`, `models/vae/`
- Dependencies: None (QwenImageBatch replaces KJNodes ImageBatchMulti)
- Optional: KJNodes (for other utilities), ComfyUI-VAE-Utils (for Wan2.1-upscale2x)
- Wrapper nodes: Archived in `nodes/archive/wrapper/` (had VRAM issues)

## Implementation Decisions

## Known Issues

1. **Multi-image memory**: 4+ images may OOM (optimal: 1-3, use `max_dimension_1024` for VRAM relief)
2. **Spatial tokens**: Not used by DiffSynth (use EliGen/masks instead)
3. **Inpainting approach**: Post-processing blend, not in-model attention masking like DiffSynth, see: [internal/inpainting_vs_eligen.md](internal/inpainting_vs_eligen.md)
4. **QwenInpaintSampler**: 548-line implementation for simple blend - consider using KSampler + LatentCompositeMasked instead, deprecated for now
5. **Zoom-out on large images**: Fixed in v2.6.1 with `preserve_resolution` default (was `area_1024`)

## Debug Features

- `debug_mode=True` in encoder: UI output with token/dimension details
- `verbose_log=True`: Console tracing of model forward passes
- QwenDebugController: Centralized debug interface

### Z-Image Debug Output (v2.9.6)
- Debug output escapes HTML (`<` -> `&lt;`) so think tags display correctly in Preview nodes
- Explicit "Think Tag Check" shows `Contains '<think>': True/False`
- Formatted prompt logged to server console: `[Z-Image] Formatted prompt:`
