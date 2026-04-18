# QwenVLTextEncoder

**Category:** QwenImage/Encoding
**Display Name:** Qwen2.5-VL Text Encoder

## Description

Main encoder for Qwen2.5-VL supporting text-to-image generation and image editing. Implements DiffSynth/Diffusers-compatible encoding with proper vision token handling, multi-image support, and automatic mode synchronization with Template Builder.

**IMPORTANT:** When using Template Builder, connect BOTH `mode` and `system_prompt` outputs to this encoder. The `mode` connection ensures vision tokens are formatted correctly.

## Inputs

### Required
- **clip** (CLIP)
  - Qwen2.5-VL model from QwenVLCLIPLoader

- **text** (STRING, multiline)
  - Your prompt text
  - Default: "" (empty)

- **mode** (STRING)
  - `text_to_image`: Generate from scratch (no vision tokens)
  - `image_edit`: Modify existing image (vision tokens before prompt)
  - `multi_image_edit`: Multi-reference editing (vision tokens inside prompt with Picture labels)
  - `inpainting`: Mask-based editing (vision tokens before prompt)
  - **Connect from Template Builder `mode` output for auto-sync**
  - Can also type manually: "text_to_image", "image_edit", etc.
  - Default: `image_edit`

### Optional
- **edit_image** (IMAGE)
  - Single image or batch (use QwenImageBatch for multiple)
  - Required for `image_edit` and `multi_image_edit` modes
  - Auto-detects pre-scaled images from QwenImageBatch (skips VAE scaling)

- **vae** (VAE)
  - Required for image editing - encodes reference latents
  - 16-channel Qwen VAE recommended

- **system_prompt** (STRING, multiline)
  - System prompt from Template Builder
  - Enables proper token dropping when provided
  - **Connect from Template Builder `system_prompt` output**
  - Default: ""

- **scaling_mode** (ENUM)
  - `preserve_resolution` (default): Keeps original size with 32px alignment
  - `max_dimension_1024`: Scales largest side to 1024px
  - `area_1024`: Scales to ~1024x1024 area (legacy)
  - Note: Auto-skipped when using QwenImageBatch (no double-scaling)
  - Default: `preserve_resolution`

- **debug_mode** (BOOLEAN)
  - Show processing details in UI output
  - Default: False

- **auto_label** (BOOLEAN)
  - Automatically add "Picture X:" labels for 2+ images
  - Matches DiffSynth standard
  - Default: True

- **verbose_log** (BOOLEAN)
  - Enable verbose console logging
  - Default: False

## Outputs

- **conditioning** (CONDITIONING)
  - Text/vision embeddings (3584 dimensions)
  - Includes reference latents (if VAE provided)
  - Token dropping applied if system_prompt provided

- **debug_output** (STRING)
  - Detailed processing information when debug_mode=True
  - Shows vision tokens, dimensions, scaling info, pre-scaling detection

## Implementation Details

### Resolution Handling
- **Vision encoder**: 384×384 target area (always area-based scaling)
- **VAE encoder**: Configurable via `scaling_mode` parameter
- **32-pixel alignment** for both (required for VAE compatibility)
- `calculate_dimensions()` preserves aspect ratio
- Pre-scaled detection: Skips VAE scaling if images from QwenImageBatch

### Scaling Modes
- `preserve_resolution`: Keeps original size (32px aligned) - best quality
- `max_dimension_1024`: Scales largest side to 1024px - reduces VRAM
- `area_1024`: Scales to ~1024x1024 area - legacy behavior
- See [resolution_tradeoffs.md](resolution_tradeoffs.md) for detailed analysis

### Vision Token Format
- **image_edit mode**:
  - Single image: `<|vision_start|><|image_pad|><|vision_end|>`
  - Multi-image (auto_label=True): `Picture 1: <|vision_start|><|image_pad|><|vision_end|>Picture 2: ...`
  - Multi-image (auto_label=False): Concatenated vision tokens without labels
- **multi_image_edit mode** (DiffSynth `encode_prompt_edit_multi`):
  - Vision tokens placed INSIDE prompt (not before)
  - Always uses "Picture X:" labels
  - Matches DiffSynth-Studio pattern

### Token Dropping (DiffSynth-compatible)
- **text_to_image**: Drop first 34 embeddings
- **image_edit**: Drop first 64 embeddings
- **multi_image_edit**: Drop first 64 embeddings
- Only applied when `system_prompt` is provided
- Applied AFTER encoding (not during)

### Template Mode Auto-Sync
- Connect Template Builder `mode` output → encoder `template_mode` input
- Encoder automatically syncs to template's mode
- Prevents vision token placement/drop index mismatches
- Manual mode selection overridden when template_mode connected

### Reference Latents
- Each image encoded separately via VAE
- Attached to conditioning dict as `reference_latents`
- 16-channel format: `[B, 16, H/8, W/8]`
- Passed through without dimension manipulation

## Example Usage

### Text-to-Image
```
QwenVLCLIPLoader → QwenVLTextEncoder (mode: text_to_image, text: "sunset over mountains")
                        ↓
                   Conditioning → KSampler
```

### Single Image Editing
```
LoadImage → QwenVLTextEncoder (mode: image_edit, text: "make it rainy")
                 ↓
            Conditioning → KSampler
```

### Multi-Image Editing (with QwenImageBatch)
```
LoadImage ─┐
LoadImage ─┼─> QwenImageBatch → QwenVLTextEncoder (mode: image_edit or multi_image_edit)
LoadImage ─┘                            ↓
                                   Conditioning → KSampler
```

### With Template Builder (Recommended)
```
QwenTemplateBuilder (template_mode: multi_image_edit)
      ├─ (system_prompt) ──> QwenVLTextEncoder (system_prompt input)
      └─ (mode) ───────────> QwenVLTextEncoder (mode input)
                              Both connections required!
```

**Why connect both outputs?**
- `system_prompt`: Provides instruction text for the model
- `mode`: Ensures correct vision token formatting (labels, placement, token dropping)
- Missing `mode` connection = vision token mismatch = broken generation

## Multi-Image Support

- Use QwenImageBatch node to combine images (recommended)
- Auto-detects up to 10 images, skips empty inputs
- Optimal: 1-3 images (4+ may cause memory issues)
- Each image processed individually for vision/VAE
- Automatic "Picture X:" labeling when auto_label=True (image_edit mode with 2+ images)
- QwenImageBatch prevents double-scaling via metadata propagation
- See [QwenImageBatch.md](QwenImageBatch.md) and [resolution_tradeoffs.md](resolution_tradeoffs.md)

## Debug Mode Output

When `debug_mode=True`, shows:
- Input/output shapes
- Vision token formatting
- Full prompt being encoded
- Processing dimensions for each image
- Pre-scaled detection (from QwenImageBatch)
- Scaling mode used (or skipped if pre-scaled)
- Reference latent information
- Character counts

## RoPE Fix

Includes monkey patch for batch processing with different image sizes:
```python
if isinstance(video_fhw, list):
    video_fhw = tuple(max([i[j] for i in video_fhw]) for j in range(3))
```
Applied on module load to fix position embedding issues.

## Related Nodes

- QwenVLCLIPLoader - Provides CLIP input
- QwenTemplateBuilder - Provides system_prompt and mode inputs
- QwenImageBatch - Provides batched images with aspect ratio preservation
- QwenVLEmptyLatent - Creates initial latents for sampling
- QwenVLTextEncoderAdvanced - Advanced version with resolution weighting

## File Location

`nodes/qwen_vl_encoder.py:120-407`
