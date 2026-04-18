# QwenTemplateBuilder

**Category:** QwenImage/Templates
**Display Name:** Qwen Template Builder

## Description

Builds DiffSynth-compatible system prompts for Qwen2.5-VL encoding. Provides 22 pre-configured templates and supports custom system prompt overrides.

**IMPORTANT:** Always connect BOTH `mode` and `system_prompt` outputs to the encoder. The `mode` output ensures vision tokens are formatted correctly for the selected template.

## Inputs

### Required
- **prompt** (STRING, multiline)
  - Your main prompt text
  - Default: "" (empty)

- **template_mode** (ENUM)
  - `default_t2i` - Standard text-to-image generation
  - `default_edit` - Standard image editing
  - `multi_image_edit` - Multiple image editing with Picture format (DiffSynth `encode_prompt_edit_multi`)
  - `face_replacement` - Face swap operations
  - `face_replacement_detailed` - Detailed face replacement
  - `face_replacement_technical` - Technical face replacement
  - `identity_transfer` - Identity transfer operations
  - `qwen_face_swap` - Qwen-specific face swap
  - `qwen_identity_merge` - Qwen identity merging
  - `structured_json_edit` - JSON-based editing
  - `xml_spatial_edit` - XML spatial editing
  - `natural_spatial_edit` - Natural language spatial
  - `artistic` - Artistic interpretation
  - `photorealistic` - Photorealistic generation
  - `minimal_edit` - Minimal changes only
  - `style_transfer` - Style transformation
  - `technical` - Technical/precise generation
  - `custom_t2i` - Custom text-to-image
  - `custom_edit` - Custom editing
  - `raw` - No template formatting
  - `show_all_prompts` - Display all available templates
  - Default: `default_edit`

- **custom_system** (STRING, multiline)
  - Override system prompt for ANY template mode
  - JavaScript UI auto-fills this when you select a preset
  - Edit the auto-filled text to customize it
  - Leave empty to use template default
  - Default: ""

## Outputs

- **prompt** (STRING)
  - User prompt text (passed through)

- **system_prompt** (STRING)
  - Formatted system prompt for encoder
  - If custom_system provided: uses custom_system
  - Otherwise: uses template default
  - For show_all_prompts: returns all template info

- **mode** (STRING)
  - Mode for encoder (text_to_image, image_edit, multi_image_edit, or inpainting)
  - **REQUIRED CONNECTION:** Connect to encoder's `mode` input for auto-sync
  - Ensures correct vision token formatting (labels, placement, token dropping)
  - Without this connection, you risk mismatched vision token formatting

- **mode_info** (STRING)
  - Mode metadata description

## Template Examples

### default_t2i
```
System: "Describe the image by detailing the color, shape, size, texture,
quantity, text, spatial relationships of the objects and background:"
Mode: text_to_image
```

### default_edit
```
System: "Describe the key features of the input image (color, shape, size,
texture, objects, background), then explain how the user's text instruction
should alter or modify the image. Generate a new image that meets the user's
requirements while maintaining consistency with the original input where appropriate."
Mode: image_edit
```

### multi_image_edit
```
System: "Describe the key features of each input image (color, shape, size,
texture, objects, background), identify which image each element comes from,
then explain how to combine or modify these images according to the user's
instruction."
Mode: multi_image_edit
Vision token placement: Inside prompt with "Picture X:" labels
```

### artistic
```
System: "You are an experimental artist. Break conventions. Be bold and creative.
Interpret the prompt with artistic freedom."
Mode: text_to_image
```

### minimal_edit
```
System: "Make only the specific changes requested. Preserve all other aspects
of the original image exactly."
Mode: image_edit
```

## DiffSynth Template Format

When used with QwenVLTextEncoder and system_prompt provided:
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{vision_tokens}{text}<|im_end|>
<|im_start|>assistant
```

This format enables proper token dropping in the encoder:
- text_to_image: Drop first 34 tokens
- image_edit: Drop first 64 tokens
- multi_image_edit: Drop first 64 tokens

## Example Usage

### Correct Usage (ALWAYS USE THIS)
```
QwenTemplateBuilder (template_mode: default_edit, prompt: "make the sky blue")
  ├─ (system_prompt) ──> QwenVLTextEncoder (system_prompt input)
  └─ (mode) ──────────> QwenVLTextEncoder (mode input)
                        Both connections required!
```

### Why Both Connections Matter
```
Template: multi_image_edit
  ├─ system_prompt: "Describe features of each input image..."
  └─ mode: "multi_image_edit"
       ↓
    Encoder uses mode to:
    - Place vision tokens INSIDE prompt (not before)
    - Add "Picture X:" labels
    - Apply correct token dropping (64 tokens)

Without mode connection:
  - Encoder uses default mode (image_edit)
  - Wrong token placement = broken generation
```

### Custom System Override
```
QwenTemplateBuilder (
  template_mode: default_edit,
  custom_system: "Be creative but preserve composition",
  prompt: "artistic style"
)
  ↓ (system_prompt with custom override)
QwenVLTextEncoder
```

### Show All Templates
```
QwenTemplateBuilder (template_mode: show_all_prompts)
  → Outputs detailed info for all 15+ templates
```

## Template Categories

**Generation:**
- default_t2i, artistic, photorealistic, technical

**Editing:**
- default_edit, minimal_edit, style_transfer, multi_image_edit

**Face Operations:**
- face_replacement, face_replacement_detailed, face_replacement_technical
- identity_transfer, qwen_face_swap, qwen_identity_merge

**Structured:**
- structured_json_edit, xml_spatial_edit, natural_spatial_edit

**Custom:**
- custom_t2i, custom_edit, raw

## Related Nodes

- QwenVLTextEncoder - Receives system_prompt and mode outputs
- QwenVLTextEncoderAdvanced - Advanced encoder also supports mode auto-sync
- QwenProcessorV2 - Formats template and applies token dropping
- QwenImageBatch - Provides batched images for multi_image_edit mode

## File Location

`nodes/qwen_template_builder.py`
