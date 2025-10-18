# QwenVLCLIPLoader

**Category:** QwenImage/Loaders
**Display Name:** Qwen2.5-VL CLIP Loader

## Description

Loads Qwen2.5-VL model using ComfyUI's internal CLIP loader with `CLIPType.QWEN_IMAGE`. This ensures compatibility with the diffusion pipeline and standard ComfyUI workflows.

## Inputs

### Required
- **model_name** (STRING)
  - Qwen2.5-VL model from `ComfyUI/models/text_encoders/` directory
  - Filter shows models with "qwen" in filename
  - Default fallback: `qwen_2.5_vl_7b.safetensors`

## Outputs

- **clip** (CLIP)
  - Loaded Qwen2.5-VL model ready for text/vision encoding

## Implementation Details

- Uses `comfy.sd.load_clip()` with `clip_type=comfy.sd.CLIPType.QWEN_IMAGE`
- Model path: `models/text_encoders/` (ComfyUI standard location)
- Embedding directory: Uses ComfyUI's embedding folder paths

## Example Usage

```
QwenVLCLIPLoader (model: qwen_2.5_vl_7b.safetensors)
  ↓ (CLIP)
QwenVLTextEncoder
```

## Related Nodes

- QwenVLTextEncoder - Uses CLIP output for encoding
- QwenTemplateBuilder - Provides system prompts for encoding

## File Location

`nodes/qwen_vl_encoder.py:69-117`
