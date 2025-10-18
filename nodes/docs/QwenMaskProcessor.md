# QwenMaskProcessor

**Category:** Qwen/Mask
**Display Name:** Qwen Mask Processor

**STATUS: EXPERIMENTAL - NOT FULLY TESTED**
Inpainting workflow is experimental and may not work as expected. Consider using standard ComfyUI inpainting nodes instead.

## Description

Processes spatial editor output into inpainting-ready masks for post-processing blend workflow. Provides preprocessing options for blur, expansion, and feathering.

**Part of Simple Blending Approach:**
- Prepares masks for post-processing latent blend
- Not the DiffSynth EliGen attention masking approach
- Works with existing qwen-image-edit model
- No DiT modifications required

## Important Changes (v2.6)

- **No longer outputs prompt** - Use QwenVLTextEncoder or QwenTemplateBuilder for prompts
- Outputs: IMAGE, MASK, preview, mask_preview (prompt output removed)
- Part of simple blending workflow (not EliGen attention masking)

## Inputs

### Required
- **image** (IMAGE)
  - Source image for editing
  - Can connect directly from LoadImage

- **mask_data** (STRING)
  - Base64 encoded mask from spatial editor
  - Or empty string if using mask_override
  - JavaScript interface auto-generates this

- **mask_blur** (FLOAT)
  - Gaussian blur for mask edges
  - Range: 0.0-20.0
  - Default: 2.0
  - Higher values = softer edges

- **mask_expand** (INT)
  - Expand (+) or contract (-) mask pixels
  - Range: -50 to +50
  - Default: 0
  - Positive: Grow mask region
  - Negative: Shrink mask region

- **mask_feather** (BOOLEAN)
  - Apply edge feathering for smoother blending
  - Default: True
  - Creates gradual transition at mask edges

### Optional
- **mask_override** (MASK)
  - Override generated mask with manual mask
  - Bypasses mask_data processing

## Outputs

- **image** (IMAGE)
  - Pass-through source image (unchanged)

- **mask** (MASK)
  - Processed binary mask tensor
  - Resized to match image dimensions
  - Grayscale, normalized 0-1

- **preview** (IMAGE)
  - Visual preview showing inpainting areas in red
  - Red overlay on source image where mask=white

- **mask_preview** (IMAGE)
  - Mask visualization (white=inpaint, black=preserve)

## Processing Pipeline

1. **Input**: Base64 mask data or mask override
2. **Decode**: Convert base64 to PIL Image
3. **Resize**: Match target image dimensions
4. **Grayscale**: Convert to single channel
5. **Expand/Contract**: Morphological operations if mask_expand ≠ 0
6. **Blur**: Gaussian blur if mask_blur > 0
7. **Feather**: Edge feathering if mask_feather=True
8. **Convert**: PIL → torch.Tensor (normalized 0-1)

## Mask Processing Details

### Expand/Contract (Morphological Operations)
```python
if mask_expand > 0:
    # Dilate: Grow mask region
    mask = mask.filter(ImageFilter.MaxFilter(abs(mask_expand)))
elif mask_expand < 0:
    # Erode: Shrink mask region
    mask = mask.filter(ImageFilter.MinFilter(abs(mask_expand)))
```

### Gaussian Blur
```python
if mask_blur > 0:
    mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))
```

### Edge Feathering
```python
if mask_feather:
    # Apply gradient feathering to edges
    # Creates smooth transition zone
```

## Example Usage

### Basic Mask Processing
```
LoadImage → QwenMaskProcessor (mask_data from JS, blur=2.0)
              ↓ (image, mask)
         QwenVLTextEncoder (mode: inpainting)
```

### Manual Mask Override
```
LoadImage → QwenMaskProcessor (mask_override connected, ignores mask_data)
              ↓ (mask)
         QwenInpaintSampler
```

### With Mask Refinement
```
LoadImage → QwenMaskProcessor (
              mask_blur: 5.0,      # Soft edges
              mask_expand: 10,     # Grow 10px
              mask_feather: true   # Smooth blend
            )
              ↓
         Preview shows refined mask
```

## JavaScript Interface

The spatial mask interface provides:
- Visual mask editor with canvas
- Drawing modes: bounding box, polygon, object reference
- Auto-generation of base64 mask_data
- Real-time preview
- "Generate Mask & Sync" button sends to node

Access via button on QwenMaskProcessor node (if UI available).

## Workflow Integration

### Recommended Flow
```
LoadImage → QwenMaskProcessor
              ↓ (image)    ↓ (mask)
         QwenVLTextEncoder (mode: inpainting, mask auto-resized to VAE dims)
              ↓ (conditioning with mask)
         QwenInpaintSampler
              ↓
         VAEDecode
```

### Prompt Handling (Important)
- QwenMaskProcessor does NOT output prompts (changed in v2.6)
- Enter prompt in QwenVLTextEncoder text field
- Or use QwenTemplateBuilder → QwenVLTextEncoder

## Diffusers Compatibility

Follows exact preprocessing from `QwenImageEditInpaintPipeline`:
- Binary mask (0=preserve, 1=inpaint)
- Properly sized to match image
- Preprocessed with blur/expand/feather options
- Ready for latent space operations

## Related Nodes

- QwenVLTextEncoder - Receives mask input (inpainting mode)
- QwenInpaintSampler - Uses mask for blending
- LoadImage - Provides source image
- QwenSpatialMaskInterface (JS) - Visual mask editor

## File Location

`nodes/qwen_mask_processor.py`

## Changes from Previous Version

- Removed `inpaint_prompt` input parameter
- Removed prompt from RETURN_TYPES
- Returns: `(IMAGE, MASK, IMAGE, IMAGE)` instead of `(IMAGE, MASK, STRING, IMAGE, IMAGE)`
- Prompts now handled by text encoder or template builder
