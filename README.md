*Despite the name, this is becoming more of a Qwen Image Edit repo.*

# ComfyUI Qwen Image Edit Nodes

ComfyUI nodes for Qwen-Image-Edit for precise image editing via a visual spatial editor, system prompt modifications, multi-reference support, and other fun features.

**[Changelog](CHANGELOG.md)**

## Spatial Token Editor & Reference
- New: a visual spatial token editor for bounding boxes, shapes, and point object references, with labels. Expect quirks and issues here, it's EOD and a major work in progress, but it should at least be a start.

Qwen-Image-Edit includes precision control tokens for surgical editing:

- **`<|object_ref_start|>...<|object_ref_end|>`** - Reference specific objects by description
- **`<|box_start|>x1,y1,x2,y2<|box_end|>`** - Target rectangular regions (normalized 0-1 coordinates)
- **`<|quad_start|>x1,y1,x2,y2,x3,y3,x4,y4<|quad_end|>`** - Define complex quadrilateral areas

These bounding box coordinates represent **normalized rectangular regions** in an image, using the format `(x1, y1, x2, y2)` where all values range from 0.0 to 1.0.

**Coordinate system:**
- `(0.0, 0.0)` = top-left corner of the image
- `(1.0, 1.0)` = bottom-right corner of the image
- `x1, y1` = top-left corner of the bounding box
- `x2, y2` = bottom-right corner of the bounding box

**Examples from above**
- `0.2,0.3,0.8,0.7` = rectangle from 20% right, 30% down → 80% right, 70% down (center region)
- `0.0,0.0,0.5,1.0` = rectangle from top-left → 50% width, full height (left half of image)
- `0.1,0.0,0.9,0.4` = rectangle from 10% right, top edge → 90% right, 40% down (upper banner area)

*[Examples and usage patterns →](#spatial-token-examples)*

## Multi-Reference Methods

The `QwenMultiReferenceHandler` offers three composition methods for combining images:

- **concat**: Side-by-side layout for character/object comparison. Best with `common_height` resize for aspect preservation.
- **grid**: 2x2 layout for balanced influence of all references. Uses uniform dimensions to prevent tensor mismatches.
- **offset**: Weighted blending creates single composite. Perfect for style transfer with adjustable influence via weights (e.g., `1.0,1.0,0.5,1.2`).

## Core Nodes

### QwenVLCLIPLoader
Loads Qwen2.5-VL models with RoPE position embedding fixes.
- **Use for:** Loading any Qwen2.5-VL model from `models/text_encoders/`
- **Example:** Text-to-image, image editing workflows

### QwenVLTextEncoder
Main text encoder with vision support and dual image inputs.
- **edit_image:** Vision-processed image for semantic understanding
- **context_image:** ControlNet-style spatial conditioning (no vision processing)
- **Outputs:** conditioning
- **Use for:** Text+image encoding with vision token processing
- **Example:** "Change dress to red" (edit_image) + pose skeleton (context_image)

### QwenMultiReferenceHandler
Combines up to 4 images with aspect ratio preservation to prevent distortion.
- **Methods:** concat (side-by-side), grid (2x2), offset (weighted blend)
- **Resize modes:** match_first, common_height (best for concat), common_width, largest_dims
- **Use for:** Character fusion, style transfer, pose references
- **Example:** Combine Mona Lisa + Shrek + fighting pose → Mona Lisa fighting Shrek in the specified pose

### QwenSpatialTokenGenerator
Interactive spatial editing with visual region drawing interface.
- **"Open Spatial Editor" button:** Visual bounding box/polygon drawing
- **Template integration:** Complete prompt formatting with spatial tokens
- **Multiple input methods:** Upstream workflow images, manual tokens, or visual editor
- **Use for:** Precise region-based editing with spatial token generation
- **Example:** Draw box around church → "Replace with castle" → generates full spatial prompt

### QwenLowresFixNode
Two-stage refinement: generate → upscale → polish details.
- **Use for:** Quality enhancement of generated images
- **Example:** 1024px generation → 1.5x upscale = detail refinement

## Helper Nodes

### QwenVLEmptyLatent / QwenVLImageToLatent
Creates/converts 16-channel latents for Qwen model.
- **Use for:** Latent space operations

## Template System

### QwenTemplateBuilder
Interactive template builder with style presets.
- **Presets:** DiffSynth, artistic, photorealistic, minimal edit, action, etc.
- **Use for:** Custom system prompts and specialized generation styles

## Usage Examples

### Text-to-Image
```
QwenVLCLIPLoader → QwenVLTextEncoder (mode: text_to_image) → KSampler
```

### Image Editing
```
LoadImage → QwenVLTextEncoder (edit_image, mode: image_edit) → KSampler
```

### Multi-Reference Composition
```
LoadImage(3x) → QwenMultiReferenceHandler → QwenVLTextEncoder → KSampler
```

### ControlNet-Style Conditioning
```
LoadImage(edit) + LoadImage(control) → QwenVLTextEncoder (edit_image + context_image) → KSampler
```

### Spatial Token Editing (NEW)
```
LoadImage → QwenSpatialTokenGenerator (spatial editor) → QwenVLTextEncoder → KSampler
```

### Quality Enhancement
```
KSampler → QwenLowresFixNode → Final Image
```

## Advanced Usage

**Spatial Reference Tokens** (Phase 3 roadmap):
```
"Change <|object_ref_start|>the red car<|object_ref_end|> to blue"
"Replace <|box_start|>0.2,0.3,0.8,0.7<|box_end|> with flowers"
```

**Denoise Guidelines:**
- High denoise (0.9-1.0): Full creative reimagining
- Low denoise (0.3-0.7): Structure preservation
- Use Empty Latent for high denoise, VAE Encode for low denoise

## Spatial Token Examples

### Object Reference - Independent Usage
Target specific objects by description:
```
"Change <|object_ref_start|>the red car<|object_ref_end|> to blue while keeping everything else the same"
"Make <|object_ref_start|>the woman's dress<|object_ref_end|> flowing in the wind"
"Replace <|object_ref_start|>the old wooden door<|object_ref_end|> with a modern glass entrance"
```

### Box Coordinates - Precise Region Control
Target rectangular areas with normalized coordinates (0-1):
```
"Fill <|box_start|>0.2,0.3,0.8,0.7<|box_end|> with blooming cherry blossoms"
"Replace the content in <|box_start|>0.0,0.0,0.5,1.0<|box_end|> with a mountain landscape"
"Add storm clouds to <|box_start|>0.1,0.0,0.9,0.4<|box_end|>"
```

### Quadrilateral - Complex Shape Targeting
Define irregular areas with four corner points:
```
"Paint graffiti art within <|quad_start|>0.1,0.2,0.6,0.1,0.8,0.7,0.2,0.8<|quad_end|>"
"Transform <|quad_start|>0.3,0.1,0.9,0.2,0.7,0.9,0.1,0.6<|quad_end|> into stained glass"
```

### Combined Usage - Multiple Token Types
Mix different tokens for complex edits:
```
"Change <|object_ref_start|>the building<|object_ref_end|> in <|box_start|>0.4,0.2,1.0,0.8<|box_end|> to Gothic architecture"
"Make <|object_ref_start|>the tree<|object_ref_end|> seasonal: spring leaves in <|quad_start|>0.2,0.1,0.8,0.0,0.9,0.6,0.1,0.7<|quad_end|>"
"Replace <|object_ref_start|>the sky<|object_ref_end|> above <|box_start|>0.0,0.0,1.0,0.4<|box_end|> with aurora borealis"
```

### Multi-Object Coordination
Target multiple objects with spatial relationships:
```
"Change <|object_ref_start|>the left car<|object_ref_end|> to red and <|object_ref_start|>the right car<|object_ref_end|> to blue"
"Make <|object_ref_start|>the foreground flowers<|object_ref_end|> roses and <|object_ref_start|>the background trees<|object_ref_end|> cherry blossoms"
```
