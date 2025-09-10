*Despite the name, this is becoming more of an experimental Qwen Image Edit repo, though I'm still bullish on Qwen Image and WAN video playing a big role together for video generation (and video/keyframe editing)*

# Experimental/Research ComfyUI Qwen Image Edit Nodes (and eventually + Wan)

This is an **EXPERIMENTAL** research repo with custom nodes for Qwen-Image-Edit for precise image editing via a visual spatial editor, system prompt modifications, multi-reference support, and other fun features. Expect this code to change often, and expect many things to just not work well at all. That said - expect transparency and hopefully some ideas others can use to form new ideas and directions.

**[Changelog](CHANGELOG.md)**

## Table of Contents

### Core Nodes
- **[QwenVLCLIPLoader](#qwenvlcliploader)** - Load Qwen2.5-VL models with fixes
- **[QwenVLTextEncoder](#qwenvltextencoder)** - Main text encoder with vision support
- **[QwenSpatialTokenGenerator](#qwenspatialtonkengenerator)** - Interactive spatial editing with visual region drawing *WIP*
- **[QwenMultiReferenceHandler](#qwenmultireferencehandler)** - Combine up to 4 images with aspect ratio preservation
- **[QwenLowresFixNode](#qwenlowresfixnode)** - Two-stage refinement for quality enhancement

### Helper Nodes
- **[QwenVLEmptyLatent / QwenVLImageToLatent](#qwenvlemptylatent--qwenvlimagetolatent)** - 16-channel latent operations
- **[QwenTemplateBuilder](#qwentemplatebuilder)** - Interactive template builder with style presets

### Quick Start Examples
- [Text-to-Image](#text-to-image) | [Image Editing](#image-editing) | [Multi-Reference](#multi-reference-composition) | [Spatial Tokens](#spatial-token-editing-new) | [Quality Enhancement](#quality-enhancement)

---

## Spatial Token Editor & Reference (Experimental - unknown if better than natural language)

**Note:** This is experimental. These tokens exist in Qwen2.5-VL's tokenizer but aren't documented. Whether they work better than natural language descriptions is unknown - this is for testing and experimentation.

**What it does:** Takes an image and generates structured spatial prompts with coordinates filled in automatically. You can (and should) edit the generated output to fine-tune your edits.

**Output Formats (v1.6.0):**
- **Structured JSON** (new default): Command objects with action, target, coordinates, and preservation instructions
- **XML Tags**: HTML-like elements with data-bbox attributes
- **Natural Language**: Coordinate-aware sentences
- **Traditional Tokens**: Legacy spatial token format

**Traditional Spatial Tokens:**
- **`<|object_ref_start|>...<|object_ref_end|>`** - Reference specific objects by description
- **`<|box_start|>x1,y1,x2,y2<|box_end|>`** - Target rectangular regions (absolute pixel coordinates)
- **`<|quad_start|>x1,y1,x2,y2,x3,y3,x4,y4<|quad_end|>`** - Define complex quadrilateral areas

These bounding box coordinates represent **absolute pixel positions** in an image, using the format `(x1, y1, x2, y2)` where values are actual pixel coordinates.

**Coordinate system:**
- `(0, 0)` = top-left corner of the image
- `(width, height)` = bottom-right corner of the image
- `x1, y1` = top-left corner of the bounding box (in pixels)
- `x2, y2` = bottom-right corner of the bounding box (in pixels)

**Examples for a 1024x1024 image:**
- `200,300,800,700` = rectangle from pixel (200,300) → (800,700) (center region)
- `0,0,512,1024` = rectangle from top-left → 512 pixels right, full height (left half)
- `100,0,900,400` = rectangle from (100,0) → (900,400) (upper banner area)

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
Main text encoder with vision support, dual image inputs, and dual-encoding architecture.

**Dual-Encoding Architecture (v1.4.4+):**
- **Semantic Path:** edit_image → QwenVisionProcessor → MultiFrameVisionEmbedder → high-level understanding
- **Reconstructive Path:** edit_image → VAE → structural features → low-level detail preservation
- **Fusion:** Both paths combined in MMDiT-compatible conditioning for balanced semantic coherence + visual fidelity

**Inputs:**
- **edit_image:** Vision-processed image for semantic understanding (activates dual-encoding when VAE connected)
- **context_image:** ControlNet-style spatial conditioning (no vision processing)
- **vae:** Optional VAE connection enables dual-encoding architecture

**Outputs:** conditioning with dual-encoding metadata
- **Use for:** Text+image encoding with advanced vision processing
- **Example:** "Change dress to red" (edit_image) + pose skeleton (context_image) with automatic dual-encoding when VAE connected

### QwenMultiReferenceHandler
Combines up to 4 images with aspect ratio preservation to prevent distortion.
- **Methods:** concat (side-by-side), grid (2x2), offset (weighted blend)
- **Resize modes:** match_first, common_height (best for concat), common_width, largest_dims
- **Use for:** Character fusion, style transfer, pose references
- **Example:** Combine Mona Lisa + Shrek + fighting pose → Mona Lisa fighting Shrek in the specified pose

### QwenSpatialTokenGenerator
Experimental spatial editing with visual region drawing interface.
- **Required image input:** Direct image connection with built-in resolution optimization
- **"Open Spatial Editor" button:** Canvas-based drawing for bounding boxes, polygons, and object reference points
- **Output format selection:** Choose between structured JSON (default), XML tags, natural language, or traditional tokens
- **Individual region management:** Delete buttons and inline label editing for each created region
- **Experimental token generation:** Generates different prompt formats - effectiveness compared to natural language unknown
- **Editable output:** Generated prompts appear in base_prompt field for user modification
- **Dual output:** Plain prompt (base_prompt contents) + formatted_prompt (template applied)
- **Integrated resolution optimization:** Built-in QwenOptimalResolution logic eliminates separate preprocessing
- **Debug mode:** Detailed coordinate and generation logging
- **Use for:** Testing structured spatial prompts vs natural language descriptions
- **Example:** Connect image → draw box around church → "Replace with castle" → generates optimized image + spatial tokens

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

**With Dual-Encoding (v1.4.4+):**
```
LoadImage → VAEEncode → QwenVLTextEncoder (edit_image + vae, mode: image_edit) → KSampler
```
*Automatically activates semantic + reconstructive dual encoding for improved editing quality*

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
LoadImage → QwenSpatialTokenGenerator (required image input, spatial editor) → QwenVLTextEncoder → KSampler
```

### Quality Enhancement
```
KSampler → QwenLowresFixNode → Final Image
```

## Advanced Usage

**Spatial Reference Tokens:**
```
"Change <|object_ref_start|>the red car<|object_ref_end|> to blue"
"Replace <|box_start|>200,300,800,700<|box_end|> with flowers"
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
Target rectangular areas with absolute pixel coordinates:
```
"Fill <|box_start|>200,300,800,700<|box_end|> with blooming cherry blossoms"
"Replace the content in <|box_start|>0,0,512,1024<|box_end|> with a mountain landscape"
"Add storm clouds to <|box_start|>100,0,900,400<|box_end|>"
```

### Quadrilateral - Complex Shape Targeting
Define irregular areas with four corner points (absolute pixel coordinates):
```
"Paint graffiti art within <|quad_start|>100,200,600,100,800,700,200,800<|quad_end|>"
"Transform <|quad_start|>300,100,900,200,700,900,100,600<|quad_end|> into stained glass"
```

### Combined Usage - Multiple Token Types
Mix different tokens for complex edits (absolute pixel coordinates):
```
"Change <|object_ref_start|>the building<|object_ref_end|> in <|box_start|>400,200,1024,800<|box_end|> to Gothic architecture"
"Make <|object_ref_start|>the tree<|object_ref_end|> seasonal: spring leaves in <|quad_start|>200,100,800,0,900,600,100,700<|quad_end|>"
"Replace <|object_ref_start|>the sky<|object_ref_end|> above <|box_start|>0,0,1024,400<|box_end|> with aurora borealis"
```

### Multi-Object Coordination
Target multiple objects with spatial relationships:
```
"Change <|object_ref_start|>the left car<|object_ref_end|> to red and <|object_ref_start|>the right car<|object_ref_end|> to blue"
"Make <|object_ref_start|>the foreground flowers<|object_ref_end|> roses and <|object_ref_start|>the background trees<|object_ref_end|> cherry blossoms"
```
