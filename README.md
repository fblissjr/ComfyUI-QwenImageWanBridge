# ComfyUI Qwen-Image-Edit Nodes

Advanced nodes for Qwen-Image-Edit with multi-image support, more flexibility around the vision transformer (qwen2.5-vl), custom system prompts, and some other experimental things to come.

## Features

### Core Capabilities
- **Qwen-Image-Edit-2509**: Multi-image editing with image token formatting
- **100% DiffSynth-Studio Aligned**: Verified matching implementation
- **Proper Token Dropping**: Matches DiffSynth/Diffusers (drops first 34/64 embeddings)
- **N-Image Support**: Template Builder supports 0-100 images with warnings
- **Clean Architecture**: DRY principles, single source of truth for templates
- **Enhanced Debug**: Full prompt display, character counts, no truncation

### Key Nodes

#### QwenVLTextEncoder
Main text encoder with proper token dropping.
- Token dropping after encoding (34 for text, 64 for image_edit)
- Multi-image support via Image Batch input
- Automatic image token formatting for N images
- System prompt from Template Builder (single source)
- Debug mode shows embedding dropping in action

#### QwenTemplateBuilder
Single source of truth for system prompts.
- Now supports 0-100 images (was limited to 4)
- Automatic warnings at 4+ and 10+ images
- `custom_system`: Override ANY template's system prompt
- Official DiffSynth-Studio templates included
- New edit templates for system prompts

### Helper Nodes
- **QwenVLEmptyLatent**: Creates 16-channel empty latents
- **QwenVLImageToLatent**: Converts images to 16-channel latents
- **QwenTemplateConnector**: Connects template builder to encoder (optional)
- **QwenDebugController**: Comprehensive debugging and profiling system

### Experimental Nodes (Available but Low Priority)
- **QwenSpatialTokenGenerator**: Visual editor for spatial tokens
- **QwenEliGenEntityControl**: Entity-level mask control
- **QwenEliGenMaskPainter**: Simple mask creation
- **QwenTokenDebugger**: Debug token processing
- **QwenTokenAnalyzer**: Analyze tokenization

## Workflows

### Text-to-Image
```
QwenTemplateBuilder → QwenVLTextEncoder → KSampler
                  ↘ (system_prompt)
```

### Single Image Edit
```
LoadImage → QwenVLTextEncoder (edit_image) → KSampler
QwenTemplateBuilder → QwenVLTextEncoder (system_prompt)
```

### Multi-Image Edit (2509)
```
LoadImage (×N) → Image Batch → QwenVLTextEncoder → KSampler
QwenTemplateBuilder → QwenVLTextEncoder (system_prompt)
```

Use prompts like: "Combine the person from Picture 1 with the background from Picture 2"

## Example Workflows

Located in `example_workflows/`:
- **qwen_text_to_image_2509.json** - Basic text-to-image with template
- **qwen_multi_image_2509_simplified.json** - Multi-image editing workflow

Both include comprehensive MarkdownNote documentation.

## Key Settings

### Multi-Image Requirements
- Connect multiple images to **Image Batch** node (from KJNodes pack)
- Reference images with "Picture 1", "Picture 2" in prompts
- Images should ideally have similar dimensions for best results

### Image Ordering
Images are numbered based on connection order to Image Batch:
- `image_1` input → "Picture 1" in prompts
- `image_2` input → "Picture 2" in prompts
- `image_3` input → "Picture 3" in prompts

Example: "Take the old man from Picture 1 and place him in Picture 2"

See [MULTI_IMAGE_ORDERING.md](MULTI_IMAGE_ORDERING.md) for detailed guide.

### Template System
- Connect Template Builder **prompt** output to Encoder **text** input
- Connect Template Builder **system_prompt** output to Encoder **system_prompt** input
- Use `custom_system` field to override any template's system prompt

## Troubleshooting

### System prompt appearing in images?
- Verify Template Builder **system_prompt** output connects to Encoder **system_prompt** input
- Check workflow shows proper connection (should be Link 22 in example workflows)
- Enable debug_mode to see what's being tokenized

### Multi-image not working?
- Use "Picture 1", "Picture 2" references in prompts
- Verify Image Batch node connection to edit_image input
- Enable debug_mode to see Picture formatting

### Dimension mismatch errors?
- Ensure all images have similar dimensions before batching
- Use ComfyUI's Image Scale node if needed to match dimensions

## Status

**Production Ready:**
- Text-to-image generation
- Single and multi-image editing
- Template system with custom overrides
- Dimension-aware processing

**Experimental (Low Priority):**
- Spatial token generation (QwenSpatialTokenGenerator - effectiveness unclear)
- Entity control nodes (QwenEliGenEntityControl - untested with current models)
- Token debugging tools (QwenTokenDebugger, QwenTokenAnalyzer)

**Latest Updates (v2.4):**
- 100% DiffSynth-Studio alignment verified
- Enhanced debug output with full prompts and character counts
- New face replacement templates for full scene preservation
- Templates aligned with Qwen-Image-Edit-2509 training structure

**v2.3 Updates:**
- Debug Controller node for comprehensive debugging and profiling
- Silent debug patches - no console spam unless explicitly enabled
- Performance profiling, memory tracking, and error analysis tools

## Requirements

- ComfyUI
- KJNodes custom node pack (for Image Batch node)
- Model files:
  - `Qwen-Image-Edit-2509.safetensors` (or fp8 variant)
  - `qwen_2.5_vl_7b.safetensors` text encoder
  - `qwen_image_vae.safetensors` (16-channel VAE)

## License

MIT - See LICENSE file for details.
