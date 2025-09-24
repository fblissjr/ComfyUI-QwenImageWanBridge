# ComfyUI Qwen-Image-Edit Nodes

Advanced nodes for Qwen-Image-Edit with multi-image support, more flexibility around the vision transformer (qwen2.5-vl), custom system prompts, and some other experimental things to come.

## Features

### Core Capabilities
- **Qwen-Image-Edit-2509 Support**: Multi-image editing with "Picture X:" references
- **System Prompt Separation**: Clean template system without prompt contamination
- **Multi-Image Processing**: Via ComfyUI's Image Batch node
- **Debug Modes**: Detailed logging for troubleshooting

### Key Nodes

#### QwenVLTextEncoder
Main text encoder with DiffSynth/Diffusers reference alignment.
- 32-pixel resolution alignment for aligned vision processing
- Multi-image support via Image Batch input
- Automatic "Picture X:" formatting for Qwen-Image-Edit-2509
- Separate system_prompt input for template customization
- Debug mode showing tokenization and formatting

#### QwenTemplateBuilder
System prompt generator with clean separation from encoder.
- Outputs raw `prompt` and `system_prompt` separately
- `custom_system`: Override ANY template's system prompt
- `show_all_prompts`: View all available templates
- Uses official DiffSynth-Studio system prompts

### Helper Nodes
- **QwenVLEmptyLatent**: Creates 16-channel empty latents
- **QwenVLImageToLatent**: Converts images to 16-channel latents
- **QwenTemplateConnector**: Connects template builder to encoder (optional)

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

**Recent Fix (v2.1):**
- System prompt separation prevents text contamination in generated images
- Clean workflow connections between Template Builder and Encoder

## Requirements

- ComfyUI
- KJNodes custom node pack (for Image Batch node)
- Model files:
  - `Qwen-Image-Edit-2509.safetensors` (or fp8 variant)
  - `qwen_2.5_vl_7b.safetensors` text encoder
  - `qwen_image_vae.safetensors` (16-channel VAE)

## License

MIT - See LICENSE file for details.
