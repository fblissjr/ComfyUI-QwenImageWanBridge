# ComfyUI Qwen-Image-Edit Nodes

Custom nodes for Qwen-Image-Edit with multi-image support, more flexibility around the vision transformer (qwen2.5-vl), custom system prompts, and some other experimental things to come.

See [CHANGELOG.md](CHANGELOG.md) for full details and changelog history.

## Features

### Core Capabilities
- **Qwen-Image-Edit-2509**: Multi-image editing (1-3 optimal, up to 512 max)
- **Mask-Based Inpainting**: Selective editing with diffusers blending
- **100% DiffSynth-Studio Aligned**: Verified implementation
- **Advanced Power User Mode**: Per-image resolution control
- **Configurable Auto-Labeling**: Optional "Picture X:" formatting
- **Memory Optimization**: VRAM budgets and weighted resolution
- **Full Debug Output**: Complete prompts, character counts, memory usage

### Key Features

#### Automatic Resolution Handling
- Automatically handles mismatched dimensions between empty latent and reference images
- Pads to nearest even dimensions for model compatibility
- Works with any aspect ratio - not limited to 1024x1024

### Key Nodes

#### QwenVLTextEncoder
Standard encoder with automatic labeling.
- Token dropping: 34 (text), 64 (image_edit)
- Multi-image support via Image Batch
- auto_label parameter for "Picture X:" control
- System prompt from Template Builder
- Full debug output with character counts
- verbose_log for console tracing of model passes

**Resolution Scaling Modes:**
- `preserve_resolution` (default): Keeps original size with 32px alignment
  - Best quality, no zoom-out effect
  - Recommended for typical images (512px-2048px)
  - May use more VRAM on very large images
- `max_dimension_1024`: Scales largest side to 1024px
  - Reduces VRAM on large images
  - Good for 4K images or VRAM constraints
  - Some zoom-out on large images
- `area_1024`: Scales to ~1024×1024 area (legacy)
  - Consistent output size but poor scaling behavior
  - Not recommended for general use

#### QwenVLTextEncoderAdvanced
Power user encoder with resolution control.
- All standard features plus:
- Per-image resolution weighting
- Memory budget management (max_memory_mb)
- Hero/reference modes for importance
- Custom resolution targets (vision & VAE separate)
- Choice of "Picture" vs "Image" labels
- Simplified interface (no validation_mode)
- verbose_log for console tracing of model passes

#### QwenTemplateBuilder
System prompt templates.
- DiffSynth-Studio templates included
- custom_system override for any template
- show_all_prompts mode to view options

### Helper Nodes
- **QwenVLEmptyLatent**: Creates 16-channel empty latents
- **QwenVLImageToLatent**: Converts images to 16-channel latents
- **QwenTemplateConnector**: Connects template builder to encoder (optional)
- **QwenDebugController**: Comprehensive debugging and profiling system

### Inpainting Nodes
- **QwenMaskProcessor**: Mask preprocessing with blur, expand, feather controls
- **QwenInpaintSampler**: Diffusers-pattern inpainting with strength control

### Experimental Nodes (Available but Low Priority)
- **QwenSpatialTokenGenerator**: Visual editor for spatial tokens that don't seem to do much of anything right now
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

### Mask-Based Inpainting
```
LoadImage → QwenMaskProcessor → QwenVLTextEncoder (inpainting mode)
                    ↓ (mask)              ↓
                    ↓             QwenInpaintSampler → VAEDecode
                    ↓                      ↑
                    └──────────────────────┘
```

Selective editing with mask controls (blur, expand, feather)

## Example Workflows

Located in `example_workflows/`:
- **qwen_text_to_image_2509.json** - Basic text-to-image with template
- **qwen_multi_image_2509_simplified.json** - Multi-image editing workflow
- **qwen_edit_2509_mask_inpainting.json** - Mask-based inpainting workflow
- **nunchaku_qwen_mask_inpainting.json** - Nunchaku variant for inpainting

All include comprehensive documentation and notes.

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

### Subjects appearing zoomed out or too small?
**Problem:** Characters or objects in edited images appear smaller than expected.

**Solution:** Use `preserve_resolution` scaling mode (default in v2.6.1+)
- The old `area_1024` mode aggressively downscaled large images
- Example: 1477×2056 was scaled to 864×1216 (0.58x), causing zoom-out
- New default preserves original size: 1477×2056 → 1472×2048 (1.00x)

**Alternative:** If hitting VRAM limits, try `max_dimension_1024` mode instead

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

### Out of memory (OOM) errors with large images?
- Switch to `max_dimension_1024` scaling mode
- This reduces VRAM usage while maintaining reasonable quality
- For 4K images: max_dimension_1024 is recommended over preserve_resolution

## Status

**Production Ready:**
- Text-to-image generation
- Single and multi-image editing
- Mask-based inpainting with diffusers blending
- Template system with custom overrides
- Dimension-aware processing

**Experimental (Low Priority):**
- Spatial token generation (QwenSpatialTokenGenerator - effectiveness unclear)
- Entity control nodes (QwenEliGenEntityControl - untested with current models)
- Token debugging tools (QwenTokenDebugger, QwenTokenAnalyzer)

**Latest Updates:**
- **v2.6.1**: Resolution scaling fix with three modes (preserve/max_dimension/area)
- **v2.6**: Mask-based inpainting system with QwenMaskProcessor and QwenInpaintSampler
- **v2.5**: Advanced encoder for power users with resolution control
- Configurable auto-labeling (can now be disabled)
- Memory optimization for limited VRAM
- Hero/reference image weighting

## License

MIT - See LICENSE file for details.
