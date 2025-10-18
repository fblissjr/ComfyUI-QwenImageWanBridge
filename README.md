# ComfyUI Qwen-Image-Edit Nodes

Custom nodes for Qwen-Image-Edit with multi-image support, more flexibility around the vision transformer (qwen2.5-vl), custom system prompts, and some other experimental things to come.

## BREAKING CHANGE (v2.7.0+)

**Existing workflows will break.** You must:
1. Delete and re-add, or Right Click -> Recreate Template Builder, Encoder, and Encoder Advanced nodes from your workflow
2. Connect: Template Builder `template_output` → Encoder `template_output` (single connection only)

Old multi-connection system (mode + system_prompt) no longer works as it was getting convoluted and confusing. One connection from template builder now handles everything (prompt, mode, system_prompt).

**Updated Workflows Available** [here](example_workflows/nunchaku_qwen_image_edit_2509.json)
1. [Multi image edit workflow](example_workflows/nunchaku_qwen_image_edit_2509.json)
2. [Single image edit workflow](example_workflows/qwen_edit_2509_single_image_edit.json)

---

## Required Connections

**When using Template Builder:**
- Connect: Template Builder `template_output` → Encoder `template_output`
- That's it. One connection handles everything (prompt, mode, system_prompt)

**Without Template Builder:**
- Use encoder's dropdown for mode selection
- Type text and system_prompt manually

---

**Documentation:**
- [CHANGELOG.md](CHANGELOG.md) - Full changelog history
- [nodes/docs/](nodes/docs/) - Detailed node documentation
- [nodes/docs/resolution_tradeoffs.md](nodes/docs/resolution_tradeoffs.md) - Resolution and scaling guide
- [nodes/docs/QwenImageBatch.md](nodes/docs/QwenImageBatch.md) - Batch node documentation

## Features

### Core Capabilities
- **Qwen-Image-Edit-2509**: Multi-image editing (1-3 optimal, up to 10 max because I had to pick something)
- **QwenImageBatch**: Smart batching with auto-detection, aspect ratio preservation, scaling, batching strategy
- **Resolution Control & Power User Mode**: Per-image resolution control
- **Template Builder Auto-Sync**: Automatic mode matching between template and encoder
- **System Prompt Control**: Customizable system prompts via the template builder
- **Automatic Double-Scaling Prevention**: Batch node and better encoder intelligence
- **Full Debug Output**: Complete prompts, character counts, aspect ratio tracking

### Still Experimental or Not Working Well (or at all)
- Mask-Based Inpainting: Selective editing with diffusers blending, including Eligen entity control
- All wrapper nodes, probably more stuff I'm omitting

### v2.7.0 Highlights

**File-Based Template System** - Single source of truth
- 9 templates in `nodes/templates/*.md` files (was 22 hardcoded)
- YAML frontmatter for metadata (mode, vision, experimental flags)
- Easy to add/edit without code changes
- JavaScript UI auto-fills `custom_system` field for editing

**Template Builder → Encoder Connection**
- `mode` is now STRING input (accepts connections or manual typing)
- **REQUIRED**: Connect BOTH `mode` and `system_prompt` to encoder
- Missing either = broken vision token formatting

**QwenImageBatch** - No more KJNodes dependency ([docs](nodes/docs/QwenImageBatch.md))
- Auto-detects up to 10 images (no inputcount parameter)
- Skips empty inputs (no black images)
- `max_dimensions` strategy (minimal distortion) or `first_image` (hero-driven)
- Prevents double-scaling with metadata propagation
- See [resolution_tradeoffs.md](nodes/docs/resolution_tradeoffs.md) for detailed scaling guide

### Nodes

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
- **scaling_mode** - Same three modes as standard encoder (preserve/max_dimension/area)
- **resolution_mode** - Applies weights to scaling_mode base (balanced/hero_first/hero_last/progressive)
- Per-image resolution weighting
- Memory budget management (max_memory_mb)
- Hero/reference modes for importance
- Custom resolution targets (vision & VAE separate)
- Choice of "Picture" vs "Image" labels
- Simplified interface (no validation_mode)
- verbose_log for console tracing of model passes

#### QwenTemplateBuilder
File-based system prompt templates (9 templates).
- Templates loaded from `nodes/templates/*.md` files
- YAML frontmatter for metadata
- JavaScript UI auto-fills `custom_system` field
- **REQUIRED**: Connect BOTH `mode` and `system_prompt` to encoder
- Available templates: `default_t2i`, `default_edit`, `multi_image_edit`, `artistic`, `photorealistic`, `minimal_edit`, `technical`, `inpainting`, `raw`

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


## Example Workflows

Located in `example_workflows/`

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
- **IMPORTANT**: Always connect BOTH outputs from Template Builder to encoder:
  - Template Builder **mode** → Encoder **mode** (ensures correct vision token formatting)
  - Template Builder **system_prompt** → Encoder **system_prompt** (provides instruction text)
- Template Builder **prompt** → Encoder **text** (your actual prompt text)
- Use `custom_system` field to edit any template's system prompt before using it
- JavaScript UI auto-fills `custom_system` when you select a preset

**Why connect both?**
- `mode` controls how vision tokens are formatted (labels, placement, token dropping)
- `system_prompt` provides the instruction text for the model
- Using mismatched mode/system_prompt causes incorrect token formatting
- Example: `inpainting` system prompt with `image_edit` mode = broken workflow

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

**Working/Tested:**
- Text-to-image generation
- Single and multi-image editing
- Resolution control
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
