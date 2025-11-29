# ComfyUI Nodes for Qwen's LLM related models

First and foremost, this is a research repo and sandbox. While I've straddled between both worlds of image / video models and LLMs since the early days, I tend to be more comfortable on the LLM. With natively trained multimodal LLMs (see Gemini 3 and Nano Banana Pro) in commercial models, and open source models now using more common LLMs and vision LLMs like Qwen3 and Qwen2.5-VL, I started this repo to see if I could lend my background to the space. Despite the worlds of DiT models and LLM models converging into a single system, many people on both sides of the coin tend to not know what the other is doing (feels like a backend/frontend paradigm). The goal of this repo is to explore DiT models that leverage modern autoregressive, vision LLMs and find fun ways to use them in new (or more optimal) ways.

There will be breaking changes, and this isn't meant to be a prod repo. If you find a version that works for you and is stable, I'd recommend pinning that version and not updating unless you find a good reason to. If you're more interested in tinkering and research, then by all means, join the party. This is NOT optimized for a 24/7 production environment.

Yes, I know that doesn't include Wan yet, but I think eventually it will. Qwen Image & Qwen Image Edit are the most built out and most useful. HunyuanVideo 1.5 was added to experiment mostly with system prompts and other things. 

## So what's this repo for then?

Custom nodes for :
 - Z-Image - Experimental text encoding with system prompts, thinking blocks, assistant prompts, and turn builders, along with templates
 - Qwen-Image-Edit with multi-image support, more flexibility around the vision transformer (qwen2.5-vl), custom system prompts, and some other experimental things
 - HunyuanVideo 1.5 Text-to-Video - Custom system prompts, experiments with attention, and other random experiments


### Z-Image Text Encoder

Z-Image uses Qwen3-4B as its text encoder. Our nodes follow the exact Qwen3-4B chat template format from `tokenizer_config.json`.

> **"What does this actually do?"** - See [Z-Image Intro Guide](nodes/docs/z_image_intro.md) for a quick overview with examples.

**Example Workflows:**
- [Basic Encoder](example_workflows/z-image_custom_nodes_workflow.json) - Simple replacement for CLIPTextEncode
- [Multi-Turn / Turn Builder](example_workflows/z-image_custom_nodes_turn_builder_workflow.json) - Iterative character edits

**Nodes:**
- `ZImageTextEncoder` - Full-featured with templates, system prompts, thinking blocks, conversation chaining
- `ZImageTextEncoderSimple` - Simplified encoder for quick use / **negative prompts**
- `ZImageTurnBuilder` - Add conversation turns for multi-turn workflows (with optional direct encoding)
- `PromptKeyFilter` - Strip quotes from JSON keys to prevent them appearing as text

**Key Features:**
- System prompts via 140+ templates or custom text
- **Extended template format** - templates can pre-fill thinking content and assistant content
- **Structured prompt templates** - `json_structured`, `yaml_structured`, `markdown_structured` for LLM-generated prompts
- Thinking blocks (`<think>`) to guide model interpretation
- Multi-turn conversations for iterative character edits
- `raw_prompt` for complete control with your own special tokens
- Debug output to see exactly what gets encoded

**Negative Prompts:**
```
Positive: ZImageTextEncoder -> KSampler (positive)
Negative: ZImageTextEncoderSimple -> KSampler (negative)
         user_prompt: "bad anatomy, blurry, watermark"
```

**Example: Character Consistency with Multi-Turn**

Define a detailed character, then make precise edits while maintaining consistency:

```
Turn 1 (ZImageTextEncoder):
  system_prompt: "Generate in comic book style"
  user_prompt: "# Character: Walter Finch, 72yo British gentleman
               - Ice-blue eyes with gold flecks
               - White beard, gold-rimmed glasses
               - Checkered shirt, wool trousers"
  thinking_content: "Make his beard a little red"

Turn 2 (ZImageTurnBuilder):
  user_prompt: "Add a flying sloth above him"
  thinking_content: "Black sloth, floating above his head"
```

The full conversation context is preserved - the model sees the original definition AND all modifications. See the [Character Generation Guide](nodes/docs/z_image_character_generation.md) for the complete Walter Finch example and visual vocabulary reference.

**Documentation:**
- [Z-Image Intro Guide](nodes/docs/z_image_intro.md) - Quick overview, should you use it, progressive examples
- [Z-Image Encoder Reference](nodes/docs/z_image_encoder.md) - Full technical documentation
- [Character Generation Guide](nodes/docs/z_image_character_generation.md) - Multi-turn character consistency, LLM prompt generation

### HunyuanVideo 1.5 Text-to-Video Support

Text-to-video encoding using Qwen2.5-VL with ComfyUI's native HunyuanVideo sampler/VAE.

**Nodes:**
- `HunyuanVideoCLIPLoader` - Load Qwen2.5-VL (byT5 optional for multilingual)
- `HunyuanVideoTextEncoder` - T2V with dual output (positive, negative)
  - Built-in video templates via `template_preset` dropdown
  - `additional_instructions` to layer modifications on templates
  - `custom_system_prompt` for full manual control

**Workflow:** `HunyuanVideoCLIPLoader` -> `HunyuanVideoTextEncoder` -> `KSampler` (or `SamplerCustomAdvanced`) -> `VAEDecode`

## BREAKING CHANGE (v2.7.0+)

**Existing workflows will break.** You must:
1. Delete and re-add, or Right Click -> Recreate Template Builder, Encoder, and Encoder Advanced nodes from your workflow
2. Connect: Template Builder `template_output` → Encoder `template_output` (single connection only)

Old multi-connection system (mode + system_prompt) no longer works as it was getting convoluted and confusing. One connection from template builder now handles everything (prompt, mode, system_prompt).

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

## Features

### Core Capabilities (Qwen-Image-Edit)
- **Qwen-Image-Edit-2509**: Multi-image editing (1-3 optimal, up to 10 max because I had to pick something)
- **QwenImageBatch**: Smart batching with auto-detection, aspect ratio preservation, scaling, batching strategy
- **Resolution Control & Power User Mode**: Per-image resolution control
- **Template Builder Auto-Sync**: Automatic mode matching between template and encoder
- **System Prompt Control**: Customizable system prompts via the template builder
- **Automatic Double-Scaling Prevention**: Batch node and better encoder intelligence
- **Full Debug Output**: Complete prompts, character counts, aspect ratio tracking

### Still Experimental or Not Working Well (or at all)
- Mask-Based Inpainting: Selective editing with diffusers blending, including Eligen entity control

### Recent Highlights

**HunyuanVideo 1.5 (v2.8.0)**
- Video templates in `nodes/templates/hunyuan_video/`
- `HunyuanVideoTextEncoder` with template dropdown + additional instructions
- Dual output (positive/negative) for direct KSampler connection

**QwenImageBatch** - 
- Auto-detects up to 10 images (no inputcount parameter)
- Skips empty inputs (no black images)
- `max_dimensions` strategy (minimal distortion) or `first_image` (hero-driven)
- Prevents double-scaling with metadata propagation

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

### Experimental Nodes (Available but likely not working or deprecated)
- **QwenSmartCrop**: Automated face
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

### Template System (Qwen-Image-Edit)
- Use `QwenTemplateBuilder` to select from 9 image editing templates
- Connect `template_output` to encoder's `template_input`
- Templates in `nodes/templates/*.md` with YAML frontmatter

### Template System (HunyuanVideo)
- Use `template_preset` dropdown directly on `HunyuanVideoTextEncoder`
- Video templates in `nodes/templates/hunyuan_video/` (cinematic, animation, documentary, etc.)
- Use `additional_instructions` to layer modifications on any template
- Use `custom_system_prompt` for full manual control

## Debugging

Debug patches are **disabled by default**. To enable tracing for troubleshooting reference latent flow:

```bash
# Enable debug patches (still silent by default)
export QWEN_ENABLE_DEBUG_PATCHES=true

# Enable verbose output (logs tensor shapes, values, timing)
export QWEN_DEBUG_VERBOSE=true
```

Or use the `QwenDebugController` node for runtime control.

**Note:** Verbose mode adds overhead (GPU-CPU sync for tensor stats). Only enable when actively debugging.

## License

MIT - See LICENSE file for details.
