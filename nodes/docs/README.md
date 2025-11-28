# Node Documentation

Comprehensive documentation for all ComfyUI Qwen nodes.

## Z-Image

Text encoding for Z-Image with Qwen3-4B. Matches diffusers by default, with optional experimental parameters.

### Nodes
- [ZImageTextEncoder](z_image_encoder.md#zimagetextencoder-full-featured) - Full encoder with templates, system prompts, debug output, conversation output
- [ZImageTurnBuilder](z_image_encoder.md#zimageturnbuilder-multi-turn-conversations) - Add conversation turns for multi-turn workflows

### Documentation
- [Z-Image Encoder Guide](z_image_encoder.md) - Complete documentation (nodes, workflows, troubleshooting)
- [Z-Image Turbo Workflow Analysis](z_image_turbo_workflow_analysis.md) - Official workflow breakdown

---

## Qwen-Image-Edit

### Loaders
- [QwenVLCLIPLoader](QwenVLCLIPLoader.md) - Load Qwen2.5-VL text/vision encoder

### Encoding
- [QwenVLTextEncoder](QwenVLTextEncoder.md) - Standard encoder with text-to-image, image_edit, and multi_image_edit modes
- QwenVLTextEncoderAdvanced - Advanced encoder with resolution weighting (docs TODO)

### Templates
- [QwenTemplateBuilder](QwenTemplateBuilder.md) - System prompt templates with 15+ presets and mode auto-sync

### Utilities
- [QwenImageBatch](QwenImageBatch.md) - Smart batching with aspect ratio preservation (v2.6.2)

## Helper Nodes

### Latents
- QwenVLEmptyLatent - 16-channel empty latent creation (docs TODO)
- QwenVLImageToLatent - Image to 16-channel latent conversion (docs TODO)

## Experimental Nodes

**Note:** These nodes are experimental and may not work as expected. Use with caution.

### Inpainting (Not fully tested)
- [QwenMaskProcessor](QwenMaskProcessor.md) - Mask preprocessing (EXPERIMENTAL)
- [QwenInpaintSampler](QwenInpaintSampler.md) - Inpainting sampler (EXPERIMENTAL, consider KSampler + LatentCompositeMasked instead)

### Spatial Control (Low priority)
- QwenSpatialTokenGenerator - Visual spatial token editor (effectiveness unclear, docs TODO)
- QwenEliGenEntityControl - Entity-level mask control (untested, docs TODO)

## Guides

- [resolution_tradeoffs.md](resolution_tradeoffs.md) - Comprehensive resolution and scaling guide
  - Single image edit tradeoffs
  - Multi-image batching tradeoffs
  - Scaling mode details (preserve_resolution, max_dimension_1024, area_1024, no_scaling)
  - QwenImageBatch parameter explanation
  - When to use what strategy

## Quick Links by Use Case

**Text-to-Image:**
- QwenVLCLIPLoader → QwenTemplateBuilder → QwenVLTextEncoder → KSampler

**Single Image Edit:**
- LoadImage → QwenVLTextEncoder → KSampler

**Multi-Image Edit:**
- LoadImage (×N) → QwenImageBatch → QwenVLTextEncoder → KSampler

**With Template Builder Auto-Sync:**
- QwenTemplateBuilder (mode output) → QwenVLTextEncoder (template_mode input)

## Node Priorities

### Tested / Core
- QwenVLCLIPLoader
- QwenVLTextEncoder (standard and advanced)
- QwenTemplateBuilder
- QwenImageBatch
- QwenVLEmptyLatent
- QwenVLImageToLatent

### Experimental or Not Working (Use with caution)
- QwenMaskProcessor
- QwenInpaintSampler
- Wrapper nodes (incomplete)

### Low Priority (Archived/Unclear effectiveness)
- QwenSpatialTokenGenerator
- QwenEliGenEntityControl

---

## HunyuanVideo 1.5

### Nodes
- [HunyuanVideo 1.5 Nodes](hunyuanvideo_15_nodes.md) - CLIP loader, text encoder with 39 templates

### Guides
- [HunyuanVideo 1.5 Workflow Guide](hunyuanvideo_15_workflow_guide.md) - T2V setup
- [HunyuanVideo Prompting Experiments](hunyuanvideo_prompting_experiments.md) - Template testing

---

## Example Workflows

Located in `example_workflows/`:

### Z-Image
- `official_workflows/comfy_z_image_turbo_example_workflow.json` - Official workflow (uses CLIPTextEncode)
- To use our encoder: Replace CLIPTextEncode with ZImageTextEncoder

### Qwen-Image-Edit
- `qwen_edit_2509_single_image_edit.json` - Single image editing
- `nunchaku_qwen_image_edit_2509.json` - Multi-image with Nunchaku

### HunyuanVideo
- `hunyuanvideo_15_t2v_example.json` - Text-to-video
