# Node Documentation

Comprehensive documentation for all ComfyUI Qwen-Image-Edit nodes.

## Nodes

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
