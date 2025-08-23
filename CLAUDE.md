# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code and Writing Style Guidelines

- **No emojis** in code, display names, or documentation
- Keep all naming and display text professional
- Avoid "Pure", "Enhanced", "Advanced", "Ultimate" type prefixes - use descriptive names instead
- Clean, simple node names that describe what they do
- Keep descriptions minimal and factual

## Organization

- `nodes/` - Production-ready nodes only
- `nodes/research/` - Experimental and testing nodes
- `example_workflows/` - Example JSON workflows
- `Documentation/` - Technical documentation and insights

## Critical Architecture Understanding

### Qwen-Image Architecture
- Uses **Qwen2.5 7B VL** as multimodal encoder (3584 dim embeddings)
- **16-channel VAE latents** with specific normalization
- Special vision tokens: `<|vision_start|><|image_pad|><|vision_end|>`
- Vision features from vision tower injected at IMAGE_PAD positions
- Joint attention between text and image streams
- Requires transformers library for proper multimodal processing

### WAN Architecture
- **WAN 2.1**: 16-channel latents (Compatible with Qwen)
  - Same channel count as Qwen-Image
  - Uses specific normalization (mean/std per channel)
  - Best option for Qwen→WAN bridge

- **WAN 2.2**: 48-channel latents
  - 3x more channels than Qwen
  - Requires channel expansion (16→48)

- Both use **UMT5-XXL** for text encoding (4096 dim)
- Both have I2V and T2V cross-attention mechanisms

### Tensor Format Differences
- **Kijai's Wrapper**: `(C, T, H, W)` without batch dimension
- **Native ComfyUI**: `(B, C, T, H, W)` with batch dimension

## ComfyUI Qwen-Image Support

### Official ComfyUI Implementation (Built-in)
ComfyUI has native support for Qwen-Image models. Key components:
- **CLIPLoader**: Load Qwen models with type `"qwen_image"`
- **TextEncodeQwenImageEdit**: Official image edit encoder
- **UNETLoader**: Load fp8 optimized models
- **VAELoader**: Load 16-channel VAE

Example workflow: `Documentation/0822_qe_2.json`

### Our Custom Implementation
Alternative nodes with more flexibility:
- **QwenVLLoader**: Proper Qwen2.5-VL model loader using transformers
- **QwenVLTextEncoder**: Real multimodal text encoder with vision processing
- **QwenVLEmptyLatent**: Generate 16-channel latents with proper normalization
- **QwenVLImageToLatent**: Encode images to Qwen's 16-channel latent space
- **QwenWANNativeBridge**: Native ComfyUI with noise modes
- **QwenWANChannelAdapter**: 16→48 channel expansion
- **QwenAutoregressiveEditor**: Sequential editing capability
- **QwenEliGenController**: Entity-level generation control
- **QwenTextEncoderLoader**: Load text encoder from models/text_encoders/
- **QwenDiffusionModelLoader**: Load diffusion model
- **QwenVAELoader**: Load 16-channel VAE

### When to Use Which?

**Use Official ComfyUI nodes when:**
- You want a production-ready, tested pipeline
- You have fp8 optimized models
- You need Lightning LoRAs for speed
- You prefer native integration

**Use Our Custom nodes when:**
- You need direct transformers integration
- You want transparent processing
- You need flexible loading options
- You're experimenting with modifications

### Key Features

**Vision Token Support (FIXED)**:
- Real multimodal processing using transformers Qwen2VLForConditionalGeneration
- Proper vision encoder that processes images through vision tower
- Exact system prompts from DiffSynth-Studio reference implementation
- Vision features properly injected at IMAGE_PAD position
- Both T2I and Image Edit modes with correct templates

**Channel Adaptation**:
- Direct compatibility with WAN 2.1 (16 channels)
- Multiple expansion modes for WAN 2.2 (48 channels)
- Proper WAN normalization applied

## Recommendations

1. **Install transformers library**: `pip install transformers` (required for custom nodes)
2. **Model Precision**: Use fp8_e4m3fn models for lower VRAM usage
3. **Speed Optimization**: Use Lightning LoRAs (4-step or 8-step variants)
4. **Resolution**: Use preferred Qwen resolutions for best results:
   - Square: 1024×1024, 1328×1328
   - Landscape: 1328×800, 1456×720, 1584×1056
   - Portrait: 800×1328, 720×1456, 1056×1584
5. **Sampler Settings** (from official workflow):
   - With Lightning LoRA: 8 steps, cfg 2.5, euler
   - Without LoRA: 20-40 steps, cfg 2.5
   - Distilled models: 10 steps, cfg 1.0
6. **Model file organization** - For local models in `models/text_encoders/`:
   ```
   models/text_encoders/qwen2.5-vl-7b/
   ├── config.json           # Required: model configuration
   ├── tokenizer.json        # Required: tokenizer
   ├── tokenizer_config.json # Required: tokenizer config
   └── model.safetensors     # Required: model weights
   ```
7. **Use WAN 2.1** for direct Qwen compatibility (16 channels)
8. **For WAN 2.2**, use channel adapter for 16→48 expansion
9. **Low denoise (0.1-0.3)** preserves Qwen structure

## Performance Benchmarks (RTX 4090D 24GB)

| Configuration | VRAM | 1st Gen | 2nd Gen |
|--------------|------|---------|---------|
| fp8_e4m3fn | 86% | ~94s | ~71s |
| fp8 + 8-step LoRA | 86% | ~55s | ~34s |
| Distilled fp8 | 86% | ~69s | ~36s |

## Key Code Locations

- `comfy/text_encoders/qwen_image.py` - Qwen text encoder
- `comfy/ldm/qwen_image/model.py` - Qwen model architecture
- `comfy/ldm/wan/model.py` - WAN model with I2V/T2V attention
- `comfy/latent_formats.py` - WAN 2.1 (16ch) and 2.2 (48ch) definitions
- `comfy/text_encoders/wan.py` - WAN UMT5-XXL encoder

## Testing Checklist

- [ ] Try WAN 2.1 models (16 channels)
- [ ] Test channel adapter for WAN 2.2
- [ ] Use appropriate loaders for each model component
- [ ] Keep denoise low (0.1-0.3) for preservation

## Recent Fixes (2025-08-23)

### Fixed Critical Issues
1. **Vision tokens now actually work** - Images processed through vision tower
2. **Proper multimodal model loading** - Using transformers Qwen2VLForConditionalGeneration
3. **Exact system prompts** - Matches DiffSynth-Studio reference for autoregressive consistency
4. **No more CLIP hacks** - Direct transformers integration instead of forcing through CLIP
5. **Proper 16-channel VAE support** - Correct normalization values from reference

### What Was Wrong Before
- Vision tokens were just text strings, no actual vision processing
- Used ComfyUI's CLIP infrastructure for a vision-language model
- Wrong system prompts affecting autoregressive generation
- Missing pixel_values and image_grid_thw inputs
- Vision features never injected at IMAGE_PAD positions

## Documentation

For detailed feature comparison and technical implementation details, see:
- [Implementation Comparison](Documentation/IMPLEMENTATION_COMPARISON.md) - Complete feature comparison
- [Complete Node Tutorial](Documentation/COMPLETE_NODE_TUTORIAL.md) - Node-by-node guide
- [Reference Latents Explained](Documentation/REFERENCE_LATENTS_EXPLAINED.md) - Reference methods and strength values
- [Qwen Implementation Debug Notes](Documentation/QWEN_IMPLEMENTATION_DEBUG_NOTES.md) - Critical debugging insights and fixes
