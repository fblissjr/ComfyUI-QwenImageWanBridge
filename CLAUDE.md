# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code and Writing Style Guidelines

- **No emojis** in code, display names, or documentation
- Keep all naming and display text professional
- Avoid "Enhanced", "Advanced", "Ultimate" type prefixes - use descriptive names instead
- Clean, simple node names that describe what they do

## Organization

- `nodes/` - Production-ready nodes only
- `nodes/research/` - Experimental and testing nodes
- `example_workflows/` - Example JSON workflows
- `Documentation/` - Technical documentation and insights

## Critical Architecture Understanding (UPDATED)

### Qwen-Image Architecture
- Uses **Qwen2.5 7B VLI** as text encoder (3584 dim embeddings)
- **16-channel VAE latents** (confirmed)
- Special vision tokens: `<|vision_start|><|image_pad|><|vision_end|>`
- Joint attention between text and image streams (see `comfy/ldm/qwen_image/model.py`)

### WAN Architecture - CRITICAL DISCOVERY
- **WAN 2.1**: 16-channel latents (COMPATIBLE with Qwen!)
  - Same channel count as Qwen-Image
  - Uses specific normalization (mean/std per channel)
  - Best option for Qwen→WAN bridge
  
- **WAN 2.2**: 48-channel latents (INCOMPATIBLE)
  - 3x more channels than Qwen
  - This explains pixelation in native ComfyUI
  - Requires sophisticated channel expansion (16→48)
  
- Both use **UMT5-XXL** for text encoding (4096 dim)
- Both have I2V and T2V cross-attention mechanisms

### Tensor Format Differences
- **Kijai's Wrapper**: `(C, T, H, W)` without batch dimension
- **Native ComfyUI**: `(B, C, T, H, W)` with batch dimension
- This requires different handling in each implementation

## Project Status: Partially Working

### Testing Results Summary

**Kijai's Wrapper (16ch assumed)**:
- Single frame (num_frames=1): Recognizable but degraded
- Multiple frames: Temporally coherent but low quality
- Works because wrapper might handle 16ch internally

**Native ComfyUI**:
- **I2V**: Pixelated output - likely due to 16ch→48ch mismatch
- **T2V**: Text prompt drives 99% of generation
- Qwen latent acts as weak "reference" at best

### Root Cause Analysis

The pixelation and quality issues are now understood:
1. **Channel Mismatch**: Feeding 16-channel Qwen to 48-channel WAN 2.2
2. **Normalization**: WAN expects specific mean/std distributions
3. **VAE Differences**: Even 99.98% similarity isn't enough for I2V
4. **Text Encoder Mismatch**: Qwen uses Qwen2.5, WAN uses UMT5-XXL

## Current Nodes

### Production Ready (Native ComfyUI Only)
- **QwenWANNativeBridge**: Native ComfyUI with noise modes
- **QwenWANNativeProper**: NEW - Handles WAN 2.1 (16ch) vs 2.2 (48ch)
- **QwenWANChannelAdapter**: NEW - Sophisticated 16→48 channel expansion

Note: All production nodes work with **native ComfyUI** implementation only. They return standard `LATENT` types compatible with ComfyUI's KSampler. Kijai's wrapper would require different tensor formats `(C, T, H, W)` and specific return types like `WANVIDIMAGE_EMBEDS`.

### Key Features of New Nodes

**QwenWANNativeProper**:
- Detects WAN version (2.1 vs 2.2)
- Direct compatibility with WAN 2.1 (16 channels)
- Multiple channel expansion modes for WAN 2.2
- Proper WAN normalization applied

**QwenWANChannelAdapter**:
- Frequency-based channel expansion
- Multi-scale representations
- Phase-shifted variations
- Mixed adaptation strategies

## Recommendations

1. **Use WAN 2.1 instead of WAN 2.2** for Qwen compatibility
2. **For WAN 2.2**, use QwenWANChannelAdapter for proper 16→48 expansion
3. **Low denoise (0.1-0.3)** preserves Qwen structure
4. **Reference/VACE modes** treat Qwen as guidance rather than exact input

## What Works (Kind Of)

1. **WAN 2.1 with proper normalization** - Best compatibility
2. **Reference mode** - Qwen as "phantom" influence
3. **VACE-style** - Qwen as keyframe references
4. **Low denoise** - Preserves some structure

## What Doesn't Work

1. **Direct 16→48 channel feeding** - Causes pixelation
2. **High denoise** - Loses all Qwen structure
3. **Pure T2V** - Ignores latent, uses text only
4. **Expecting pixel-perfect** - VAE differences prevent this

## The Bottom Line

**For production quality**, use the traditional VAE route:
```
Qwen → VAE Decode → Image → WAN VAE Encode → WAN
```

**For experimentation**:
- Try WAN 2.1 (16 channels) with QwenWANNativeProper
- Use channel adapter for WAN 2.2
- Treat Qwen as reference/guidance, not exact input

## Key Code Locations

- `comfy/text_encoders/qwen_image.py` - Qwen text encoder
- `comfy/ldm/qwen_image/model.py` - Qwen model architecture
- `comfy/ldm/wan/model.py` - WAN model with I2V/T2V attention
- `comfy/latent_formats.py` - WAN 2.1 (16ch) and 2.2 (48ch) definitions
- `comfy/text_encoders/wan.py` - WAN UMT5-XXL encoder

## Testing Checklist

- [ ] Try WAN 2.1 models (16 channels)
- [ ] Use QwenWANNativeProper with wan_version="wan21"
- [ ] Test channel adapter for WAN 2.2
- [ ] Use reference/VACE modes for guidance approach
- [ ] Keep denoise low (0.1-0.3)
- [ ] Don't expect pixel-perfect results