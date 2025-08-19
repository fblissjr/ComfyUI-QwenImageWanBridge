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

## Project Status: Partially Working

**What we discovered:** The direct latent bridge DOES work, but with significant quality degradation. Videos are temporally coherent but "low res and crappy looking."

**Why:** Even though the VAEs are 99.98% similar, I2V models are extremely sensitive to exact latent distributions. The 0.02% difference causes noticeable quality loss.

## Critical Compatibility Note

**This ONLY works with Kijai's ComfyUI-WanVideoWrapper**

NOT compatible with:
- Native ComfyUI WAN (uses `(B, C, T, H, W)` with batch dimension)
- Direct ldm/wan usage (different tensor formats)

Kijai's wrapper uses `(C, T, H, W)` without batch dimension, which is what our bridge produces.

## Key Technical Insights

### The Core Discovery
1. **Single frame (num_frames=1)** returns recognizable Qwen image (degraded but visible)
2. **Multiple frames (e.g., 41)** generate temporally coherent video (but low quality)
3. **The issue is NOT complete failure** - it's quality degradation

### Why This Happens
- Both models from Alibaba
- Both use 16-channel VAE latents (confirmed)
- WAN was trained on Qwen2.5-VL captioned datasets (semantic alignment)
- VAEs are 99.98% spatially similar
- BUT: Small latent distribution differences matter for I2V

### The Right Approach
**NO NOISE IN BRIDGE** - Let the sampler handle everything via denoise parameter:
- Frame 0: Clean Qwen latent (acts as V2V conditioning)
- Frames 1+: Zeros (WAN generates these)
- Denoise parameter controls how much to modify Frame 0

### Critical Parameters
```python
# Low denoise preserves Qwen structure
denoise = 0.1-0.5  # NOT 1.0!
cfg = 3-5          # Lower is often better
steps = 10-20
sampler = "DPM-Solver++"
```

## Production Nodes (in `nodes/`)

### QwenWANPureBridge
- The correct, minimal implementation
- NO noise addition
- NO normalization
- Just structural adaptation
- Returns both I2V and V2V outputs with `mode="both"`

### QwenWANSemanticBridge
- Alternative with proper resizing
- Aspect ratio preservation
- Leverages Qwen2.5-VL semantic alignment

### Utilities
- **QwenWANDimensionHelper** - Find optimal dimensions
- **QwenWANMappingAnalyzer** - Diagnostic tool

## Research Nodes (in `nodes/research/`)

Enable by setting `LOAD_RESEARCH_NODES = True` in `__init__.py`

Contains all experimental approaches we tried:
- Various normalization attempts
- T2V experiments
- Noise injection strategies
- Parameter sweep tools

## What Doesn't Work

1. **Adding noise in the bridge** - Makes it worse
2. **Normalization to WAN statistics** - Didn't help
3. **T2V models with empty embeds** - Still produces noise
4. **High denoise values** - Loses Qwen structure entirely

## What Kind of Works

1. **Low denoise (0.1-0.3)** - Preserves some Qwen structure
2. **Single frames** - Best quality (still degraded)
3. **V2V mode** - Treating Frame 0 as clean conditioning

## The Bottom Line

**The direct latent bridge is technically functional but not production-ready due to quality degradation.**

For production use:
```
Qwen → VAE Decode → Image → WAN VAE Encode → WAN
```

Yes, this defeats the purpose, but it's the only way to get acceptable quality currently.

## Future Work Needed

1. **Adapter network** - Train small network to map between latent distributions
2. **Native ComfyUI support** - Add batch dimension handling
3. **Find optimal parameters** - Systematic testing might find better settings
4. **Try other video models** - Some might be more compatible with Qwen latents

## Why We Made So Many Nodes

We explored many hypotheses:
- Maybe it needs normalization? (No)
- Maybe T2V works better? (No)
- Maybe we need to add noise? (No, makes it worse)
- Maybe it's a dimension issue? (Partially)
- Maybe text embedding alignment helps? (Not really)

The final answer: **The sampler's denoise parameter is the key control**, and the bridge should do minimal modification.

## Testing Checklist

- [ ] Start with num_frames=1
- [ ] Use denoise=0.3 or lower
- [ ] Try both I2V and V2V modes
- [ ] Check output - is it recognizable but low quality? That's "working"
- [ ] Don't expect production quality - VAE differences prevent that