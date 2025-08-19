# ComfyUI-QwenImageWanBridge

Direct latent bridge from Qwen-Image to WAN 2.2 video generation. Doesn't work well, may not ever work well, but my head's moving more toward T2V and aligning qwen2.5-vl prompt embeddings with umt5-xxl prompt embeddings, along with the latent qwen image. Perhaps adding noise / 0s to 80 frames of an 81 frame V2V-type generation. Few more ideas to try.

Despite 99.98% VAE similarity and semantic alignment (both Alibaba, WAN trained on Qwen2.5-VL), the small latent distribution differences cause quality loss. I2V models are extremely sensitive to exact latent characteristics. Not sure I2V will ever work better than just decoding it to an image with the VAE. T2V is where we might see something interesting. I don't know what interesting means yet though.

*note* This bridge works ONLY with [Kijai's ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)

## Other Thoughts

1. Frame 0 should not have noise, BUT frames 0+n needs to absolutely think frame 0 has noise and its just denoising from here... so how?
2. If we aligned the two text embeddings, could we somehow predict the noise that frame 0 needs to have?
3. How do we create frames 0+n if we want frame 0 to be the input image without using I2V?
4. Can V2V play a role? (or in this case, simulated V2V)


## Status: Partially Working

**What I think works:**
- Generates temporally coherent video albeit short ones and of low quality
- Preserves Qwen image content in latents and into WAN video
- No VAE decode/encode needed

**Current limitations:**
- Output quality is degraded (low res) - likely due to resolution mismatches and/or noise differences or all sorts of other variables that need time to test
- It may just not work - flat out
- No great workflow yet - but connect qwen-image in comfy native to qwen-image (may need to rescale this - actually probably do) to the bridge node to Kijai's ComfyUI-WanVideoWrapper.

## Quick Start

### Installation
```bash
cd ComfyUI/custom_nodes
git clone [this-repo] ComfyUI-QwenImageWanBridge
```

### Basic Workflow
```
Qwen-Image → QwenWANPureBridge → WanVideoSampler → WanVideoDecode
                    ↓
            (mode="both" for I2V and V2V)
```

### Recommended Settings
```python
# Bridge
width = 832
height = 480
num_frames = 9  # Start small
mode = "both"

# Sampler (critical!)
denoise = 0.3  # Low preserves Qwen
cfg = 4.0
steps = 15
sampler = "DPM-Solver++"
```

## Available Nodes

- **QwenWANPureBridge** - Main bridge, minimal and clean
- **QwenWANSemanticBridge** - Alternative with better resizing
- **QwenWANDimensionHelper** - Find optimal dimensions
- **QwenWANMappingAnalyzer** - Diagnostic tool

### Research (Optional)
Enable in `__init__.py` by setting `LOAD_RESEARCH_NODES = True`



## Alternative for Production

For production quality, use traditional VAE route:
```
Qwen → VAE Decode → Image → WAN VAE Encode → WAN Sampler
```

## Credits

- Qwen-Image and WAN 2.2 by Alibaba
- ComfyUI-WanVideoWrapper by Kijai
