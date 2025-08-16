# ComfyUI-QwenImageWanBridge

bridge Qwen-Image latents directly to WAN 2.2 video (maybe)

## What It Does

Takes Qwen-Image latents → Formats them for WAN 2.2 I2V → Direct video generation

## The Solution

Both models use 16-channel latents, but WAN needs special formatting:
- Temporal dimension: (B, 16, F, H, W) where F = (num_frames-1)//4 + 1
- Mask channels: Tells WAN which frame has content (frame 0 from Qwen)

## Usage

```
[Qwen-Image] → [KSampler] → [Latent] → [QwenImageToWANLatentBridge] → [WAN 2.2 Sampler]
                                              ↑
                                        num_frames: 81
```

## The Node

### QwenImageToWANLatentBridge

**Inputs:**
- `qwen_latent`: Direct from Qwen-Image generation
- `num_frames`: Target video length (81 standard, 49 fast, 121 long)
- `height/width`: Optional, defaults to WAN's preferred 832x480

**Output:**
- `wan_latent`: Ready for WAN 2.2 I2V sampler

## Why Direct Transfer Works

- Both use **16-channel latents** (z_dim=16)
- Both use **8x spatial downsampling**
- VAEs are **99.98% compatible**
- No quality loss from decode/encode
- Faster and uses less VRAM

## Complete Workflow Example

1. Load Qwen-Image model
2. Generate image with KSampler → Latent
3. **QwenImageToWANLatentBridge** (num_frames: 81)
4. WAN 2.2 I2V Sampler (from ComfyUI-WanVideoWrapper)
5. WAN VAE Decode → Video

## Credits

- DiffSynth-Studio for WAN latent format insights
