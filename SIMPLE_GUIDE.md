# Simple Usage Guide - Qwen to WAN I2V Bridge

## Just Use This One Node: QwenWANUnifiedI2V

The **Unified I2V** node is all you need. It's the Swiss Army knife that does everything.

## Basic Setup (90% of use cases)

```
[Load Image]
    ↓
[Qwen2VLFlux Encode]
    ↓
[Text Encode] → positive/negative
    ↓
[QwenWANUnifiedI2V]
    Connect:
    - qwen_latent (from Qwen encoder)
    - positive (from text encode)
    - negative (from text encode)

    Settings:
    - i2v_mode: "standard"
    - noise_mode: "no_noise"
    - width: 832, height: 480
    - num_frames: 81
    ↓
[KSampler]
    Connect ALL outputs:
    - positive (from unified node)
    - negative (from unified node)
    - latent_image (from unified node)
    - model: Your WAN model
    ↓
[VAE Decode]
    ↓
[Save Video]
```

## Key Settings Explained

### i2v_mode (What strategy to use)
- **"standard"** - Best for most cases, proper I2V conditioning
- **"direct_latent"** - Experimental, injects Qwen directly
- **"reference"** - Uses Qwen as reference/guide only
- **"hybrid"** - Blends approaches
- **"vace_style"** - Keyframe-based conditioning

### noise_mode (How much variation)
- **"no_noise"** - Exact Qwen latent (highest fidelity)
- **"add_noise"** - Adds variation on top
- **"mix_noise"** - Blends with noise
- **"scaled_noise"** - Increases over time
- **"decay_noise"** - Decreases over time

### Other Important Settings
- **noise_strength**: 0.0-0.2 for subtle, 0.3-0.5 for moderate
- **start_frames**: How many frames to condition (1-3 typical)
- **frame_blend**: How much influence decays (1.0 = full)
- **wan_version**: "auto" detects, or force "wan21"/"wan22"

## Common Scenarios

### "I want the video to look exactly like my input image"
```
i2v_mode: "standard"
noise_mode: "no_noise"
start_frames: 3
frame_blend: 1.0
```

### "I want creative variation based on my image"
```
i2v_mode: "hybrid"
noise_mode: "mix_noise"
noise_strength: 0.3
frame_blend: 0.7
```

### "I want to use multiple keyframes"
```
i2v_mode: "vace_style"
start_frames: 4 (number of keyframes)
noise_strength: 0.2
```

### "I'm getting pixelated output"
```
wan_version: "wan21" (force WAN 2.1)
```

## Input Options

The node accepts multiple input types:
- **qwen_latent**: From Qwen2VLFlux Encode (recommended)
- **bridge_latent**: From our other bridge nodes
- **start_image + vae**: Direct image input

## Core Concept Reminder

This bridge attempts to use Qwen-Image's latent representation directly with WAN video generation, avoiding the quality loss from VAE decode/encode cycles.

**Why it's tricky**: Despite both being from Alibaba and WAN being trained on Qwen embeddings, the latent spaces aren't perfectly aligned. Small differences cause quality issues.

**Best results**:
- Use WAN 2.1 (16 channels match)
- Start with "standard" mode
- Adjust noise for variation
- Test different settings

## Troubleshooting

**Nothing connects?**
- Restart ComfyUI after installing

**Error about channels?**
- Set wan_version to "wan21"

**Too much/little motion?**
- Adjust noise_strength

**Doesn't look like input?**
- Reduce noise_strength
- Increase start_frames
- Try "standard" mode
