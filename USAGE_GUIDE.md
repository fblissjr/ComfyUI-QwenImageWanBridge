# Usage Guide for QwenImageWanBridge
basic usage guide, ymmv since in active dev/research

### 1. Recommended: QwenWANUnifiedI2V (All-in-One)

```
[Load Image]
    ↓
[Qwen2VLFlux Encode]
    ↓
[Text Encode] → positive/negative
    ↓
[QwenWANUnifiedI2V]  ← The Swiss Army Knife
    Connect:
    - qwen_latent (from Qwen encoder)
    - positive (from text encode)
    - negative (from text encode)

    Key Settings:
    - i2v_mode: "standard"
    - noise_mode: "no_noise"
    - wan_version: "auto"
    - width: 832, height: 480
    - num_frames: 81
    ↓
[KSampler]  ← Connect ALL THREE outputs!
    - positive (from unified node)
    - negative (from unified node)
    - latent_image (from unified node)
    - model: Your WAN model
    ↓
[VAE Decode]
    ↓
[Save Video]
```

### 2. Direct I2V Conditioning (Better Quality)

```
[Load Image]
    ↓
[Qwen2VLFlux Encode]
    ↓
[QwenWANI2VBridge]  ← Creates proper I2V conditioning
    ├─ positive/negative: text conditioning
    ├─ width: 832
    ├─ height: 480
    └─ num_frames: 81
    ↓
[KSampler]  ← Use the conditioned positive/negative
    ├─ model: WAN model
    ├─ latent: from bridge
    └─ positive/negative: from bridge
    ↓
[VAE Decode]
    ↓
[Save Video]
```

### 3. Latent I2V (Experimental)

```
[Load Image]
    ↓
[Qwen2VLFlux Encode]
    ↓
[QwenWANI2VDirect]  ← Direct latent-to-latent
    ├─ mode: "direct"/"reference"/"hybrid"
    ├─ strength: 0.5-1.0
    └─ dimensions
    ↓
[KSampler]
    ├─ model: WAN 2.1 model
    ├─ steps: 20-30
    └─ denoise: 1.0
    ↓
[VAE Decode]
    ↓
[Save Video]
```

### 4. Using WAN 2.1 vs 2.2 (QwenWANNativeProper)

```
[Load Image]
    ↓
[Qwen2VLFlux Encode]
    ↓
[QwenWANNativeProper]
    ├─ wan_version: "wan21" (or "wan22")
    ├─ channel_mode: "repeat" (for wan22)
    └─ num_frames: 9
    ↓
[KSampler]
    ↓
[VAE Decode]
```

## Node Parameters Explained

### QwenWANNativeBridge

**Main Parameters:**
- `qwen_latent`: Connect from Qwen2VLFlux Encode output
- `width/height`: Target video dimensions (must match model training)
- `num_frames`: Number of video frames to generate

**Noise Modes:**
- `no_noise`: Pure Qwen latent (cleanest but may lack temporal variation)
- `add_noise`: Adds noise on top of Qwen latent
- `mix_noise`: Blends Qwen with noise (good balance)
- `scaled_noise`: Increases noise over time
- `reference_mode`: Treats Qwen as reference/guidance
- `vace_style`: VACE-like conditioning

**Noise Strength:**
- 0.0-0.2: Subtle variation
- 0.3-0.5: Moderate variation (recommended)
- 0.6-1.0: Heavy modification

### QwenWANNativeProper

**Key Parameters:**
- `wan_version`:
  - "wan21": 16-channel model (COMPATIBLE)
  - "wan22": 48-channel model (needs expansion)

- `channel_mode` (for WAN 2.2):
  - "repeat": Simple channel repetition
  - "learnable": Learned projection
  - "attention": Attention-based expansion
  - "frequency": Frequency-domain expansion

### QwenToImage

**Purpose:** Decodes Qwen latent to image for use with standard WanImageToVideo

**Parameters:**
- `qwen_latent`: From Qwen2VLFlux Encode
- `vae`: WAN VAE model

**Use when:** You want to use the native WanImageToVideo node

### QwenWANI2VBridge

**Purpose:** Creates proper I2V conditioning while keeping everything in latent space

**Parameters:**
- `qwen_latent`: From Qwen2VLFlux Encode
- `positive/negative`: Text conditioning
- `width/height`: Video dimensions
- `num_frames`: Total frames (e.g., 81)
- `clip_vision_output`: Optional CLIP vision features

**Outputs:**
- `positive`: Conditioned with I2V latents
- `negative`: Conditioned with I2V latents
- `latent`: Empty latent for generation
- `info`: Debug information

**Use when:** You want better quality by avoiding VAE decode/encode

### QwenWANI2VDirect

**Purpose:** Experimental direct latent-to-latent transfer

**Parameters:**
- `mode`:
  - "direct": Copy Qwen to all frames
  - "reference": First frame only, rest noise
  - "hybrid": Decaying influence over time
- `strength`: How much Qwen influences output (0-1)

**Use when:** Experimenting with different conditioning approaches

## Recommended Settings by Use Case

### High Fidelity (Close to Original)
```
noise_mode: "no_noise" or "add_noise"
noise_strength: 0.1-0.2
denoise: 0.8-0.9
```

### Creative Variation
```
noise_mode: "mix_noise" or "reference_mode"
noise_strength: 0.3-0.5
denoise: 1.0
```

### Maximum Temporal Coherence
```
noise_mode: "vace_style"
noise_strength: 0.2-0.3
Use WAN 2.1 model
```

## Common Issues and Solutions

### Issue: Pixelated/Low Quality Output
**Solution:** You're likely using WAN 2.2 with the basic bridge. Either:
1. Switch to WAN 2.1 model
2. Use QwenWANNativeProper with proper channel expansion

### Issue: No Motion in Video
**Solution:**
1. Increase noise_strength (0.3-0.5)
2. Try "scaled_noise" mode
3. Ensure denoise is set to 1.0 in KSampler

### Issue: Output Doesn't Match Input Image
**Solution:**
1. Reduce noise_strength
2. Use "no_noise" or "add_noise" mode
3. Lower denoise in KSampler (0.7-0.9)

### Issue: Tensor Shape Errors
**Solution:** If using kijai's wrapper, convert from native comfyui, but currently only testing with native comfy for now

## Advanced Workflows

### Multi-Stage Generation
```
Stage 1: Qwen → Bridge (reference_mode) → Low-res preview
Stage 2: Preview → Upscale → Bridge (no_noise) → High-res output
```

### Hybrid Text+Image Control
```
[Qwen Latent] → Bridge → KSampler
                           ↑
                    [Text Prompt]
```

## Tips for Best Results

1. **Start with WAN 2.1** - It's natively compatible with Qwen's 16 channels
2. **Test noise modes** - Different content works better with different modes
3. **Adjust KSampler settings** - CFG scale 7-15, steps 20-30
4. **Match training resolution** - WAN works best at its training resolutions
5. **Use seed fixing** - Set a fixed seed while testing parameters

## Example Complete Workflow

```python
# Node setup in ComfyUI
nodes = {
    "load_image": LoadImage(image="input.jpg"),
    "qwen_encode": Qwen2VLFluxEncode(
        image=nodes["load_image"],
        prompt="a video of"
    ),
    "bridge": QwenWANNativeBridge(
        qwen_latent=nodes["qwen_encode"],
        width=832,
        height=480,
        num_frames=9,
        noise_mode="mix_noise",
        noise_strength=0.3,
        seed=42
    ),
    "sampler": KSampler(
        model=wan21_model,
        latent=nodes["bridge"],
        steps=25,
        cfg=10,
        denoise=1.0
    ),
    "decode": VAEDecode(
        samples=nodes["sampler"],
        vae=wan_vae
    ),
    "save": SaveAnimatedWEBP(
        images=nodes["decode"],
        fps=8
    )
}
```

## Debugging

Enable info output to see what's happening:
- Bridge nodes output an "info" string with details
- QwenWANNoiseAnalyzer can show latent statistics
- Use PreviewImage nodes to check intermediate frames

## Next Steps

1. Start with the simple workflow
2. Experiment with noise modes
3. Try WAN 2.1 vs 2.2 comparison
4. Adjust parameters based on your content
5. Share results and feedback for improvements!
