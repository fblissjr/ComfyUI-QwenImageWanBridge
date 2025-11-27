# HunyuanVideo 1.5 Workflow Guide

Complete guide for text-to-video (T2V) and image-to-video (I2V) generation using HunyuanVideo 1.5 in ComfyUI.

---

## Prerequisites

### Required Models

**Text Encoders (in `models/text_encoders/`):**
- `Qwen2.5-VL-7B-Instruct` (or similar Qwen2.5-VL 7B model)
- `byt5-small` (optional, for multilingual text)
- `Glyph-SDXL-v2` (optional, for multilingual text rendering)

**Vision Encoder (in `models/clip_vision/`):**
- `siglip_vision_patch14_384.safetensors` (for I2V only)

**HunyuanVideo Models (ComfyUI native, in standard model folders):**
- HunyuanVideo 1.5 DiT transformer
- HunyuanVideo 1.5 VAE
- (Optional) HunyuanVideo upsampler

### Node Categories

Our custom nodes appear in:
- `HunyuanVideo/Loaders` - Model loaders
- `HunyuanVideo/Encoding` - Text/vision encoding

ComfyUI native nodes you'll also use:
- `loaders` - LoadImage, VAELoader, etc.
- `conditioning` - CLIPVisionEncode (for I2V)
- `sampling` - HunyuanVideoSampler
- `latent` - VAEDecode

---

## Workflow 1: Text-to-Video (T2V)

Generate video from text description only.

### Node Chain

```
┌──────────────────────────┐
│ HunyuanVideoCLIPLoader   │ Load dual text encoders
│ (HunyuanVideo/Loaders)   │ (Qwen2.5-VL + optional byT5)
└────────────┬─────────────┘
             │ clip
             ↓
┌──────────────────────────┐
│ HunyuanVideoTextEncode   │ Encode prompt
│ (HunyuanVideo/Encoding)  │
└────────────┬─────────────┘
             │ conditioning
             ↓
┌──────────────────────────┐
│ HunyuanVideoSampler      │ Generate video (ComfyUI native)
│ (sampling)               │
└────────────┬─────────────┘
             │ latent
             ↓
┌──────────────────────────┐
│ VAEDecode                │ Decode to frames (ComfyUI native)
│ (latent)                 │
└────────────┬─────────────┘
             │ IMAGE
             ↓
┌──────────────────────────┐
│ SaveImage / VideoSave    │ Save output
└──────────────────────────┘
```

### Step-by-Step Setup

#### 1. HunyuanVideoCLIPLoader
**Category:** `HunyuanVideo/Loaders`

**Inputs:**
- `qwen_model`: Select your Qwen2.5-VL model
  - Example: `Qwen2.5-VL-7B-Instruct`
- `byt5_model`: `None` (for English-only)
  - Or select `byt5-small` + `Glyph-SDXL-v2` for multilingual
- `enable_byt5`: `false` (English-only, faster)
  - Set `true` only if you need multilingual text rendering

**Outputs:**
- `clip` → Connect to HunyuanVideoTextEncode

**Notes:**
- byT5 adds ~2-3GB VRAM and slower encoding
- Only enable if your prompt contains non-English text in quotes

#### 2. HunyuanVideoTextEncode
**Category:** `HunyuanVideo/Encoding`

**Inputs:**
- `clip`: From HunyuanVideoCLIPLoader
- `text`: Your video description
  - Example: `"A cinematic shot of ocean waves crashing on a beach at sunset. The camera slowly pans right, revealing palm trees swaying in the breeze. Warm golden light fills the scene."`

**Outputs:**
- `conditioning` → Connect to HunyuanVideoSampler

**Prompt Tips:**
- Describe the scene, motion, camera movement, lighting
- Be specific about timing (e.g., "slowly", "quickly")
- Include atmosphere/mood descriptors
- Natural language works best
- No need to manually add system templates (handled automatically)

#### 3. HunyuanVideoSampler (ComfyUI Native)
**Category:** `sampling`

**Inputs:**
- `model`: HunyuanVideo 1.5 DiT model
- `positive`: From HunyuanVideoTextEncode
- `negative`: (optional) Negative prompt
- `latent_image`: Empty latent or first frame latent
- `steps`: 50 (default)
- `cfg`: 6.0-8.0 (recommended)
- `sampler_name`: `euler` (recommended)
- `scheduler`: `normal`
- `denoise`: 1.0 (for T2V)

**Outputs:**
- `LATENT` → Connect to VAEDecode

#### 4. VAEDecode (ComfyUI Native)
**Category:** `latent`

**Inputs:**
- `samples`: From HunyuanVideoSampler
- `vae`: HunyuanVideo VAE

**Outputs:**
- `IMAGE` → Connect to SaveImage or video output

### Example T2V Prompt

```
A professional cinematic video shot. The camera starts with a close-up of dewdrops
on a spider web at dawn. Soft focus bokeh in the background. The camera slowly pulls
back to reveal the web hanging between two branches in a misty forest. Golden morning
sunlight filters through the trees, creating volumetric rays of light. The motion
is smooth and ethereal, with a dreamlike quality.
```

---

## Workflow 2: Image-to-Video (I2V)

Animate a static image using text to describe the motion.

### Node Chain

```
┌──────────────────────────┐
│ LoadImage                │ Load reference frame
└────────────┬─────────────┘
             │ IMAGE
             ↓
         ┌───────────────────────┐
         │ CLIPVisionEncode      │ Encode reference image
         │ (ComfyUI native)      │ with SigLIP
         └───────┬───────────────┘
                 │ CLIP_VISION_OUTPUT
                 │
┌──────────────────────────┐     │
│ HunyuanVideoVisionLoader │     │
│ (HunyuanVideo/Loaders)   │     │
└────────────┬─────────────┘     │
             │ clip_vision        │
             └────────────────────┘
                 (connect to CLIPVisionEncode)

┌──────────────────────────┐
│ HunyuanVideoCLIPLoader   │ Load text encoders
└────────────┬─────────────┘
             │ clip
             ↓
┌──────────────────────────┐
│ HunyuanVideoTextEncodeI2V│ Combine text + vision
│ (HunyuanVideo/Encoding)  │
└────────────┬─────────────┘
             │ conditioning
             ↓
┌──────────────────────────┐
│ HunyuanVideoSampler      │ Generate video
└────────────┬─────────────┘
             │ latent
             ↓
┌──────────────────────────┐
│ VAEDecode                │ Decode to frames
└────────────┬─────────────┘
             │ IMAGE
             ↓
┌──────────────────────────┐
│ SaveImage / VideoSave    │ Save output
└──────────────────────────┘
```

### Step-by-Step Setup

#### 1. LoadImage (ComfyUI Native)
**Category:** `loaders`

**Inputs:**
- `image`: Your reference frame (first frame of the video)

**Outputs:**
- `IMAGE` → Connect to CLIPVisionEncode

**Notes:**
- Resolution will be handled by the model
- Aspect ratio preserved during encoding

#### 2. HunyuanVideoVisionLoader
**Category:** `HunyuanVideo/Loaders`

**Inputs:**
- `vision_model`: Select SigLIP model
  - Example: `siglip_vision_patch14_384.safetensors`

**Outputs:**
- `clip_vision` → Connect to CLIPVisionEncode

#### 3. CLIPVisionEncode (ComfyUI Native)
**Category:** `conditioning`

**Inputs:**
- `clip_vision`: From HunyuanVideoVisionLoader
- `image`: From LoadImage
- `crop`: `center` (recommended)

**Outputs:**
- `CLIP_VISION_OUTPUT` → Connect to HunyuanVideoTextEncodeI2V

**Notes:**
- This extracts visual features from your reference image
- SigLIP processes it at 384×384 internally

#### 4. HunyuanVideoCLIPLoader
**Category:** `HunyuanVideo/Loaders`

Same as T2V workflow (see above).

#### 5. HunyuanVideoTextEncodeI2V
**Category:** `HunyuanVideo/Encoding`

**Inputs:**
- `clip`: From HunyuanVideoCLIPLoader
- `clip_vision_output`: From CLIPVisionEncode
- `text`: Motion description (NOT image description)

**Outputs:**
- `conditioning` → Connect to HunyuanVideoSampler

**IMPORTANT Prompting for I2V:**
- **Don't** describe the static image
- **Do** describe the motion/animation
- **Do** describe camera movement
- **Do** describe timing and speed

**Good I2V Prompt:**
```
The camera slowly zooms in on the subject's face. They turn their head
to the left and smile slightly. Soft natural lighting remains constant.
The motion is smooth and cinematic, taking 3-4 seconds total.
```

**Bad I2V Prompt:**
```
A portrait of a person with brown hair wearing a blue shirt against
a white background.
```
(This describes the image, not the motion!)

#### 6. HunyuanVideoSampler (ComfyUI Native)
**Category:** `sampling`

Same as T2V, but:
- `denoise`: 0.75-0.85 (for I2V, not 1.0)
- Lower denoise = more faithful to reference image
- Higher denoise = more creative interpretation

#### 7. VAEDecode (ComfyUI Native)
Same as T2V workflow.

### Example I2V Prompts

**Portrait Animation:**
```
The subject slowly turns their head from facing forward to looking over their
right shoulder. Their expression changes from neutral to a subtle smile.
Natural lighting creates soft shadows. The motion takes about 3 seconds and
is smooth and natural.
```

**Landscape Animation:**
```
The camera performs a slow right-to-left pan across the scene. Clouds drift
slowly in the sky. Leaves rustle gently in the breeze. Golden hour lighting
gradually shifts as the sun lowers. The entire motion takes 5-6 seconds with
a cinematic, contemplative pace.
```

**Product Animation:**
```
The camera orbits 90 degrees clockwise around the product. Studio lighting
remains constant with a key light from the left. The product rotates on its
base to reveal different angles. Motion is smooth and commercial-quality,
completing in 4 seconds.
```

---

## Settings & Parameters

### Text Encoder Settings

**byT5 Usage:**
- **Disable** (`enable_byt5=false`):
  - English-only prompts
  - Faster encoding
  - ~2-3GB less VRAM

- **Enable** (`enable_byt5=true`):
  - Multilingual text in quotes
  - Example: `A sign that says "你好世界"`
  - Triggers Glyph-SDXL-v2 for accurate text rendering

### Sampling Parameters

**Steps:**
- 50 steps: Default, good quality
- 30-40 steps: Faster, slight quality loss
- 60-80 steps: Maximum quality, slower

**CFG (Classifier-Free Guidance):**
- 6.0-7.0: Balanced (recommended)
- 4.0-5.0: More creative, less prompt adherence
- 8.0-10.0: Strong prompt adherence, may be less natural

**Denoise:**
- **T2V**: 1.0 (generate from scratch)
- **I2V**: 0.75-0.85 (animate reference image)
  - 0.75: Very faithful to reference
  - 0.85: More creative interpretation
  - 0.90+: May lose reference details

### Resolution & Aspect Ratios

**Supported aspect ratios:**
- 16:9 (landscape, default)
- 9:16 (portrait)
- 1:1 (square)
- 4:3 (standard)

**Typical resolutions:**
- 480p: 854×480 (faster, lower VRAM)
- 720p: 1280×720 (balanced)
- 1080p: 1920×1080 (high quality, more VRAM)

**VRAM estimates:**
- 480p @ 50 steps: ~8-10GB
- 720p @ 50 steps: ~12-16GB
- 1080p @ 50 steps: ~20-24GB

---

## Troubleshooting

### "Model not found" errors

**Solution:** Ensure models are in correct folders:
```
ComfyUI/models/
  text_encoders/
    Qwen2.5-VL-7B-Instruct/
    byt5-small/
    Glyph-SDXL-v2/
  clip_vision/
    siglip_vision_patch14_384.safetensors
  diffusion_models/
    hunyuan-video-1.5/
  vae/
    hunyuan-video-1.5-vae/
```

### Out of Memory (OOM)

**Solutions:**
1. Lower resolution (480p instead of 720p)
2. Reduce steps (30-40 instead of 50)
3. Disable byT5 (`enable_byt5=false`)
4. Enable CPU offloading in ComfyUI settings
5. Close other applications

### I2V doesn't match reference image

**Solutions:**
1. Increase `denoise` (try 0.80-0.85)
2. Improve prompt specificity
3. Check that CLIPVisionEncode is properly connected
4. Verify reference image loaded correctly

### Jerky or unnatural motion

**Solutions:**
1. Increase steps (60-80)
2. Adjust CFG (try 6.0-7.0)
3. Improve prompt with timing words ("slowly", "smoothly", "gradually")
4. Use I2V with lower denoise for smoother interpolation

### byT5 text rendering not working

**Solutions:**
1. Ensure `enable_byt5=true`
2. Put text in quotes: `"你好"` or `"Hello"`
3. Verify both byt5-small AND Glyph-SDXL-v2 are loaded
4. Check model paths in error logs

---

## Advanced Tips

### Camera Movement Vocabulary

Use these terms for specific camera motions:
- **Pan**: Horizontal camera rotation (left/right)
- **Tilt**: Vertical camera rotation (up/down)
- **Dolly/Track**: Camera moves forward/backward
- **Truck**: Camera moves left/right (parallel to subject)
- **Pedestal**: Camera moves up/down (vertical)
- **Zoom**: Lens zoom in/out
- **Orbit**: Camera circles around subject

### Timing & Pacing

Specify timing explicitly:
- "over 2 seconds"
- "gradually over 5 seconds"
- "quickly in 1 second"
- "holds for a beat, then..."

### Lighting & Atmosphere

Rich descriptions improve quality:
- "golden hour lighting"
- "soft window light from the left"
- "dramatic rim lighting"
- "overcast diffuse light"
- "volumetric god rays"

### Combining Multiple Motions

Structure complex motions clearly:
```
The camera starts with a close-up, then slowly dollies backward
while simultaneously panning right. As the camera moves, the subject
turns to face the camera. The entire sequence takes 6 seconds, with
the camera movement completing at 4 seconds and the head turn
completing at 5 seconds.
```

---

## Next Steps

### Experimentation Ideas

1. **Loop Creation**: Generate 4-5 second clips designed to loop seamlessly
2. **Style Mixing**: Try different artistic styles in prompts (cinematic, documentary, commercial)
3. **Speed Variations**: Test slow-motion vs real-time motion
4. **Multi-Stage**: Chain multiple I2V generations with last frame as next reference
5. **Upsampling**: Use HunyuanVideo upsampler for higher resolution output

### Workflow Variations

**T2V with Negative Prompts:**
Add negative prompt node to avoid:
- "static camera, no motion"
- "blurry, low quality"
- "jerky motion, stuttering"

**I2V with Multiple References:**
(Advanced) Use sequence of images for smoother animations

**Batch Processing:**
Create multiple variations by connecting multiple text encode nodes

---

## Resources

- **Resolution & Frame Guide:** `nodes/docs/hunyuanvideo_15_resolution_frame_guide.md`
  - Complete technical reference for resolutions, aspect ratios, frame counts
  - VRAM requirements by configuration
  - Model versions (standard, distilled, sparse)
  - Super-resolution upsampling details
- Example workflows: `example_workflows/hunyuanvideo_15_*.json`
- Technical details: `CLAUDE.md` (HunyuanVideo 1.5 section)
- Model downloads:
  - Qwen2.5-VL: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
  - byT5: https://huggingface.co/google/byt5-small
  - Glyph-SDXL-v2: https://huggingface.co/AI-ModelScope/Glyph-SDXL-v2
  - SigLIP: https://huggingface.co/timm (or included in ComfyUI)

---

**Last Updated:** 2025-01-24
**Version:** 1.0
