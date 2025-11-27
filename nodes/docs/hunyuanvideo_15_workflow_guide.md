# HunyuanVideo 1.5 Workflow Guide

Complete guide for text-to-video (T2V) generation using HunyuanVideo 1.5 in ComfyUI.

**Note:** Image-to-video (I2V) is NOT currently implemented in this node pack.

---

## Prerequisites

### Required Models

**Text Encoders (in `models/text_encoders/`):**
- `Qwen2.5-VL-7B-Instruct` (or similar Qwen2.5-VL 7B model)
- `byt5-small` (optional, for multilingual text)
- `Glyph-SDXL-v2` (optional, for multilingual text rendering)

**HunyuanVideo Models (ComfyUI native, in standard model folders):**
- HunyuanVideo 1.5 DiT transformer
- HunyuanVideo 1.5 VAE
- (Optional) HunyuanVideo upsampler

### Node Categories

Our custom nodes appear in:
- `HunyuanVideo/Loaders` - Model loaders
- `HunyuanVideo/Encoding` - Text encoding

ComfyUI native nodes you'll also use:
- `loaders` - VAELoader, etc.
- `sampling` - KSampler
- `latent` - VAEDecode

---

## Text-to-Video (T2V)

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
│ HunyuanVideoTextEncoder  │ Encode prompt (dual output)
│ (HunyuanVideo/Encoding)  │
└────────────┬─────────────┘
             │ positive
             │ negative
             ↓
┌──────────────────────────┐
│ KSampler                 │ Generate video (ComfyUI native)
│ (sampling)               │ positive, negative, model, latent_image
└────────────┬─────────────┘
             │ LATENT
             ↓
┌──────────────────────────┐
│ VAEDecode                │ Decode to frames (ComfyUI native)
│ (latent)                 │
└────────────┬─────────────┘
             │ IMAGE
             ↓
┌──────────────────────────┐
│ VHS_VideoCombine or      │ Save output
│ SaveAnimatedWEBP         │
└──────────────────────────┘
```

**Important:** All encoder nodes output both `positive` and `negative` conditioning. Connect both to the sampler.

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
- `clip` → Connect to HunyuanVideoTextEncoder

**Notes:**
- byT5 adds ~2-3GB VRAM and slower encoding
- Only enable if your prompt contains non-English text in quotes

#### 2. HunyuanVideoTextEncoder
**Category:** `HunyuanVideo/Encoding`

**Inputs:**
- `clip`: From HunyuanVideoCLIPLoader
- `text`: Your video description
- `negative_prompt`: What to avoid (defaults to standard quality terms)
- `template_preset`: Select from 39 built-in video templates (optional)
- `custom_system_prompt`: Manual system prompt override (optional)
- `additional_instructions`: Extra instructions appended to any template (optional)
- `debug_mode`: Show encoding details (optional)

Example positive prompt:
```
A cinematic shot of ocean waves crashing on a beach at sunset.
The camera slowly pans right, revealing palm trees swaying in the breeze.
Warm golden light fills the scene.
```

**Outputs:**
- `positive` → Connect to sampler's positive input
- `negative` → Connect to sampler's negative input
- `debug_output` → Optional debug info string

**Template Options:**
- `none`: Use ComfyUI default behavior
- `hunyuan_video_cinematic`: Professional cinematography
- `hunyuan_video_animation`: Animation style
- 36 more templates for various genres and styles

**Prompt Tips:**
- Describe the scene, motion, camera movement, lighting
- Be specific about timing (e.g., "slowly", "quickly")
- Include atmosphere/mood descriptors
- Natural language works best
- Use templates to guide style, add `additional_instructions` for fine-tuning

#### 3. KSampler (ComfyUI Native)
**Category:** `sampling`

**Inputs:**
- `model`: HunyuanVideo 1.5 DiT model (from UNETLoader or similar)
- `positive`: From HunyuanVideoTextEncoder positive output
- `negative`: From HunyuanVideoTextEncoder negative output
- `latent_image`: From EmptyHunyuanLatentVideo
- `seed`: Any integer
- `steps`: 30-50 (30 for distilled, 50 for standard)
- `cfg`: 1.0 (distilled) or 6.0 (standard)
- `sampler_name`: `euler`
- `scheduler`: `simple` or `normal`
- `denoise`: 1.0 (for T2V)

**Outputs:**
- `LATENT` → Connect to VAEDecode

#### 4. VAEDecode (ComfyUI Native)
**Category:** `latent`

**Inputs:**
- `samples`: From KSampler
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

### Jerky or unnatural motion

**Solutions:**
1. Increase steps (60-80)
2. Adjust CFG (try 6.0-7.0)
3. Improve prompt with timing words ("slowly", "smoothly", "gradually")

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
4. **Upsampling**: Use HunyuanVideo upsampler for higher resolution output

### Workflow Variations

**T2V with Templates:**
Use the template_preset dropdown to select a style:
- `hunyuan_video_cinematic` - dramatic narrative, professional cinematography
- `hunyuan_video_documentary` - factual, observational
- `hunyuan_video_slowmo` - slow motion effects
- `hunyuan_video_animation` - character design, motion principles
- `hunyuan_video_nature` - landscapes, natural phenomena
- `hunyuan_video_action` - fast-paced, dynamic movement
- Plus 33 more templates (horror, comedy, scifi, fantasy, urban, aerial, etc.)

**T2V with Additional Instructions:**
Layer modifications on top of templates:
- Template: `hunyuan_video_cinematic`
- Additional: `"noir style, high contrast, rain-soaked streets"`

**T2V with Negative Prompts:**
Use the negative_prompt input to avoid:
- "static camera, no motion"
- "blurry, low quality"
- "jerky motion, stuttering"

**Batch Processing:**
Create multiple variations by connecting multiple text encode nodes

---

## Quick Reference

### All 39 Templates

**Core:** t2v, cinematic, animation

**Genre:** action, horror, comedy, scifi, fantasy

**Subject:** nature, wildlife, sports, urban, underwater, aerial

**Production:** product, commercial, documentary, educational, music, interview

**Technical:** timelapse, slowmo, abstract

**Experimental:** structured_realism, minimal_structure, temporal_only, camera_focused, lighting_focused, style_spam, anti_pattern, self_expand

**Fun:** drunk_cameraman, 80s_music_video, majestic_pigeon, wes_anderson_fever, michael_bay_mundane, excited_dog_pov, infomercial_disaster, romcom_lighting

All templates are in `nodes/templates/hunyuan_video_*.md`

### Model Downloads

- **Qwen2.5-VL:** huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- **byT5:** huggingface.co/google/byt5-small
- **Glyph-SDXL-v2:** huggingface.co/AI-ModelScope/Glyph-SDXL-v2

---

**Last Updated:** 2025-11-27
**Version:** 1.3
