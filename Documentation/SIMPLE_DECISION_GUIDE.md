# Qwen Image Edit - Simple Decision Guide

## Quick Setup Checklist

### For ANY Qwen Image Workflow:
1. Load Qwen model → `QwenVLCLIPLoader`
2. Connect VAE to `QwenVLTextEncoder` (ALWAYS!)
3. Choose mode: `text_to_image` or `image_edit`
4. Pick your latent input method (see below)

---

## THE MOST IMPORTANT DECISION

### What connects to KSampler's `latent_image` input?

```
┌─────────────────────────────────────────────────────────┐
│  Want to KEEP the original structure?                    │
│  (Change colors, minor edits, style tweaks)              │
└─────────────────────────────────────────────────────────┘
                          ↓ YES
         LoadImage → VAEEncode → KSampler.latent_image
         Denoise: 0.3-0.7 (lower = more preservation)
         Template: "minimal_edit"


┌─────────────────────────────────────────────────────────┐
│  Want to TRANSFORM completely?                           │
│  (New style, major changes, creative freedom)            │
└─────────────────────────────────────────────────────────┘
                          ↓ YES
         EmptyLatentImage → KSampler.latent_image
         Denoise: 0.9-1.0 (MUST be high!)
         Template: "creative" or "default"
```

---

## Parameter Quick Reference

### QwenVLTextEncoder Settings

| Parameter | What to Choose | Why |
|-----------|---------------|-----|
| **mode** | `image_edit` if you have an input image | Tells model what to expect |
| **template_style** | | |
| → For small changes | `minimal_edit` | Preserves most of original |
| → For creative freedom | `creative` | Allows artistic interpretation |
| → For realistic output | `photorealistic` | Maintains real-world logic |
| **token_removal** | Keep at `auto` | Unless you're debugging |
| **optimize_resolution** | Turn ON | Better quality, uses Qwen resolutions |
| **apply_template** | Keep ON | Unless you're an expert |
| **debug_mode** | Turn ON if confused | Shows what's happening |

### Denoise Settings

**With VAE Encode (structure preservation):**
- 0.3 = Tiny changes only
- 0.5 = Moderate changes
- 0.7 = Significant changes

**With Empty Latent (full generation):**
- Always use 0.9-1.0
- Lower values will look terrible

---

## Common Workflows

### 1. "Change the car color to blue"
```
Mode: image_edit
Latent: VAEEncode → KSampler
Denoise: 0.4
```

### 2. "Make it cyberpunk style"
```
Mode: image_edit
Template: creative
Latent: EmptyLatent → KSampler
Denoise: 1.0
```

### 3. "Fix the lighting"
```
Mode: image_edit
Latent: VAEEncode → KSampler
Denoise: 0.5
```

### 4. "Generate a landscape"
```
Mode: text_to_image
Latent: EmptyLatent → KSampler
Denoise: 1.0
```

---

## Still Confused?

### Turn on debug_mode and check console for:
- Image dimensions at each step
- Token counts
- Reference latent status
- Template being applied

### Remember:
- **VAE must ALWAYS be connected** to encoder for reference latents
- **Reference latents ≠ Structure preservation**
- **Latent input choice** determines structure vs. vision guidance
- **Template style** affects how much creative freedom

---

## Advanced: Two-Stage Refinement

Want even better quality? Add `QwenLowresFixNode`:

```
KSampler → QwenLowresFixNode → VAEDecode
         ↑                    ↑
    (your output)     (refined output)
```

Settings:
- Upscale: 1.5x (default is good)
- Denoise: 0.5 (polish without changing)
