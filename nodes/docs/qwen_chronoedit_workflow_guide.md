# Qwen → ChronoEdit Workflow Guide

Complete guide for integrating Qwen Image Edit with ChronoEdit video generation using Kijai's WanVideoWrapper nodes.

## Overview

This workflow bridges two powerful systems:
1. **Qwen-Image-Edit**: High-quality image editing with vision-language understanding
2. **ChronoEdit**: Image-to-video generation with temporal consistency

The result: Edit an image with Qwen's precise understanding, then animate it into a video with ChronoEdit.

## Prerequisites

### Models Required

1. **Qwen Models** (QwenImageWanBridge)
   - Text/Vision Encoder: `Qwen/Qwen2.5-VL-7B-Instruct`
   - DiT Model: `qwen-image-edit-2509` (fp8 or Nunchaku)
   - VAE: `qwen_image_vae.safetensors` (16-channel)

2. **ChronoEdit Models** (Kijai's WanVideoWrapper)
   - DiT Model: `Wan2_1-14B-I2V_ChronoEdit_fp8_scaled_KJ.safetensors`
   - LoRA: `chronoedit_distill_lora.safetensors` (for 8-step inference)
   - Text Encoder: `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
   - CLIP Vision: `clip_vision_h.safetensors`
   - VAE: `wan_2.1_vae.safetensors` (same 16-channel as Qwen)

### Custom Nodes Required

1. **ComfyUI-QwenImageWanBridge** (this repository)
   - Provides Qwen editing nodes
   - Provides bridge nodes

2. **Kijai's WanVideoWrapper**
   - Provides ChronoEdit nodes
   - Repository: https://github.com/kijai/ComfyUI-WanVideoWrapper

## Workflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: QWEN IMAGE EDITING                                │
└─────────────────────────────────────────────────────────────┘

LoadImage
    ↓
QwenVLCLIPLoader → QwenVLTextEncoder (mode: image_edit)
    ↓                   ↓
QwenVLEmptyLatent   (text + vision conditioning)
    ↓                   ↓
    └───────→ KSampler ←┘
                ↓
         VAEDecode (Qwen VAE)
                ↓
         [Edited Image]
                ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: PREPARE FOR CHRONOEDIT                            │
└─────────────────────────────────────────────────────────────┘

[Edited Image] → QwenToChronoEditBridge
                        ↓ ↓ ↓
                (image, width, height, frames)
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: CHRONOEDIT VIDEO GENERATION                       │
└─────────────────────────────────────────────────────────────┘

┌── CLIP Vision Path ──┐          ┌── Text Conditioning ──┐
│ CLIPVisionLoader →   │          │ LoadWanVideoT5 →      │
│ WanVideoClipVision   │          │ WanVideoTextEncode    │
│ Encode               │          └───────────────────────┘
└──────────────────────┘                      ↓
         ↓                                    ↓
         └──→ WanVideoImageToVideoEncode ←───┘
                (start_image + clip_embeds)
                        ↓
                (image_embeds dict)
                        ↓
         ┌──────────────┴──────────────┐
         ↓                             ↓
WanVideoModelLoader            WanVideoVAELoader
(ChronoEdit + LoRA)
         ↓
         └───────→ WanVideoSampler
                        ↓
                 VAEDecode (Wan VAE)
                        ↓
                  [Video Frames]
                        ↓
              VHS_VideoCombine → Save
```

## Detailed Node Configuration

### Phase 1: Qwen Image Editing

#### 1. Load Input Image
```
Node: LoadImage
- Select your source image
- Recommended: 1024×1024 or higher
```

#### 2. Load Qwen Model
```
Node: QwenVLCLIPLoader
- Model: Qwen/Qwen2.5-VL-7B-Instruct
```

#### 3. Prepare Text Conditioning
```
Node: QwenVLTextEncoder
- mode: "image_edit"
- text: "Your edit instructions"
  Example: "Change the character's clothing to a red jacket"
- edit_image: Connect from LoadImage
- vae_max_dimension: 2048 (or 1792 with Wan2.1-upscale2x)
```

#### 4. Create Empty Latent
```
Node: QwenVLEmptyLatent
- width: 1024 (or match your target)
- height: 1024
- batch_size: 1
```

#### 5. Sample Edit
```
Node: KSampler
- model: From QwenVLCLIPLoader
- positive: From QwenVLTextEncoder
- negative: Empty or minimal
- latent_image: From QwenVLEmptyLatent
- steps: 20-30
- cfg: 7.0-9.0
- sampler: euler or euler_ancestral
- scheduler: simple
- denoise: 0.5-0.7 (for edit mode)
```

#### 6. Decode Edited Image
```
Node: VAEDecode
- samples: From KSampler
- vae: Qwen 16-channel VAE (loaded separately)

Result: High-quality edited image
```

### Phase 2: Bridge to ChronoEdit

#### 7. Prepare Image for ChronoEdit
```
Node: QwenToChronoEditBridge
- image: From VAEDecode (edited image)
- target_width: 832 (or 720, 1024 - must be 32px aligned)
- target_height: 480 (or 1280)
- num_frames: 17 (or 81 - must be 4n+1)
- resize_mode: "fit" (maintain aspect) or "fill" (crop to fill)
- debug_info: True (to see processing details)

Outputs:
- image: Prepared image (32px aligned)
- width: Final width
- height: Final height
- num_frames: Validated frame count
- info: Processing details
```

### Phase 3: ChronoEdit Video Generation

#### 8. Load ChronoEdit Models

##### Model Loader
```
Node: WanVideoModelLoader
- model: Wan2_1-14B-I2V_ChronoEdit_fp8_scaled_KJ.safetensors
- base_precision: "bf16" (or fp16)
- quantization: "fp8_e4m3fn_scaled" (auto-detect)
- load_device: "offload_device"
- attention_mode: "sdpa" (or flash_attn_2/3)
- lora: Connect from LoraLoaderModelOnly (see below)
```

##### LoRA Loader (for 8-step inference)
```
Node: LoraLoaderModelOnly
- model: From WanVideoModelLoader (connect before)
- lora_name: chronoedit_distill_lora.safetensors
- strength: 1.0

Note: For standard 20-30 step inference, skip LoRA
```

##### Text Encoder
```
Node: LoadWanVideoT5TextEncoder
- model: umt5_xxl_fp8_e4m3fn_scaled.safetensors
```

##### CLIP Vision Encoder
```
Node: CLIPVisionLoader
- clip_name: clip_vision_h.safetensors
```

##### VAE
```
Node: WanVideoVAELoader
- vae_name: wan_2.1_vae.safetensors
```

#### 9. Extract CLIP Vision Features
```
Node: WanVideoClipVisionEncode
- clip_vision: From CLIPVisionLoader
- image_1: From QwenToChronoEditBridge (prepared image)
- strength_1: 1.0
- strength_2: 1.0
- crop: "center"
- combine_embeds: "average"
- force_offload: True

Output: clip_embeds (257 tokens)
```

#### 10. Encode Text Prompt
```
Node: WanVideoTextEncode (or CLIPTextEncode for Wan)
- clip: From LoadWanVideoT5TextEncoder
- text: "Description of desired animation"
  Example: "The character turns their head to look at the camera"
- negative_text: "Static, frozen, no motion, low quality"
```

#### 11. Encode First Frame + CLIP
```
Node: WanVideoImageToVideoEncode
- width: From QwenToChronoEditBridge (or enter manually)
- height: From QwenToChronoEditBridge
- num_frames: From QwenToChronoEditBridge (e.g., 17)
- noise_aug_strength: 0.0
- start_latent_strength: 1.0
- end_latent_strength: 1.0
- force_offload: True
- vae: From WanVideoVAELoader
- clip_embeds: From WanVideoClipVisionEncode
- start_image: From QwenToChronoEditBridge
- tiled_vae: False (or True for VRAM savings)

Output: image_embeds (latent + clip context)
```

#### 12. Apply Flow Shift (for distilled model)
```
Node: ModelSamplingSD3
- model: From WanVideoModelLoader
- shift: 2.0 (for 8-step distilled) or 1.0 (for standard)

Note: Only needed for distilled inference with LoRA
```

#### 13. Sample Video
```
Node: WanVideoSampler
- model: From ModelSamplingSD3 (or WanVideoModelLoader)
- positive: From WanVideoTextEncode
- negative: From WanVideoTextEncode (negative)
- image_embeds: From WanVideoImageToVideoEncode
- steps: 8 (distilled) or 20-30 (standard)
- cfg: 1.0 (distilled) or 7.0-9.0 (standard)
- sampler: "euler" or "euler_ancestral"
- scheduler: "simple"
- seed: Random or fixed
- denoise: 1.0

Output: video_latent [B, 16, T, H, W]
```

#### 14. Decode Video
```
Node: VAEDecode
- samples: From WanVideoSampler
- vae: From WanVideoVAELoader

Output: video_frames [B, T, H, W, 3]
```

#### 15. Save Video
```
Node: VHS_VideoCombine
- images: From VAEDecode
- frame_rate: 16 (or 24, 30)
- format: "video/h264-mp4"
- filename_prefix: "qwen_chronoedit"
```

## Parameter Recommendations

### Resolution Settings

| Target Resolution | Use Case | VRAM (24GB) | Quality |
|---|---|---|---|
| 832×480 | Landscape 16:9 | Safe | Good |
| 720×1280 | Portrait 9:16 | Safe | Good |
| 1024×1024 | Square 1:1 | Moderate | Excellent |
| 1280×720 | HD Landscape | High | Excellent |

### Frame Count Options

| Frames | Duration (16fps) | Use Case |
|---|---|---|
| 17 | ~1 second | Quick test, subtle motion |
| 81 | ~5 seconds | Short animation |
| 161 | ~10 seconds | Full animation |

### Sampling Settings

#### Distilled (8-step with LoRA)
- Steps: 8
- CFG: 1.0
- Flow Shift: 2.0
- Speed: ~2-3 min for 17 frames

#### Standard (no LoRA)
- Steps: 20-30
- CFG: 7.0-9.0
- Flow Shift: 1.0
- Speed: ~5-8 min for 17 frames

## VRAM Optimization

### For 24GB (RTX 4090)
- Use fp8 quantization for all models
- Set `load_device: "offload_device"`
- Enable `force_offload: True` on encoders
- Use `tiled_vae: True` for large images
- Lower resolution or frame count if needed

### For 12-16GB (RTX 4070 Ti)
- Use all above optimizations
- Stick to 832×480 resolution
- Limit to 17 frames
- Consider running Qwen and ChronoEdit separately

## Troubleshooting

### Dimension Errors
**Problem**: "Expected 32px alignment"
**Solution**: Use QwenToChronoEditBridge - it auto-aligns to 32px

### CLIP Token Count Errors
**Problem**: "Expected 257 tokens, got X"
**Solution**: Ensure using clip_vision_h.safetensors, not other CLIP variants

### Out of Memory
**Problem**: CUDA OOM during sampling
**Solutions**:
1. Enable all force_offload options
2. Lower resolution (e.g., 832×480 instead of 1024×1024)
3. Reduce frame count (17 instead of 81)
4. Use tiled_vae=True
5. Close other applications

### Static Video Output
**Problem**: Video has no motion, looks frozen
**Possible Causes**:
1. Wrong flow shift (use 2.0 for distilled, 1.0 for standard)
2. CFG too high with distilled model (use 1.0, not 7.0+)
3. Missing LoRA for distilled model
4. Text prompt too similar to image state

### Artifacts/Speckles
**Problem**: Video has grain or dots
**Solutions**:
1. Use Wan2.1-VAE-upscale2x (eliminates speckles)
2. Lower noise_aug_strength to 0.0
3. Check VAE is actually 16-channel Wan VAE

## Advanced Tips

### Multi-Pass Refinement
1. Edit image with Qwen (low denoise 0.5)
2. Generate video with ChronoEdit
3. Extract key frame
4. Re-edit with Qwen (higher denoise 0.7)
5. Generate refined video

### Prompt Engineering

#### For Qwen Edit
- Be specific about changes: "Change X to Y"
- Mention style preservation: "while keeping the same art style"
- Use negatives: "without changing the background"

#### For ChronoEdit Animation
- Describe motion: "slowly turns head", "blinks eyes"
- Specify direction: "looks toward camera", "moves left"
- Add temporal keywords: "gradual", "smooth", "natural motion"

### Batch Processing
Use ComfyUI's queue feature to:
1. Edit multiple images with Qwen
2. Animate each with ChronoEdit
3. Combine into longer sequence

## Example Prompts

### Qwen Edit Prompts
```
"Change the character's clothing to a futuristic spacesuit while maintaining the same pose and background"

"Add realistic rain effects to the scene with reflections and water droplets"

"Transform the daytime scene to nighttime with appropriate lighting and moon"
```

### ChronoEdit Animation Prompts
```
"The character slowly turns their head to look directly at the viewer with a slight smile"

"Gentle camera push-in while the character's hair moves in a light breeze"

"The character blinks naturally and takes a deep breath"
```

### Negative Prompts (ChronoEdit)
```
"Static, frozen frame, no motion, jerky movement, flickering, compression artifacts, low quality, JPEG artifacts, blurry, distorted"
```

## Performance Benchmarks

### RTX 4090 (24GB) - fp8 Models
- Qwen Edit (1024×1024): ~30s (20 steps)
- ChronoEdit Video (832×480, 17 frames): ~2-3 min (8 steps distilled)
- ChronoEdit Video (1024×1024, 17 frames): ~4-5 min (8 steps distilled)
- Total Pipeline: ~3-6 minutes

### RTX 3090 (24GB) - fp8 Models
- Qwen Edit: ~45s
- ChronoEdit Video (832×480): ~3-4 min
- Total: ~4-8 minutes

## Workflow Variations

### Standalone ChronoEdit (No Qwen)
Skip Phase 1, use LoadImage directly:
```
LoadImage → QwenToChronoEditBridge → ChronoEdit Pipeline
```

### Qwen-Only (No Animation)
Use just Phase 1 for high-quality image editing:
```
LoadImage → Qwen Edit → SaveImage
```

### Alternative: QwenToWanFirstFrameLatent (For Export)
If using external tools:
```
Qwen Edit → QwenToWanFirstFrameLatent (bridge_mode: chronoedit)
         → QwenToWanLatentSaver → Export for external use
```

## Next Steps

After mastering the basic workflow:
1. Experiment with different resolutions and frame counts
2. Try both distilled (8-step) and standard (20-30 step) inference
3. Combine multiple edits into video sequences
4. Explore temporal prompts for complex motions
5. Use Wan2.1-VAE-upscale2x for higher quality output

## References

- QwenImageWanBridge Documentation: `nodes/docs/README.md`
- Kijai's WanVideoWrapper: https://github.com/kijai/ComfyUI-WanVideoWrapper
- ChronoEdit Paper: https://arxiv.org/abs/[ChronoEdit arXiv ID]
- Wan Video: https://github.com/NVIDIA/Wan

## Support

For issues or questions:
- QwenImageWanBridge: GitHub Issues
- ChronoEdit/WanVideoWrapper: Kijai's GitHub
- ComfyUI: ComfyUI Discord/Forums
