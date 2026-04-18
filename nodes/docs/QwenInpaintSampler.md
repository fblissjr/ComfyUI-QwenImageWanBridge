# QwenInpaintSampler

**Category:** Qwen/Sampling
**Display Name:** Qwen Inpainting Sampler

**STATUS: EXPERIMENTAL - NOT FULLY TESTED AND LIKELY NOT WORKING**
Inpainting workflow is experimental and may not work as expected or at all. There's likely better alternatives out there.

## Description

Inpainting sampler implementing post-processing latent blending. Core formula: `final = (1-mask)*original + mask*generated`

**Implementation Approach:** Simple blending AFTER generation (post-processing)
- Blends latents outside the model
- No DiT modifications required
- Works with existing qwen-image-edit model

**NOT the DiffSynth EliGen approach** (which uses attention masking inside the DiT)

## Important Notes

1. **This is a 548-line implementation for a simple blending operation.** Consider using standard KSampler + LatentCompositeMasked instead for simpler workflows.

2. **Post-processing vs Attention Masking:**
   - Our approach: Blend latents AFTER sampling (outside model)
   - DiffSynth EliGen: Modify attention DURING sampling (inside model)
   - Trade-off: Simpler but less control than full EliGen

## Inputs

### Required
- **model** (MODEL)
  - Diffusion model (Qwen Image Edit)

- **positive** (CONDITIONING)
  - Positive conditioning from QwenVLTextEncoder

- **negative** (CONDITIONING)
  - Negative conditioning (empty or from encoder)

- **latent_image** (LATENT)
  - Source image encoded to latent space
  - Accepts 4-channel (auto-converts to 16) or 16-channel

- **mask** (MASK)
  - Binary mask from QwenMaskProcessor
  - White areas = inpaint, Black areas = preserve

- **strength** (FLOAT)
  - Inpainting strength
  - Range: 0.0-1.0
  - Default: 1.0
  - 0.0 = preserve original, 1.0 = complete regeneration

- **steps** (INT)
  - Inference steps
  - Range: 1-100
  - Default: 8 (recommended for Lightning LoRA)

- **true_cfg_scale** (FLOAT)
  - True CFG scale from diffusers implementation
  - Range: 0.0-10.0
  - Default: 1.0

- **sampler_name** (ENUM)
  - Sampling method
  - Default: "euler"
  - Options: All ComfyUI samplers

- **scheduler** (ENUM)
  - Scheduler type
  - Default: "normal"
  - Options: All ComfyUI schedulers

- **seed** (INT)
  - Random seed for reproducibility
  - Range: 0 to max int

### Optional
- **padding_mask_crop** (INT)
  - Crop padding around mask
  - Range: 0-512
  - Default: 0 (disabled)
  - May cause shape errors if enabled

## Outputs

- **LATENT**
  - Final inpainted latents with diffusers blending applied
  - 16-channel format for Qwen VAE

## Core Blending Pattern

The critical formula from QwenImageEditInpaintPipeline (line 364):
```python
blended = (1 - mask) * original_latents + mask * generated_latents
```

This ensures:
- Black mask areas (0): 100% original preserved
- White mask areas (1): 100% generated content
- Gray areas: Proportional blend

## Implementation Details

### Channel Handling
- **4-channel input**: Auto-converts to 16-channel (repeats 4x)
- **16-channel input**: Used as-is
- **5D tensors**: Auto-converts to 4D for DiT operations
- **Mask expansion**: 4-channel mask → 16-channel (repeats 4x)

### Processing Steps
1. Validate input channels (4 or 16)
2. Convert 5D→4D if needed (squeeze time dimension)
3. Prepare mask in latent space
4. Expand mask channels to match latent channels
5. Apply padding crop (optional)
6. Generate noise for masked regions
7. Sample with standard ComfyUI sampler
8. Apply core blending formula
9. Return blended latents

### Mask Preparation
```python
def _prepare_mask_latents(mask, latent_image, device, num_channels):
    # Resize to latent dimensions (H/8, W/8)
    # Normalize to 0-1 range
    # Add batch/channel dimensions
    # Return 4D tensor [B, C, H/8, W/8]
```

### Device/Dtype Handling
All tensors synced to:
- Device: Model's device (CPU/CUDA)
- Dtype: Latent's dtype (float16/float32)

## Example Usage

### Basic Inpainting
```
LoadImage → VAEEncode → QwenInpaintSampler (strength=1.0, steps=8)
              ↓ (mask from QwenMaskProcessor)
         QwenVLTextEncoder (conditioning)
              ↓
         VAEDecode → SaveImage
```

### Partial Strength
```
QwenInpaintSampler (
  strength: 0.7,        # 70% regeneration
  steps: 20,
  true_cfg_scale: 1.0
)
```

### With Padding Crop
```
QwenInpaintSampler (
  padding_mask_crop: 64,  # Crop to 64px around mask
  # Processes smaller region for speed
)
```

## Alternative: Simplified Workflow

Instead of this 548-line custom sampler, use:
```
LoadImage → QwenMaskProcessor
              ↓ (image)    ↓ (mask)
         VAEEncode    KSampler (standard ComfyUI)
              ↓            ↓
         LatentCompositeMasked ← mask
              ↓
         VAEDecode
```

Benefits of alternative:
- Uses standard ComfyUI nodes
- Simpler, more maintainable
- Same blending result
- Better integration

## Logging

Detailed logging shows:
- Input validation (channels, dimensions)
- Device/dtype synchronization
- Tensor shape transformations
- Mask preparation steps
- Blending operation details

Enable with Python logging to see full pipeline.

## Diffusers Compatibility

Direct port from `QwenImageEditInpaintPipeline`:
- Exact blending formula (line 1054 in reference)
- Same mask preparation pattern
- Matching strength control
- Compatible noise initialization

## Known Issues

1. **Overcomplicated**: 548 lines for simple blend operation
2. **Padding crop**: May cause shape errors if enabled
3. **5D conversion**: Multiple tensor shape manipulations
4. **Alternative exists**: Standard ComfyUI nodes achieve same result

## Related Nodes

- QwenMaskProcessor - Provides mask input
- QwenVLTextEncoder - Provides conditioning
- VAEEncode - Provides latent_image
- VAEDecode - Decodes output latents
- LatentCompositeMasked - Simpler alternative

## File Location

`nodes/qwen_inpaint_sampler.py`

## Recommendation

**For new workflows**: Consider using `KSampler + LatentCompositeMasked` instead of this custom sampler. Achieves same result with standard ComfyUI nodes and simpler integration.

**Use this node when**: You need exact diffusers compatibility or specific strength/padding features not available in standard nodes.
