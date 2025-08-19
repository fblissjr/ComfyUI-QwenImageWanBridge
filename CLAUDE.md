# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code and Writing Style Guidelines

- **No emojis** in code, display names, or documentation
- Keep all naming and display text professional
- Avoid "Enhanced", "Advanced", "Ultimate" type prefixes - use descriptive names instead
- Clean, simple node names that describe what they do

## Organization
- `__init__.py` has all nodes
- `example_workflows` has all workflows
- `nodes` has all nodes
- `Documentation` has technical documentation

## Project Overview

ComfyUI-QwenImageWanBridge enables direct latent transfer from Qwen-Image to WAN 2.2 video generation without VAE decode/encode cycles. This leverages the 99.98% VAE compatibility between the models.

## Critical Technical Requirements

See `Documentation/wan_i2v_technical_details.md` for complete implementation details.

### Key Points:
1. **No Batch Dimension**: WAN uses `(C, T, H, W)` not `(B, C, T, H, W)`
2. **I2V Mode**: Don't pass `samples` to sampler, only `image_embeds`
3. **Mask Structure**: Complex 4-channel mask with specific reshaping
4. **Frame Alignment**: Must be `4n+1` frames (81, 85, 89...)
5. **has_ref Flag**: Set False for normal I2V (True drops first frame)

## Node Types

### QwenWANBridgeV2
- Exact replication of WanVideoImageToVideoEncode logic
- Takes Qwen latents instead of images
- Handles all mask creation and frame alignment
- Returns WANVIDIMAGE_EMBEDS for WAN sampler

### QwenWANBridge (Original)
- Simpler implementation with multiple fill modes
- Supports i2v, v2v_zeros, v2v_noise, v2v_repeat, interpolate
- Returns both image_embeds and optional samples

## Common Issues

1. **Pure Noise Generation**: Usually means I2V conditioning not recognized
   - Check no `samples` passed to sampler
   - Verify mask shape is exactly `(4, T, H, W)`
   - Ensure image_embeds is `(C, T, H, W)` without batch

2. **Shape Errors**: WAN expects no batch dimension
   - Remove batch before passing to WAN nodes
   - Add batch back for ComfyUI native nodes

3. **Frame Count Issues**: Must align to 4n+1
   - Use: `((num_frames - 1) // 4) * 4 + 1`

## Workflow Example

```
QwenImage → LATENT → QwenWANBridgeV2 → WANVIDIMAGE_EMBEDS → WanVideoSampler
                                                                    ↓
                                                                 LATENT
                                                                    ↓
                                                            WanVideoDecode → VIDEO
```

## Testing

Run test workflow:
```bash
python tests/test_workflow.py
```

Check debug output for:
- Correct shapes (no batch dimension)
- Proper mask structure
- Frame alignment to 4n+1
