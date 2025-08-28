# Enhanced Qwen Image Nodes Guide

## Overview
Our enhanced Qwen nodes now include all fixes and features from DiffSynth-Studio and DiffSynth-Engine, achieving 100% compatibility with ComfyUI while adding advanced capabilities.

## Key Enhancements

### 1. RoPE Position Embedding Fix
- **What**: Fixes batch processing with different image sizes
- **Why**: Prevents crashes and artifacts when processing multiple images
- **Source**: DiffSynth-Studio commit 8fcfa1d
- **Status**: Automatically applied via monkey patch on module load

### 2. Reference Latents Support
- **What**: Properly passes reference latents through conditioning
- **Why**: Essential for image editing to preserve structure
- **Methods**:
  - `standard`: Default ComfyUI method
  - `index`: Flux-style indexed reference
  - `offset`: Flux-style offset reference

### 3. Template Styles
- **default**: Standard Qwen templates from DiffSynth
- **minimal_edit**: Preserves maximum original content
- **creative**: Artistic interpretation allowed
- **photorealistic**: Focus on realistic output

### 4. Optimal Resolution Support
- **What**: Auto-resizes to nearest Qwen-supported resolution
- **Why**: Better quality and faster generation
- **Resolutions**: 36 official Qwen resolutions from 256x256 to 1792x512

### 5. Lowres Fix (Two-Stage Refinement)
- **What**: DiffSynth-Studio's two-stage generation method
- **Why**: Higher quality outputs with better details
- **Process**:
  1. Generate at current resolution (full denoise)
  2. Upscale by factor (1.0-4.0x)
  3. Refine with partial denoise (0.3-0.7 typical)

## Node Reference

### QwenVLTextEncoder (Enhanced)

**Parameters:**
- `use_custom_system_prompt`: (default False)
  - False: Applies default Qwen formatting automatically
  - True: Uses pre-formatted text from Template Builder
- `optimize_resolution`: (default False) Auto-optimize to Qwen resolution with aspect ratio preservation
  - False: Standard 1M pixel scaling
  - True: Snap to nearest Qwen-supported resolution
- `multi_reference`: Optional input accepts `QwenMultiReferenceHandler` output
- `debug_mode`: Detailed logging for troubleshooting
- Note: `reference_method` removed - automatically determined from multi-reference data

**Usage Examples:**

#### Basic Text-to-Image
```
Mode: text_to_image
Use Custom System Prompt: False (applies default formatting)
Token Removal: auto
```

#### With Template Builder
```
Template Builder → QwenVLTextEncoder
Use Custom System Prompt: True (uses Template Builder output)
Mode: image_edit
Optimize Resolution: True
```

#### Direct Prompt (No System Prompt)
```
Mode: image_edit  
Use Custom System Prompt: True (but pass raw text)
Your text: Full formatted prompt with <|im_start|> tags
```

### QwenLowresFixNode

**Parameters:**
- `upscale_factor`: 1.0-4.0 (typically 1.5)
- `denoise`: 0.3-0.7 for refinement
- `steps`: Will use half for stage 2

**Workflow:**
1. Connect after initial KSampler
2. Use same positive/negative conditioning
3. Connect VAE for encode/decode
4. Output goes to VAE Decode

### Multi-Reference Image Support

**Single Image (default):**
```
LoadImage → QwenVLTextEncoder.edit_image
```

**Multiple Images via QwenMultiReferenceHandler:**

**Parameters:**
- **Resize Modes:**
  - `keep_proportion` (default): Maintains aspect ratio with padding
  - `stretch`: Force fit to dimensions
  - `resize`: Match average dimensions  
  - `pad`: Add black borders
  - `pad_edge`: Repeat edge pixels
  - `crop`: Center crop to smallest
- **Upscale Methods:**
  - `nearest-exact` (default): Preserves pixels exactly
  - `bilinear`: Smooth interpolation
  - `area`: Good for downscaling
  - `bicubic`: Higher quality  
  - `lanczos`: Best quality
- **Combination Methods:**
  - `index`: Keep images separate for position-based referencing
  - `offset`: Weighted average blending
  - `concat`: Side-by-side combination
  - `grid`: 2x2 layout

**Connection:**
```
QwenMultiReferenceHandler.multi_reference → QwenVLTextEncoder.multi_reference
```

**Example - Style Transfer:**
```
LoadImage (content) → Handler.image1
LoadImage (style)   → Handler.image2
                     Method: index
                     → QwenVLTextEncoder.multi_reference
Prompt: "Apply the style of the second image to the first"
```

## Advanced Workflow Configuration

### Performance Optimizations (from qe_node_00059_.json)

This workflow uses several advanced techniques for better quality and performance:

1. **Model Pipeline:**
   - Base model: `qwen_image_edit_fp8_e4m3fn.safetensors` (FP8 quantized)
   - Lightning LoRA: `Qwen-Image-Edit-Lightning-4steps-V1.0` (4-step generation)
   - TorchCompile: Inductor backend for faster inference
   - SAGE Attention: Memory-efficient attention mechanism
   - CFG Zero: Classifier-free guidance optimization

2. **Conditioning Pipeline:**
   - Template Builder → QwenVLTextEncoder → ConditioningZeroOut
   - Proper reference latents through VAE encoding
   - Template styles for different edit modes

3. **Sampling Configuration:**
   - Steps: 4 (with Lightning LoRA)
   - CFG: 1.0 (with CFG normalization)
   - Sampler: euler
   - Scheduler: simple
   - Full denoise for complete regeneration

## Workflow Tips

### Image Editing Best Practices

1. **Always connect VAE** to encoder for reference latents
2. **Use optimize_resolution** for better quality
3. **Template styles**:
   - Use `minimal_edit` for small changes
   - Use `creative` for artistic freedom
   - Use `photorealistic` for realistic edits

### Critical: Latent Input Choice

This determines whether you're editing structure or using vision for semantic reimagining:

#### **Option 1: VAE Encode → KSampler (Structure Preservation)**
```
LoadImage → VAEEncode → KSampler.latent_image
```
- **Purpose**: Preserve composition, modify details
- **Denoise**: 0.3-0.7 (lower = more preservation)
- **Template**: `minimal_edit` 
- **Result**: Original structure with requested changes
- **Example**: "Change car color from red to blue"

#### **Option 2: Empty Latent → KSampler (Semantic Reimagining)**
```
EmptyLatentImage → KSampler.latent_image
```
- **Purpose**: Complete transformation using vision understanding
- **Denoise**: 0.9-1.0 (always high for generation from scratch)
- **Template**: `creative` or `default`
- **Result**: New image guided by vision tokens and prompt
- **Example**: "Transform into cyberpunk style"

**Important**: Reference latents (via VAE in encoder) are ALWAYS passed through conditioning regardless of which latent input you choose. They provide guidance, not structure.

### Denoise Settings Guide

**With VAE Encoded Input:**
- 0.3-0.5: Minimal changes, strong preservation
- 0.5-0.7: Moderate changes, some preservation
- 0.7-0.9: Significant changes, minimal preservation

**With Empty Latent:**
- Always use 0.9-1.0 (generating from scratch)
- Vision tokens guide the generation

### Two-Stage Refinement

**When to Use:**
- Final production renders
- Need maximum quality
- Have time for extra processing

**Settings:**
- Stage 1: Full generation
- Upscale: 1.5x typical
- Stage 2 Denoise: 0.5 typical

## Technical Details

### What ComfyUI Already Had
- Reference latents infrastructure
- Basic VAE encoding for edits
- Token handling system

### What We Fixed
- RoPE batch processing bug
- Template application
- Token removal options
- Resolution optimization

### What We Added
- Multiple template styles
- Flux-style reference methods
- DiffSynth resolution list
- Two-stage refinement node
- Debug mode for troubleshooting

## Comparison with Native ComfyUI

| Feature | ComfyUI Native | Our Enhanced |
|---------|---------------|--------------|
| Reference Latents | ✅ Basic | ✅ + Methods |
| Templates | ❌ | ✅ Multiple Styles |
| RoPE Fix | ❌ | ✅ |
| Optimal Resolution | ❌ | ✅ |
| Lowres Fix | ❌ | ✅ |
| Debug Mode | ❌ | ✅ |

## Migration Guide

### From Old Nodes
Simply replace:
- `QwenVLTextEncoder` → Works the same, more options
- Add `QwenLowresFixNode` for quality boost

### From ComfyUI Native
Replace:
- `TextEncodeQwenImageEdit` → `QwenVLTextEncoder`
- Get template styles and resolution optimization

## Troubleshooting

### Enable Debug Mode
Set `debug_mode: True` to see:
- Image processing steps
- Resolution changes
- Token counts
- Reference latent info

### Common Issues

**No reference latents:**
- Connect VAE to encoder
- Check debug output

**Wrong resolution:**
- Enable `optimize_resolution`
- Check QWEN_RESOLUTIONS list

**Template not applied:**
- Check `apply_template: True`
- Verify `template_style` selection

## Future Enhancements

### Not Yet Implemented
- Context Control (DiffSynth-specific, no ComfyUI support)
- EliGen V2 (Advanced entity control)
- Qwen2VLProcessor (Would require deeper integration)

These require architectural changes beyond node wrappers.