# QwenImageBatch

**Category:** QwenImage/Utilities
**Display Name:** Qwen Image Batch

## Description

Multi-image batching node that solves issues with standard batch nodes. Auto-detects up to 10 images, skips empty inputs (no black images), and preserves aspect ratios with configurable scaling and batching strategies.

## Key Features

- Auto-detects connected images (up to 10) - no manual inputcount parameter needed
- Skips None/empty inputs automatically (no black images)
- Preserves aspect ratios with minimal distortion
- Applies v2.6.1 scaling modes
- Prevents double-scaling via metadata propagation
- Enhanced debug logging with aspect ratio tracking
- Works with both standard and advanced encoders

## Inputs

### Required
- **image_1** (IMAGE)
  - First image (required)

### Optional
- **image_2** through **image_10** (IMAGE)
  - Additional images (auto-detected)
  - Only connected images are processed
  - Empty inputs automatically skipped

- **scaling_mode** (ENUM)
  - `preserve_resolution` (default): Keeps original size with 32px alignment - best quality
  - `max_dimension_1024`: Scales largest side to 1024px - reduces VRAM
  - `area_1024`: Scales to ~1024x1024 area - legacy behavior
  - `no_scaling`: Pass through as-is (only applies 32px alignment)
  - Default: `preserve_resolution`

- **batch_strategy** (ENUM)
  - `max_dimensions` (default): Uses max width/height across all images - minimal distortion
  - `first_image`: Forces all images to match first image's dimensions - hero-driven
  - Default: `max_dimensions`

- **debug_mode** (BOOLEAN)
  - Show batching details in console
  - Displays aspect ratios, scale factors, dimension adjustments
  - Default: False

## Outputs

- **images** (IMAGE)
  - Batched tensor with uniform dimensions
  - Marked with metadata to prevent double-scaling
  - Shape: `[batch_size, height, width, channels]`

- **count** (INT)
  - Number of images batched

- **info** (STRING)
  - Batching summary including:
    - Number of images
    - Scaling mode used
    - Batch strategy used
    - Individual image dimensions
    - Output shape
    - Warnings (if 4+ images or aspect ratio adjustments)

## How It Works

### Two-Pass Scaling Process

**Pass 1: Calculate ideal dimensions per image**
- Each image gets target dimensions based on `scaling_mode`
- Respects original aspect ratios
- Applies 32px alignment

**Pass 2: Unify dimensions based on batch_strategy**
- `max_dimensions`: Finds max width and max height across all images, scales all to those dimensions
- `first_image`: Uses first image's dimensions as target for all images

### Metadata Propagation

Marks batched tensor with attributes to prevent double-scaling:
```python
batched.qwen_pre_scaled = True
batched.qwen_scaling_mode = scaling_mode
batched.qwen_batch_strategy = batch_strategy
```

When encoders detect `qwen_pre_scaled=True`, they skip VAE scaling (vision encoder still scales to 384x384 as required).

## Example Usage

### Basic Multi-Image Edit
```
LoadImage ─┐
LoadImage ─┼─> QwenImageBatch (scaling_mode: preserve_resolution,
LoadImage ─┘                   batch_strategy: max_dimensions)
                     ↓
              QwenVLTextEncoder (mode: image_edit)
                     ↓
              KSampler → VAEDecode
```

### Hero Image with Advanced Encoder
```
LoadImage (hero) ─┐
LoadImage (ref1) ─┼─> QwenImageBatch (scaling_mode: preserve_resolution,
LoadImage (ref2) ─┘                   batch_strategy: first_image)
                            ↓
                  QwenVLTextEncoderAdvanced (resolution_mode: hero_first,
                                             hero_weight: 1.5)
                            ↓
                  KSampler → VAEDecode
```

### No Scaling (Let Advanced Encoder Handle It)
```
LoadImage ─┐
LoadImage ─┼─> QwenImageBatch (scaling_mode: no_scaling,
LoadImage ─┘                   batch_strategy: max_dimensions)
                     ↓
              QwenVLTextEncoderAdvanced (scaling_mode: preserve_resolution,
                                        resolution_mode: balanced)
                     ↓
              KSampler
```

## Batch Strategy Comparison

### max_dimensions (default, recommended)
**When to use:**
- Mixed aspect ratios but you want batching
- Want minimal aspect distortion
- General use case

**Behavior:**
- Calculates ideal dimensions per image
- Finds max width and max height across all
- Scales all images to those max dimensions

**Example:**
- Image 1: 1024x1024 → 1344x1024
- Image 2: 1024x1024 → 1344x1024
- Image 3: 1328x1024 → 1344x1024 (minimal scaling on this one)

### first_image (hero-driven)
**When to use:**
- You want your hero/main image to dictate all dimensions
- Acceptable if other images get distorted
- Complementary with advanced encoder hero modes

**Behavior:**
- Uses first image's calculated dimensions as target
- All other images scaled to match first image exactly
- Hero image preserved perfectly, others adjusted

**Example:**
- Image 1: 1024x1024 → 1024x1024 (preserved)
- Image 2: 1024x1024 → 1024x1024 (no change)
- Image 3: 1328x1024 → 1024x1024 (distorted to match hero)

## Scaling Mode Details

See [resolution_tradeoffs.md](resolution_tradeoffs.md) for comprehensive guide.

**preserve_resolution** (default):
- Best quality, no zoom-out
- Recommended for typical images (512px-2048px)
- May use more VRAM on very large images

**max_dimension_1024**:
- Reduces VRAM on large images
- Good for 4K images or VRAM constraints
- Some zoom-out on large images

**area_1024** (legacy):
- Consistent ~1MP output size
- Poor scaling behavior, not recommended

**no_scaling**:
- Only applies 32px alignment
- Useful with advanced encoder (let it handle scaling)

## Resolution Tradeoffs

### Identical dimensions (1024x1024, 1024x1024, 1024x1024)
- All strategies work identically
- Zero aspect distortion
- Most predictable results

### Same aspect ratio (512x768, 1024x1536, 768x512 - all 2:3)
- Minimal distortion
- Clean batching
- Smaller images scaled up (some quality loss)

### Similar aspect ratios (portrait group: 3:4, 2:3, 9:16)
- Small adjustments, mostly manageable
- Some distortion but usually acceptable

### Mixed portrait + landscape (1024x768 + 768x1024)
- Significant distortion inevitable
- Choose which images you're OK with distorting

### Extreme differences (512x512 + 3840x2160)
- One image will suffer quality loss
- Better to pre-scale before batching

See [resolution_tradeoffs.md](resolution_tradeoffs.md) for detailed analysis.

## Warnings

- **4+ images**: May cause VRAM issues (optimal: 1-3 images)
- **Aspect ratio adjustments**: Node warns when multiple unique aspect ratios detected
- **Double-scaling**: Automatically prevented via metadata, but ensure you're not manually scaling before this node

## Debug Mode Output

When `debug_mode=True`, shows in console:
```
[QwenImageBatch] Image 1: 1024x1024 (torch.Size([1, 1024, 1024, 3]))
[QwenImageBatch] Image 2: 1024x1024 (torch.Size([1, 1024, 1024, 3]))
[QwenImageBatch] Image 3: 1328x1024 (torch.Size([1, 1024, 1328, 3]))
[QwenImageBatch] Batch strategy: max_dimensions (all scaled to 1344x1024) (scaling_mode: preserve_resolution)
[QwenImageBatch]   Image 1: 1024x1024 -> 1344x1024 (1.31x, aspect adjusted (AR: 1.00 → 1.31, diff: 0.31))
[QwenImageBatch]   Image 2: 1024x1024 -> 1344x1024 (1.31x, aspect adjusted (AR: 1.00 → 1.31, diff: 0.31))
[QwenImageBatch]   Image 3: 1328x1024 -> 1344x1024 (1.01x, aspect adjusted (AR: 1.30 → 1.31, diff: 0.01))
[QwenImageBatch] Final batch shape: torch.Size([3, 1024, 1344, 3])
```

## Related Nodes

- QwenVLTextEncoder - Standard encoder (detects pre-scaled images)
- QwenVLTextEncoderAdvanced - Advanced encoder with resolution weighting
- QwenTemplateBuilder - System prompts for multi-image editing

## Technical Implementation

**File Location:** `nodes/qwen_image_batch.py`

**Key Methods:**
- `calculate_dimensions()`: Matches v2.6.1 scaling logic from encoders
- `batch_images()`: Two-pass scaling with auto-detection

**Dependencies:**
- `comfy.utils.common_upscale`: For bicubic scaling with disabled cropping

**Metadata Attributes:**
- `qwen_pre_scaled`: Boolean flag for encoders
- `qwen_scaling_mode`: Scaling mode used
- `qwen_batch_strategy`: Batch strategy used
