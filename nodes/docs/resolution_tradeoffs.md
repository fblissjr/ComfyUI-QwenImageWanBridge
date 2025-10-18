# Resolution Tradeoffs Guide

Comprehensive guide to resolution handling in Qwen nodes, covering single-image editing, multi-image batching, and scaling strategies.

## Single Image Edit - Resolution Tradeoffs

**Small images (<512px - e.g., 384x384, 256x512):**
- `preserve_resolution`: Keeps small (384x384 → 384x384)
- `max_dimension_1024`: Upscales (384x384 → 1024x1024)
- `area_1024`: Upscales (384x384 → 1024x1024)
- Tradeoff: Small = fast/low VRAM but may lack detail for model. Upscaling adds blur but gives model more pixels to work with

**Typical images (512px-2048px - e.g., 1024x1024, 1477x2056, 1328x1328):**
- `preserve_resolution`: Keeps original (1328x1328 → 1344x1344 with 32px alignment)
- `max_dimension_1024`: Downscales larger (1328x1328 → 1024x1024)
- `area_1024`: Downscales (1328x1328 → 1024x1024)
- Tradeoff: Larger = more detail/quality but higher VRAM. Downscaling reduces VRAM but loses some detail. We don't actually know if 1024x1024 produces better results than 1328x1328 or 2048x2048 - just that it uses less VRAM

**Large images (2048px-3000px - e.g., 2560x1440, 2048x2048):**
- `preserve_resolution`: Keeps large (2560x1440 → 2560x1440)
- `max_dimension_1024`: Downscales (2560x1440 → 1024x576)
- Tradeoff: preserve gives max quality if you have VRAM. max_dimension prevents OOM but you're downscaling significantly

**4K+ images (3840x2160, 4096x4096):**
- `preserve_resolution`: Keeps huge (3840x2176 with alignment)
- `max_dimension_1024`: Downscales heavily (3840x2160 → 1024x576)
- Tradeoff: preserve will likely OOM on most GPUs. max_dimension is basically required unless you have 48GB+ VRAM

**Odd dimensions (1023x1025, 1477x2056):**
- All modes apply 32px alignment (1023x1025 → 1024x1024, 1477x2056 → 1472x2048)
- Minor cropping, usually imperceptible
- Tradeoff: None really, alignment is required for VAE

## Multi-Image Edit - Resolution Tradeoffs

**Identical dimensions (1024x1024, 1024x1024, 1024x1024):**
- All strategies work identically
- Zero aspect distortion
- Most predictable, cleanest results
- Tradeoff: You need to pre-process images to match

**Same aspect ratio (512x768, 1024x1536, 768x512 - all 2:3):**
- `max_dimensions` scales to largest (1024x1536)
- Minimal distortion since aspect matches
- Clean batching
- Tradeoff: Still need to scale smaller images up (quality loss on upscale)

**Similar aspect ratios (portrait group: 3:4, 2:3, 9:16):**
- `max_dimensions`: Small adjustments, mostly manageable
- Some distortion but usually acceptable
- Tradeoff: Subjects may appear slightly stretched/squashed depending on how different the ratios are

**Mixed portrait + landscape (1024x768 + 768x1024):**
- `max_dimensions`: Forces one orientation to match the other
- Significant distortion inevitable
- `first_image`: Hero image preserved, others distorted heavily
- Tradeoff: You're basically choosing which images you're OK with distorting. Works but expect visible artifacts on the non-dominant orientation

**Extreme differences (512x512 + 3840x2160):**
- `max_dimensions`: Huge upscale on small image (quality loss)
- `first_image`: Depends on order - either massive downscale or massive upscale
- Tradeoff: One image will suffer quality loss regardless. Better to pre-scale before batching

**Very large gaps (512x512 + 1024x1024 + 2048x2048):**
- `max_dimensions`: Scales all to 2048x2048 (512 gets 4x upscale)
- Small images get blurry, large images OK
- Tradeoff: VRAM usage based on largest image, quality loss on smallest

## Scaling Mode Details

### preserve_resolution (default, recommended)
- Keeps original dimensions with 32px alignment
- No zoom-out effect, subjects stay full-size
- Best quality output, minimal cropping
- Works perfectly for typical image sizes (512px-2048px)
- May use significant VRAM with very large images (4K+)

### max_dimension_1024 (for 4K/large images)
- Scales largest side to 1024px
- Reduces VRAM usage significantly on large images
- Balanced quality vs performance tradeoff
- Prevents OOM errors on 4K images
- Some zoom-out effect on images larger than 1024px

### area_1024 (legacy, not recommended)
- Scales to ~1024x1024 area (~1 megapixel)
- Consistent output size
- Aggressive zoom-out on large images (major quality loss)
- Upscales small images unnecessarily (wastes quality)
- Poor behavior across different input sizes

### no_scaling (batch node only)
- Pass images through as-is
- Only applies 32px alignment
- Useful when you want full control over dimensions
- Recommended when using with advanced encoder (let it handle scaling)

## QwenImageBatch Parameters

**scaling_mode**: How to calculate target resolution for each image
- `preserve_resolution` (default): Keeps original size (32px aligned) - best quality basically
- `max_dimension_1024`: Scales largest side to 1024px - reduces VRAM
- `area_1024`: Scales to ~1024x1024 area - legacy behavior
- `no_scaling`: Pass through as-is

**batch_strategy**: How to handle different aspect ratios when batching
- `max_dimensions` (default): Uses max width/height across all images - minimal distortion ideally but still expect it as you add more images, though YMMV depending on aspect ratio diffs across inputs
- `first_image`: Forces all images to match first image's dimensions - likely to distort more but lets you control which one you care about, which theoretically should be complementary with the advanced encoder node for letting you weight as well. Basically the hero image wins.

## Example Scenarios

### 3 images all with same aspect ratios (1024x1024, 1024x1024, 1024x1024):
- All strategies work identically
- No aspect distortion regardless of choice

### Different aspect ratios (1024x1024, 1024x1024, 1328x1024):
- `max_dimensions`: Scales all to 1344x1024 (minimal distortion, but YMMV based on image resolution diffs across input images, image 3 stays around the same)
- `first_image`: Scales all to 1024x1024 (more distortion on image 3)

## When to Use What

**Default to `max_dimensions`**: Likely to get best quality typically
- Mixed aspect ratios but you want batching
- Best option for minimizing aspect distortion if that's important for your use case
- Works with all encoder nodes

**Use `first_image`**:
- You want your hero/main image to dictate all dimensions
- Acceptable if other images get slightly distorted
- Complementary with advanced node hero modes and weights but not necessary to use unless you want more control

**Single image editing**: Don't use QwenImageBatch - connect LoadImage directly to encoder's `edit_image` input. Batch node is only needed for 2+ images.

## Technical Notes

- Vision encoder always scales to 384x384 target area (unchangeable)
- VAE encoder respects scaling_mode setting
- 32px alignment required throughout pipeline for VAE compatibility
- Batch node marks images with metadata (`qwen_pre_scaled=True`) to prevent double-scaling in encoders
- Advanced encoder can apply additional resolution weighting on top of batch scaling
