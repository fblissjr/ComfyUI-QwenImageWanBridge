# Changelog

## v2.7.1 - Experimental Face Cropping (QwenSmartCrop)

### Added (Experimental)

**QwenSmartCrop Node** - Automated face isolation for multi-image composition
- Multiple detection strategies: geometric, saliency-based, VLM-powered
- VLM mode uses Qwen3-VL via [shrug-prompter nodes](https://github.com/fblissjr/shrug-prompter)
- Zero-dependency fallback modes (saliency, geometric)
- `face_headshot` anchor mode (uses bbox width for tight face crops)
- Adjustable padding and square output options
- Auto-fallback strategy with graceful degradation
- Solves community-discovered "tight crop" technique for headshot changing tasks
- [Documentation](nodes/docs/QwenSmartCrop.md)

### Technical Details
- Qwen3-VL coordinate system support (0-1000 → converted to percentages)
  - ** Note - if you worked with Qwen2.5-VL, these coordinates for bbox's are different
- JSON bbox parsing with markdown fence handling
- Edge detection using PyTorch gradient computation
- Configurable VLM parameters (max_tokens, temperature, top_p)

## BREAKING CHANGE (v2.7.0+)

**Existing workflows will break.** You must:
1. Delete and re-add, or Right Click -> Recreate Template Builder, Encoder, and Encoder Advanced nodes from your workflow
2. Connect: Template Builder `template_output` → Encoder `template_output` (single connection only)

Old multi-connection system (mode + system_prompt) no longer works as it was getting convoluted and confusing. One connection from template builder now handles everything (prompt, mode, system_prompt).

**Updated Workflows Available** [here](example_workflows/nunchaku_qwen_image_edit_2509.json)
1. [Multi image edit workflow](example_workflows/nunchaku_qwen_image_edit_2509.json)
2. [Single image edit workflow](example_workflows/qwen_edit_2509_single_image_edit.json)

---

## v2.7.0 - Template System Refactor & Simplified Connection

### BREAKING CHANGES

**Simplified Template Connection**
- Template Builder now outputs single `template_output` (contains everything)
- Connect ONE output: Template Builder `template_output` → Encoder `template_output`
- Old system (separate mode + system_prompt connections) removed
- **You must recreate Template Builder and Encoder nodes in existing workflows**

### Changed

**File-Based Template System**
- Templates now stored as `.md` files in `nodes/templates/` directory (this now means you can make your own templates super easily)
- Single source of truth for all templates (no hardcoded dicts)
- YAML frontmatter for metadata (mode, vision, experimental flags)
- Easy to add/edit templates without code changes
- Shared `template_loader.py` used by all nodes (cached on load)

**Template Builder UI Enhancement**
- JavaScript extension now populates `custom_system` field when selecting presets
- Select a preset → see/edit the system prompt before using it
- Fixed widget name mismatch (was looking for `system_prompt`, now correctly uses `custom_system`)

### Technical Details
- `nodes/template_loader.py` - Shared loader with singleton pattern
- Templates use YAML frontmatter: `mode`, `vision`, `use_picture_format`, `experimental`, `no_template`
- Python node reads templates on module import (cached)
- JavaScript keeps hardcoded copy for UI responsiveness (synced manually)

## v2.6.2 - Multi-Image Batch Node

### Added

**QwenImageBatch Node** - Smart multi-image batching node to handle images with control over strategy and scaling
- Auto-detects up to 10 images (no manual inputcount)
- Skips empty inputs (no black images)
- Two batching strategies: `max_dimensions` (minimal distortion) and `first_image` (hero-driven)
- Automatic double-scaling prevention (marks images as pre-scaled, encoders skip VAE scaling)
- Enhanced debug logging (aspect ratios, scale factors, dimension adjustments)

**multi_image_edit Mode** - DiffSynth-compatible multi-reference encoding
- Vision tokens inside prompt (not before) - matches DiffSynth `encode_prompt_edit_multi`
- Automatic "Picture X:" labeling
- Available in both standard and advanced encoders

**Template Builder → Encoder Auto-Sync**
- Template Builder outputs `mode` (3rd output)
- Encoders accept mode as STRING input (can be connected or typed manually)
- Prevents vision token placement/drop index mismatches

### Fixed
- KJNodes ImageBatchMulti black image issue (empty inputs) - resolution was to create a new node that handles our use case exactly as needed
- Aspect ratio cropping in standard batch nodes
- Double-scaling when using batch node → encoder
- Multi-image template mode mismatch

### Technical Notes
- QwenImageBatch uses v2.6.1 scaling modes + batch strategy
- Metadata propagation: `qwen_pre_scaled`, `qwen_scaling_mode`, `qwen_batch_strategy`
- Vision encoder always scales (384×384 target), VAE skips if pre-scaled
- 32px alignment maintained throughout pipeline

## v2.6.1 - Resolution Scaling Fix

### Fixed
- **Zoom-out issue in image editing** - Large images were being aggressively downscaled
  - Previous behavior: 1477×2056 scaled to 864×1216 (0.59x, causing zoom-out)
  - New default: `preserve_resolution` keeps original dimensions with 32px alignment
  - Example: 1477×2056 → 1472×2048 (minimal crop, no zoom-out)

### Added
- **scaling_mode parameter** in QwenVLTextEncoder and QwenVLTextEncoderAdvanced with three modes:
  - `preserve_resolution` (default) - Keeps input size, only applies 32px alignment
  - `max_dimension_1024` - Scales largest side to 1024px (good for 4K images)
  - `area_1024` - Legacy behavior, scales to ~1024×1024 area
- Improved tooltips with detailed tradeoffs for each mode
- Advanced encoder applies scaling_mode as base, then applies resolution_mode weights on top

### Changed
- Vision encoder always uses 384×384 area scaling (unchanged)
- VAE encoder respects new scaling_mode setting
- Debug output shows which scaling mode is active

### Scaling Mode Comparison

**preserve_resolution (default, recommended for most use cases)**
- ✓ No zoom-out effect, subjects stay full-size
- ✓ Best quality output, minimal cropping (only 32px alignment)
- ✓ Works perfectly for typical image sizes (512px-2048px)
- ✗ May use significant VRAM with very large images (4K+)
- Use when: You want maximum quality and your images are under 2500px

**max_dimension_1024 (recommended for 4K and very large images)**
- ✓ Reduces VRAM usage significantly on large images
- ✓ Balanced quality vs performance tradeoff
- ✓ Prevents OOM errors on 4K images
- ✗ Some zoom-out effect on images larger than 1024px
- Use when: Working with 4K images or hitting VRAM limits

**area_1024 (legacy, not recommended)**
- ✓ Consistent ~1 megapixel output size
- ✗ Aggressive zoom-out on large images (major quality loss)
- ✗ Upscales small images unnecessarily (wastes quality)
- ✗ Poor behavior across different input sizes
- Use when: You need exact 1024×1024 area behavior for specific workflows

### Examples by Resolution

**Small portrait (512×768 = 0.4MP)**
- preserve: 512×768 (1.00x) - unchanged, perfect
- max_dimension: 672×1024 (1.31x) - upscaled
- area: 832×1248 (1.62x) - upscaled too much

**Large portrait (1477×2056 = 3MP, the reported issue)**
- preserve: 1472×2048 (1.00x) - no zoom-out, FIXED
- max_dimension: 736×1024 (0.50x) - some zoom-out
- area: 864×1216 (0.58x) - old broken behavior

**4K landscape (3840×2160 = 8MP)**
- preserve: 3840×2176 (1.00x) - may OOM on lower VRAM
- max_dimension: 1024×576 (0.27x) - RECOMMENDED for 4K
- area: 1376×768 (0.36x) - inconsistent scaling

## v2.6 - Mask-Based Inpainting

### Added
- **QwenMaskProcessor** - Mask preprocessing node
  - Base64 mask input from spatial editor
  - Blur, expand, feather controls
  - Preview overlays showing inpaint areas
- **QwenInpaintSampler** - Inpainting sampler node
  - Implements exact diffusers blending: `(1-mask)*original + mask*generated`
  - Strength control for partial/full regeneration
  - 4-channel to 16-channel auto-conversion
- **Inpainting mode** in QwenVLTextEncoder
  - New `inpaint_mask` optional parameter
  - Mask passed to conditioning for spatial control
- **Example workflows**
  - `qwen_edit_2509_mask_inpainting.json` - Standard workflow
  - `nunchaku_qwen_mask_inpainting.json` - Nunchaku variant

### Technical Details
- Follows DiffSynth-Studio mask-based approach (not coordinate tokens)
- Spatial tokens marked as experimental based on DiffSynth analysis
- JavaScript interface already supports both systems

## v2.5.1 - Reorganized example workflows
- Example workflows have been renamed and reorganized, with a new one as well for Nunchaku 2509

## v2.5 update - Wrapper Nodes (NOT WORKING YET - Experimental)

### Added
- **Wrapper Node System** - Independent implementation (11 nodes):
  - `QwenImageDiTLoaderWrapper` - Load Qwen DiT using transformers
  - `QwenVLTextEncoderLoaderWrapper` - Load Qwen2.5-VL using transformers
  - `QwenImageVAELoaderWrapper` - Load 16-channel VAE using diffusers/ComfyUI
  - `QwenModelManagerWrapper` - Unified pipeline loader
  - `QwenProcessorWrapper` - Process text/images with Qwen2VL processor
  - `QwenProcessedToEmbedding` - Convert processed tokens to conditioning
  - `QwenImageEncodeWrapper` - Encode images to edit latents (batch support via Image Batch node)
  - `QwenImageModelWrapper` - DiffSynth-style forward pass with 2x2 packing
  - `QwenImageSamplerNode` - FlowMatch sampler with built-in scheduling
  - `QwenImageModelWithEdit` - Inject edit latents into model
  - `QwenImageSamplerWithEdit` - Alternative sampler with edit support
  - `QwenDebugLatents` - Debug latent dimensions and flow

### Technical Implementation
- **No DiffSynth dependency** - Uses only transformers, diffusers, torch
- 2x2 patch packing operation following DiffSynth model_fn_qwen_image
- Edit latent concatenation in sequence dimension (not conditioning)
- FlowMatch scheduler with resolution-aware dynamic shift
- Automatic padding for odd dimensions to enable patch processing
- Direct edit latent injection bypassing ComfyUI's conditioning system
- ComfyUI's standard VAE loader for 16-channel VAE (auto-detects config)

### Fixed
- Removed all DiffSynth imports from wrapper loaders
- Fixed type mismatch: text encoder loader outputs `QWEN_TEXT_ENCODER` not `CLIP`
- Removed redundant `QwenImageCombineLatents` node (use Image Batch instead)
- Removed redundant `QwenSchedulerNode` (sampler has built-in scheduling)
- VAE loader now uses ComfyUI's auto-config detection

### Status
- **Not fully tested** - Wrapper nodes are complete but haven't been validated with actual models
- **Use Standard Nodes** - Recommended for production use (QwenVLCLIPLoader + QwenVLTextEncoder)

## v2.5 Power User Features & Auto-Labeling

### Added
- **QwenVLTextEncoderAdvanced** for power users
  - Per-image resolution weighting (hero/reference modes)
  - Memory budget management (VRAM limits)
  - Custom resolution targets (vision & VAE separate)
  - Progressive and memory-optimized modes
- **Configurable auto-labeling**
  - auto_label parameter (on/off)
  - label_format choice (Picture/Image)
- **Documentation**
  - Advanced encoder test guide (10 configurations)
  - Model limits (512 images, token constraints)
  - Power user guide

### Changed
- Auto-labeling now optional (was always on)
- Debug shows auto_label and label_format status
- Advanced encoder simplified - removed confusing validation_mode parameter
- Added verbose_log parameter to control console logging separately from UI debug
- Enhanced debug patch with comprehensive tracing (timestamps, memory stats, NaN/Inf detection)

### Fixed
- Wan21 latent format requires 5D tensors - now properly handled
- **Dimension mismatch auto-handling** - No more errors from mismatched resolutions!
  - Automatically pads latents to even dimensions for patch processing
  - Wraps ComfyUI's QwenImage model to handle any resolution
  - Users can now use any resolution, not just 1024x1024
- Clear debug messages showing dimension adjustments
- Documentation updated with technical explanation

## v2.4 DiffSynth Alignment & Better Debug

### Added
- **100% DiffSynth-Studio alignment** verified for all components
- **Better debug output** showing full prompts without truncation
- **New face replacement templates** aligned with Qwen-Image-Edit-2509:
  - `qwen_face_swap` - Simple face swap following model's training
  - `qwen_identity_merge` - Identity transfer with full scene preservation
- **Character counts** in debug output for tracking token usage

### Fixed
- Face replacement templates now generate full images, not just face crops
- Debug output shows complete formatted text being encoded
- Templates use "generate an image" structure matching model training

### Changed
- Updated face replacement templates to preserve full scene context
- Debug output sections reorganized for better clarity

## v2.3 Debug Controller & Silent Patches

### Added
- **QwenDebugController node** - Comprehensive debugging system
  - Multi-level debugging: off/basic/verbose/trace
  - Performance profiling and memory tracking
  - Component filtering and regex pattern matching
  - Export debug sessions to JSON/text
  - System info display (GPU, CPU, memory)
- **Silent debug patches** - No console spam by default
  - Set `QWEN_DEBUG_VERBOSE=true` or use Debug Controller to enable
  - Integrates with encoder's debug_mode parameter

### Fixed
- Debug patches now run silently unless explicitly enabled
- No more console spam during normal operation

## v2.2 Token Dropping & N-Image Support

### Added
- **Proper token dropping implementation** matching DiffSynth/Diffusers behavior
  - System prompt included during encoding for context
  - First 34/64 embeddings dropped after encoding
  - Prevents text contamination while maintaining quality
- **N-image support** in Template Builder (0-100 images)
  - Automatic warnings at 4+ and 10+ images
  - 0 = auto-detect mode for future implementation
- **Simplified architecture** following DRY principles
  - Template Builder: Single source for system prompts
  - Processor: Minimal formatter (80 lines)
  - Encoder: Handles all technical encoding/dropping

### Fixed
- System prompts now provide proper context without appearing in images
- Token dropping happens after encoding (was missing entirely)

## v2.1 System Prompt Separation

### Fixed
- **System prompt appearing in generated images** - Template formatting issue
- **Template Builder and Encoder separation** - No more duplicated logic

## v2.0 Qwen-Image-Edit-2509 Multi-Image Support

### Added
- **Full Qwen-Image-Edit-2509 support** with native multi-image editing capabilities
- **"Picture X:" reference formatting** for addressing specific images in prompts
- **Automatic detection** of Picture references in prompts for smart formatting
- **Template system enhancements**:
  - `custom_system` field works as override for ANY template mode
  - `show_all_prompts` mode displays all available system prompts
  - Uses exact templates from DiffSynth-Studio repository
- **Example workflows** with comprehensive documentation
  - Multi-image 2509 workflow with detailed notes
  - Text-to-image workflow with proper connections

### Fixed
- **Dimension mismatch in native_multi mode** causing VAE latent size errors
- **QwenMultiReferenceHandler** now enforces uniform dimensions automatically
- **Template system corrections** using proper DiffSynth-Studio prompts
- **Workflow parameter alignment** in example JSON files

### Changed
- **Multi-image workflow**: Use ComfyUI's Image Batch node for multiple images
- **Template consistency**: All templates use official DiffSynth-Studio system prompts
- **Dimension handling**: Automatic fixes prevent model forward pass errors

## v1.6+ Experimental Features (Archived)

### Spatial Token Generator
- Visual spatial editing interface with region drawing
- Multiple output formats (JSON, XML, natural language, traditional tokens)
- Coordinate system handling and validation
- **Status**: Experimental, low priority - spatial tokens effectiveness unclear

### EliGen Entity Control
- Entity-level spatial generation with masks
- Up to 4 regions with individual prompts and weights
- Based on DiffSynth implementation
- **Status**: Experimental, untested with current models

### Earlier Versions
- Multi-reference image processing improvements
- Resolution optimization and aspect ratio handling
- Debug mode enhancements
- Template builder separation from encoder

## Current Working State

### Production Ready
- QwenVLTextEncoder with proper template separation
- QwenTemplateBuilder with DiffSynth-Studio templates
- Multi-image support via Image Batch node
- Dimension-aware multi-reference processing
- Debug modes for troubleshooting

### Experimental
- Spatial token generation (effectiveness unclear)
- Entity control nodes (untested)
- Advanced spatial editing features

### Removed/Archived
- WAN keyframe nodes
- Video-to-video nodes
- Dual encoding architecture (overcomplicated)
- Native implementation attempts (wrapper approach preferred)
- Inpainting nodes (broken, low priority)
