# Changelog

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
