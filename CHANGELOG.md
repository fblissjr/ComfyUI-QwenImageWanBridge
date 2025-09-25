# Changelog

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