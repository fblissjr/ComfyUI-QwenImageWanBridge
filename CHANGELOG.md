# Changelog

## v2.1 System Prompt Separation Fix

### Fixed
- **System prompt appearing in generated images** - Critical bug where template system prompts were being included in the generation prompt
- **Template Builder and Encoder separation**:
  - Template Builder now outputs raw `prompt` and `system_prompt` separately
  - Encoder handles all template formatting internally
  - No more duplicated logic between nodes
- **Workflow connections**: Updated example workflows to properly connect Template Builder outputs to Encoder inputs

### Technical Details
- Template Builder outputs: `prompt`, `system_prompt`, `mode_info`
- Encoder accepts `system_prompt` as separate input parameter
- System prompt is properly formatted with chat templates but separated from generation content
- Debug mode shows clear separation of system vs generation prompts

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