# Changelog

## v1.4.1 - Pushing out EOD, not fully tested, expecting bugs/issues

### Added
- **QwenSpatialTokenGenerator** node with visual spatial editing interface
  *note:* expect quirks here, especially with object references and polygons, but it's a start!
  - Interactive "Open Spatial Editor" with canvas drawing
  - Support for bounding boxes, polygons, and object references that correspond to qwen image tokens, so no more manual input
  - Auto-generation of spatial tokens with proper formatting
  - Debug mode
- Auto-token generation when drawing regions in spatial editor

### Potential issues
- Downstream images / upstream images loading correctly
  - Fixed coordinate parsing for different region types
  - Proper handling of object references vs bounding boxes vs polygons
- Clipboard functionality in spatial editor with browser compatibility fallback
- Template integration properly including spatial tokens in formatted output
- Debug output visibility toggle in spatial editor interface

## v1.3

### Added
- Resize modes for `QwenMultiReferenceHandler` to prevent image distortion
  - `match_first`: resize all to image1 dimensions (original behavior)
  - `common_height`: same height, preserve aspect ratios (recommended for concat)
  - `common_width`: same width, preserve aspect ratios
  - `largest_dims`: use largest dimensions found across images

### Changed
- Simplified `QwenVLTextEncoder` by removing unnecessary resolution interface
  - Removed confusing resolution controls that didn't affect generation
  - Node now focuses on text encoding and vision processing
  - Returns clean `(conditioning,)` output like standard ComfyUI text encoders
  - Resolution controlled by proper nodes (Empty Latent, Multi-Reference Handler)

### Fixed
- Multi-reference concat mode squishing images with different aspect ratios
  - Added smart resize logic before concatenation
  - Aspect ratio preservation prevents distortion
  - Maintains backward compatibility with existing workflows
- Grid mode tensor size mismatch when using aspect-preserving resize modes
  - Grid now uses uniform dimensions to prevent concatenation errors
  - Calculates average aspect ratio for balanced proportions

## v1.2

### Added
- Documentation for multi-reference spatial ordering (MULTI_REFERENCE_SPATIAL_ORDERING.md)
  - Explains how Qwen interprets concatenated images spatially
  - Provides clear guidance on prompt phrasing
  - Visual layout references for concat and grid methods

### Changed
- Improved multi-reference handler tooltips
  - Clarified that image1 appears LEFT in concat, TOP-LEFT in grid
  - Updated reference_method tooltip to recommend index for "first/second" prompts
  - Made spatial positioning explicit in all image input tooltips
- Updated SCALE_IMAGE_VS_OPTIMIZE_RESOLUTION.md
  - Explains why Scale Image node is now redundant
  - Migration guide for existing workflows
- Enhanced REFERENCE_METHODS_EXPLAINED.md
  - Clarified limitations of current single-image implementation
  - Added future multi-reference possibilities

### Fixed
- Documentation clarity around parameter defaults
  - use_custom_system_prompt defaults to False (applies default formatting)
  - optimize_resolution defaults to False (standard 1M pixel scaling)
  - upscale_method defaults to nearest-exact (preserves pixels)

## v1.1

### Added
- Multi-reference image support integrated into main `QwenVLTextEncoder`
  - Optional `multi_reference` input accepts output from `QwenMultiReferenceHandler`
  - Single encoder node now handles both single and multiple reference images
  - Backward compatible - existing workflows unchanged
- `QwenMultiReferenceHandler` node with:
  - Up to 4 reference images
  - Six resize modes: keep_proportion, stretch, resize, pad, pad_edge, crop
  - Five upscale methods: nearest-exact, bilinear, area, bicubic, lanczos
  - Four combination methods: index, offset, concat, grid
  - Weighted blending for offset method
- Template Builder nodes for custom system prompts
  - Separated from encoder to avoid duplication
  - Multiple style presets available in Template Builder
- Resolution optimization with aspect ratio preservation
  - Fixed bug where only total pixels were considered
  - Now maintains aspect ratio when selecting optimal resolution
- Two-stage refinement via `QwenLowresFixNode`
  - Upscales and refines generated images
  - Configurable denoise strength
- Debug mode for all encoder nodes
  - Detailed logging of dimensions, tokens, and processing steps

### Changed
- Renamed `apply_template` to `use_custom_system_prompt` (default False)
  - False (default): Apply default Qwen formatting automatically
  - True: Use pre-formatted text from Template Builder
- Removed `system_prompt_style` from encoder (use Template Builder instead)
- Removed redundant `reference_method` parameter from encoder (auto-determined from multi-ref data)
- Improved parameter tooltips with plain English descriptions
- Reference method tooltips now explain each option clearly
- Removed redundant Scale Image node from workflows (optimizer handles this internally)
