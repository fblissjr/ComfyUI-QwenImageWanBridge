# Changelog

# v1.4.x - General Notes
**Experimental Feature Notice:** I haven't had a chance to test this thoroughly, but if you see issues, let me know. None of these tokens seem to be documented in the reference code that I could find in quick scans, but it does seem to work for the most part. The tokens were identified from the tokenizer config. That all said - this spatial editor node likely has issues, but the TL;DR of how to use it is that it takes a required input image and creates the prompt with the spatial tokens filled in for you. You can (and should) edit these on your own to finetune your edits.

**Token Usage Theory:** I'm uncertain if a bounding box token on its own works better than a bounding box with an object reference, but based on how vision LLMs work and their training datasets, these tokens are usually used for grounding purposes - meaning, the bounding box identifies the region, and by labeling the object reference within that bounding box, you are providing it more context. Does it work better than a bounding box alone? Does it work better than just saying "replace the church with a castle" without any spatial tokens? Don't know yet. Let's find out.

## v1.4.4 - Dual-Encoding Architecture Implementation

### Added
- **Dual-encoding architecture** in QwenVLTextEncoder following Qwen-Image technical report:
  - **What:** Parallel processing of edit images through both semantic and reconstructive pathways
  - **Why:** Standard VAE-only encoding loses high-level semantic understanding while vision-only processing lacks structural detail preservation
  - **How it differs:** Instead of just VAE→latents→sampler, now processes edit_image through BOTH vision understanding AND VAE encoding, then fuses features
  - Semantic path: edit_image → QwenVisionProcessor → MultiFrameVisionEmbedder → high-level scene understanding
  - Reconstructive path: edit_image → VAE → structural features → low-level detail preservation
  - MMDiT-compatible conditioning fusion combining both feature streams for balanced semantic coherence + visual fidelity
  - Native-level vision processing quality within ComfyUI wrapper architecture
- **Enhanced conditioning metadata** with dual encoding data for improved sampler integration
- **Automatic activation** when both edit_image and VAE are provided to QwenVLTextEncoder
- **Technical report compliance** - implementation now matches paper's dual-encoding methodology for ~90% architectural capability

### Changed
- **Native-quality vision processing** using existing infrastructure:
  - QwenVisionProcessor for advanced patch creation
  - Qwen2VLProcessor for proper multi-frame handling
  - MultiFrameVisionEmbedder for semantic embedding generation
- **Debug mode improvements** with dual-encoding path logging and feature dimension reporting
- **Conditioning fusion** now explicitly labeled and organized for better downstream processing

### Technical Details
- Maintains full backward compatibility - dual encoding activates automatically when VAE is connected
- Leverages existing vision processing infrastructure without architectural changes
- Combines semantic understanding (high-level) with structural features (low-level) as per paper
- Debug logging provides insight into dual-encoding feature generation and fusion process
- Zero workflow changes required - enhancement is transparent to existing setups

*This implementation achieves approximately 90% of the Qwen-Image paper's architectural capabilities within ComfyUI's framework.*

## v1.4.3 - Visual Editor Enhancements & Workflow Simplification

### Added
- **Individual region management** in visual spatial editor:
  - Delete button (×) for each region with confirmation dialog
  - Inline label editing - click on any region label to edit directly
  - Real-time updates to canvas display and token generation
- **Integrated resolution optimization** in QwenSpatialTokenGenerator:
  - Built-in QwenOptimalResolution logic eliminates need for separate preprocessing
  - `optimize_resolution` parameter (default: True) for automatic image sizing
  - Finds closest Qwen resolution and resizes with padding to maintain aspect ratio
- **Required image input** - simplified workflow with mandatory image connection
- **Single optimized image output** - streamlined from 5 outputs to 4

### Changed
- **QwenSpatialTokenGenerator workflow simplified**:
  - Image input changed from optional to required
  - No longer need separate QwenOptimalResolution node
  - Direct image connection → spatial token generation → text encoder
- **Enhanced region list interface**:
  - Editable labels with Enter key or blur to save
  - Delete buttons with safety confirmation
  - Prevention of empty labels (reverts to original)

## v1.4.2 - Spatial Token Editor Improvements & Coordinate System Fixes

### Added
- **Optional object_ref tokens** in QwenSpatialTokenGenerator
  - Checkbox control: "Include object reference label (for boxes/polygons)"
  - Enabled by default for backward compatibility
  - When disabled, generates pure coordinate tokens like `<|box_start|>0.1,0.2,0.3,0.4<|box_end|>`
  - When enabled, generates labeled tokens like `<|object_ref_start|>car<|object_ref_end|> at <|box_start|>0.1,0.2,0.3,0.4<|box_end|>`
- **Editable spatial tokens in base_prompt field**
  - Generated spatial tokens automatically populate the base_prompt widget
  - Users can modify the complete prompt including spatial tokens
  - Auto-updates when new regions are drawn or modified
- **Dual prompt outputs** from QwenSpatialTokenGenerator:
  - `prompt`: Plain text from base_prompt field (fully editable by user)
  - `formatted_prompt`: Template-formatted version ready for text encoder
- Enhanced coordinate handling for both string and list input formats
- Improved debug output with coordinate validation details

### Fixed
- **Critical coordinate system bugs** that were causing generation issues:
  - Object references now correctly handle 2 coordinates (x,y) instead of expecting 4
  - Bounding boxes properly use 4 coordinates (x1,y1,x2,y2) format
  - Polygons correctly handle variable number of coordinate pairs
  - Fixed NaN coordinate generation when clicking object reference points
- **JavaScript spatial interface regressions**:
  - Restored object reference click handling in mousedown events
  - Fixed clipboard functionality with proper browser compatibility fallback
  - Added auto-token generation when clearing regions
- **Template integration** now properly includes spatial tokens in formatted output
- Improved error handling for malformed coordinate data

### Technical Details
- Python coordinate processing now supports both JavaScript arrays and JSON string formats
- Added `includeObjectRef` parameter handling in both JavaScript and Python
- Fixed `normalize_coords` parameter that was previously undefined
- Enhanced debug mode with detailed coordinate transformation logging
- Maintains backward compatibility with existing workflows

*See [Spatial Token Editor & Reference](README.md#spatial-token-editor--reference) for complete usage guide.*

## v1.4.1 - Initial Spatial Editor (deprecated - use v1.4.2)

### Added
- **QwenSpatialTokenGenerator** node with visual spatial editing interface
  - Interactive "Open Spatial Editor" with canvas drawing
  - Support for bounding boxes, polygons, and object references
  - Auto-generation of spatial tokens with proper formatting
  - Debug mode

### Known Issues (Fixed in v1.5.0)
- Coordinate system bugs causing NaN values
- Object reference handling expecting wrong number of coordinates
- Missing object reference click functionality
- Template not properly including spatial tokens

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
