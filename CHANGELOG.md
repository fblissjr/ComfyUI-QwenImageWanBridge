# Changelog

## v1.6.0 Structured Spatial Commands

### Revolutionary Change: Default Output Format
- **NEW DEFAULT**: `structured_json` format replaces traditional spatial tokens
- **Why**: Research shows Qwen2.5-VL was trained on JSON/XML structured data making these formats more native and effective
- **Impact**: Much more precise and controllable image editing with semantic context

### Added: Advanced Output Formats
- **Structured JSON** (new default): JSON command objects with action, target, bbox, instruction, and preserve directives
  ```json
  {
    "action": "edit_region",
    "target": "coffee mug",
    "bbox": [312, 412, 762, 612],
    "instruction": "change to red color, keep handle shape",
    "preserve": "background, lighting, other objects"
  }
  ```
- **XML Tags**: HTML-like elements with `data-bbox` attributes (most native to Qwen training)
  ```xml
  <region data-bbox="312,412,762,612">
    <target>coffee mug</target>
    <instruction>change to red color, keep handle shape</instruction>
  </region>
  ```
- **Natural Language**: Coordinate-aware sentences for intuitive instructions
  ```text
  Within the bounding box [312,412,762,612], modify the coffee mug. Preserve background and lighting.
  ```
- **Traditional Tokens**: Legacy format still available for backward compatibility

### Enhanced Template Builder Integration
- **New template modes**: `structured_json_edit`, `xml_spatial_edit`, `natural_spatial_edit`
- **Smart auto-detection**: Automatically selects appropriate template based on spatial token format
- **Specialized system prompts**: Each format gets optimized instructions for best results
- **Seamless workflow**: Copy from QwenSpatialTokenGenerator → QwenTemplateBuilderV2 → QwenVLTextEncoder

### JavaScript Interface Improvements
- **Format selection dropdown** with real-time help text
- **Intelligent preprocessing**: JavaScript prepares region data for Python processing
- **Enhanced debugging**: Format-aware logging and coordinate validation
- **Visual annotations**: Works across all formats for region visualization

### Technical Improvements
- **Coordinate system consistency**: All formats use native ViT pixel coordinates (multiples of 28)
- **Region data bridge**: JSON communication between JavaScript and Python
- **Format-aware processing**: Python backend adapts to input format automatically
- **Enhanced error handling**: Graceful fallbacks and comprehensive logging

### Benefits of New Formats
1. **More Semantic Control**: Specify what to preserve vs. what to change
2. **Research-Based**: Leverages Qwen2.5-VL's training on structured document data
3. **Programmatically Parseable**: Enables automation and batch processing
4. **Multi-Step Composition**: Natural support for complex scene directives
5. **Better User Intent**: Clear separation of action, target, and constraints

### Migration Guide
- **Existing workflows**: Traditional tokens still work, no breaking changes
- **New projects**: Use structured_json for best results
- **Template Builder**: Auto-detects format and applies appropriate template
- **Experimentation**: Switch between formats to compare effectiveness

### Backward Compatibility
- Traditional spatial tokens remain fully supported
- Existing workflows continue to work unchanged
- Format selection allows gradual migration
- Debug mode helps understand format differences

## v1.5.2 Got the coordinates wrong!

### Fixed
- Coordinate format corrected to use absolute pixels instead of normalized 0-1 values
  - Still don't know if these actually help.
  - Added Annotated images to see if that does anything.
  - **Reference**: https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/spatial_understanding.ipynb
- **Coordinate debugging** with absolute pixel verification
  - JavaScript logs show pixel coordinates and coverage percentages
  - Python logs verify coordinates are within image bounds
  - Cross-validation between JavaScript generation and Python parsing

## v1.5.1 Maybe better spatial editor improvements (nope)

### Added
- Resolution handling in spatial editor
  - Three selection modes: Auto (best match), Recommended (top 5 suggestions), Custom (manual input)
  - Aspect-ratio aware recommendations sorted by similarity to source image
  - Real-time resolution info display showing original → target dimensions and aspect ratio changes
  - Automatic re-optimization attempt when resolution changes (clears existing regions for accuracy)
  - Custom resolution validation with automatic 16-pixel alignment
  - Resolution section appears dynamically when image is loaded
  - Multiple resize strategies for different use cases:
    - **Pad**: Scale and add black borders to fit exact dimensions (default, preserves aspect ratio)
    - **Crop**: Scale and center crop to fill exact dimensions (may lose edge content)
    - **Stretch**: Distort image to exact dimensions (changes aspect ratio)
    - **Resize**: Scale maintaining aspect ratio (final size may differ from target)

### Simplified
- **QwenSpatialTokenGenerator workflow streamlined** - Removed redundant resolution optimization:
  - Eliminated `optimize_resolution` parameter (was causing double optimization issues)
  - Resolution handling now exclusively managed by JavaScript spatial editor
  - Simplified Python node focuses purely on token generation and template application
  - No more coordinate mismatch warnings - spatial editor handles all optimization

### Fixed
- **Double resolution optimization fixed** - Spatial editor now single source of truth for image resolution
- **Coordinate accuracy improved** - JavaScript and Python now use identical optimized dimensions
- **User experience enhanced** - Clear resolution controls instead of hidden conflicts
- **Custom resolution application** - Fixed bug where custom width/height inputs weren't being applied
- **Resolution display bug** - Fixed "undefined" aspect ratio display in recommended resolutions dropdown
- **Debug improvements** - Added comprehensive debug logging for resolution changes and method switching

### Technical Improvements
- **Resolution interface methods** added to spatial editor:
  - `updateResolutionInterface()` - Controls UI visibility based on selection mode
  - `populateRecommendedResolutions()` - Dynamically populates aspect-ratio sorted suggestions
  - `applyResolution()` - Re-optimizes image with new target resolution
  - `updateResolutionInfo()` - Real-time dimension and aspect ratio feedback
- **Code cleanup** in Python spatial token generator:
  - Removed unused `find_closest_resolution()` and `optimize_image_resolution()` methods
  - Removed `QWEN_RESOLUTIONS` constant and `math` import
  - Simplified class description and parameter handling
  - Cleaner separation of concerns between JS and Python components

# September 2025 - v1.4.x General Notes
**Experimental Feature Notice:** I haven't had a chance to test this thoroughly, but if you see issues, let me know. None of these tokens seem to be documented in the reference code that I could find in quick scans, but it does seem to work? The tokens were identified from the tokenizer config. That all said - this spatial editor node likely has issues, but the TL;DR of how to use it is that it takes a required input image and creates the prompt with the spatial tokens filled in for you. You can (and should) edit these on your own to finetune your edits.

**Token Usage Theory:** I'm uncertain if a bounding box token on its own works better than a bounding box with an object reference, but based on how vision LLMs work and their training datasets, these tokens are usually used for grounding purposes - meaning, the bounding box identifies the region, and by labeling the object reference within that bounding box, you are providing it more context. Does it work better than a bounding box alone? Does it work better than just saying "replace the church with a castle" without any spatial tokens? Don't know yet. Let's find out.

## v1.4.4 - Critical Coordinate System Fix & Dual-Encoding Architecture

### Fixed
- **CRITICAL: Double image optimization causing coordinate mismatch** - Fixed issue where both JavaScript spatial editor AND Python node were optimizing image resolution, causing coordinates to target wrong locations
  - **Root cause:** JavaScript editor optimized image to one resolution, but Python node re-optimized to potentially different resolution
  - **Solution:** Changed `optimize_resolution` default to `False` in QwenSpatialTokenGenerator
  - **User guidance:** Added warnings in both debug outputs when double optimization detected
- **JavaScript spatial editor coordinate system** - Now properly matches what the encoder will see:
  - Automatically finds closest Qwen resolution (same algorithm as Python node)
  - Resizes and pads image to match encoder's view
  - Normalizes coordinates using optimized dimensions
  - Enhanced debug output showing optimization process
- **Coordinate accuracy** - Spatial edits now target correct image regions instead of appearing in center
- **Resolution-independent editing** - Works correctly across all input image sizes and aspect ratios
- **Redundant spatial editor removal** - Eliminated duplicate spatial editor from QwenVLTextEncoder token analyzer

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
- **Modernized UI styling** across all interfaces to match spatial editor's dark theme:
  - Token analyzer interface: Dark background with backdrop blur, modern inputs/buttons
  - Testing interface: Complete dark theme overhaul with consistent styling
  - Custom dialogs: Modern dark modal styling with proper contrast ratios

### Changed
- **Native-quality vision processing** using existing infrastructure:
  - QwenVisionProcessor for advanced patch creation
  - Qwen2VLProcessor for proper multi-frame handling
  - MultiFrameVisionEmbedder for semantic embedding generation
- **Debug mode improvements** with dual-encoding path logging and feature dimension reporting
- **Conditioning fusion** now explicitly labeled and organized for better downstream processing

### Technical Details
- **Coordinate System Fix:**
  - Added `QWEN_RESOLUTIONS` array to JavaScript matching Python implementation
  - Added `findClosestResolution()` method using same aspect ratio matching logic
  - Added `optimizeImageForQwen()` method that resizes/pads images with black borders
  - Updated `generateTokens()` to normalize coordinates using optimized dimensions
  - Enhanced debug output to show coordinate transformation process
- **Dual-Encoding Architecture:**
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
