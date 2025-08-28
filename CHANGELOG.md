# Changelog

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

### Fixed
- RoPE (Rotary Position Embeddings) batch processing bug
  - Applied monkey patch from DiffSynth-Studio
  - Fixes issues with different image sizes in batch
- Resolution optimizer selecting wrong aspect ratios
  - Now considers both pixel count and aspect ratio
- Parameter order breaking backward compatibility
  - Moved new parameters to end of function signature
- Double image scaling causing quality degradation
  - Removed redundant scaling in workflows

### Technical Details
- Integrated fixes from DiffSynth-Studio and DiffSynth-Engine
- Maintained compatibility with ComfyUI's CLIP infrastructure
- Reference latents passed through conditioning metadata
- Support for 36 official Qwen resolutions
- 16-channel VAE latent support throughout

### Documentation
- Created REFERENCE_METHODS_EXPLAINED.md for multi-reference usage
- Added decision guides for common use cases
