# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code and Writing Style Guidelines

- **No emojis** in code, display names, or documentation
- Keep all naming and display text professional
- Avoid "Pure", "Enhanced", "Advanced", "Ultimate" type prefixes - use descriptive names instead
- Always avoid redundancy and unnecessary complexity. If you need to make a v2, there needs to be a compelling reason for it instead of simply modifying the code or creating a new git branch.
- Clean, simple node names that describe what they do
- Keep descriptions minimal and factual

## Organization

- `nodes/` - Production-ready nodes only
- `nodes/archive/` - Legacy and experimental nodes
- `example_workflows/` - Example JSON workflows with comprehensive notes
- `internal/` - Internal documentation and analysis

## Current Implementation

### Core Nodes

**QwenVLTextEncoder**
- Main text encoder with DiffSynth/Diffusers reference alignment
- 32-pixel resolution alignment for proper vision processing
- Multi-image support via ComfyUI's Image Batch node
- Automatic "Picture X:" formatting for Qwen-Image-Edit-2509
- Separate system_prompt input for template customization
- Debug mode for troubleshooting

**QwenTemplateBuilder**
- System prompt generation with DiffSynth-Studio templates
- Outputs separate prompt and system_prompt for clean separation
- custom_system field to override any template's system prompt
- show_all_prompts mode to view available templates

**QwenMultiReferenceHandler**
- Multi-image processor supporting up to 4 images
- native_multi mode for Qwen-Image-Edit-2509 (uniform dimensions required)
- Various composite modes: concat, grid, offset for other use cases
- Automatic dimension matching to prevent VAE latent errors

**QwenEliGenEntityControl** (Experimental)
- Entity-level spatial generation with masks
- Up to 4 regions with individual prompts and weights
- Based on DiffSynth EliGen implementation

### Key Features

**System Prompt Separation (Latest Fix)**
- Template Builder and Encoder now have clear separation of responsibilities
- Template Builder outputs raw prompt and system_prompt separately
- Encoder handles all formatting internally
- Fixed issue where system prompt text was appearing in generated images
- No more duplicated template logic between nodes

**Qwen-Image-Edit-2509 Support**
- Multi-image support with "Picture 1:", "Picture 2:" formatting
- Automatic detection and formatting when using Picture references
- Use ComfyUI's Image Batch node for multiple images
- Template system uses official DiffSynth-Studio prompts

**Template System**
- custom_system field works as override for ANY template mode
- show_all_prompts mode displays all available system prompts
- Uses exact templates from DiffSynth-Studio repository
- Clean separation between prompt content and system instruction

**Multi-Image Processing**
- QwenMultiReferenceHandler with native_multi mode for 2509
- Automatic dimension matching prevents VAE latent errors
- Forces uniform dimensions when needed

### Current Working State

**What's Working:**
- Text-to-image generation
- Single and multi-image editing with Qwen-Image-Edit-2509
- Template system with custom system prompts
- Debug mode for troubleshooting
- Multi-reference image processing with dimension fixes

**What's Experimental:**
- EliGen Entity Control (untested with current models)
- Spatial Token Generator (low priority, experimental)

### Basic Workflows

**Text-to-Image:**
```
QwenTemplateBuilder → QwenVLTextEncoder → KSampler
```

**Image Edit:**
```
LoadImage → QwenVLTextEncoder (edit_image) → KSampler
QwenTemplateBuilder → QwenVLTextEncoder (system_prompt)
```

**Multi-Image Edit:**
```
LoadImage (×N) → Image Batch → QwenVLTextEncoder → KSampler
QwenTemplateBuilder → QwenVLTextEncoder (system_prompt)
```
