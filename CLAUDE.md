# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code and Writing Style Guidelines

- **No emojis** in code, display names, or documentation
- Keep all naming and display text professional
- Avoid "Pure", "Enhanced", "Advanced", "Ultimate" type prefixes - use descriptive names instead
- Clean, simple node names that describe what they do
- Keep descriptions minimal and factual

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

## Organization

- `nodes/` - Production-ready nodes only
- `nodes/archive/` - Legacy and experimental nodes
- `example_workflows/` - Example JSON workflows
- `Documentation/` - Technical documentation and insights
  - [ISSUES.md](Documentation/ISSUES.md) - Detailed issue tracking with root causes and solutions

## Key Technical Documentation

**Investigation & Debugging:**
- [INITIAL_ANALYSIS.md](Documentation/INITIAL_ANALYSIS.md) - Original investigation into architecture issues
- [ENCODER_INVESTIGATION.md](Documentation/ENCODER_INVESTIGATION.md) - Discovery of random noise bug
- [ISSUES.md](Documentation/ISSUES.md) - Comprehensive issue tracking with solutions

**Implementation Guides:**
- [CUSTOM_VS_NATIVE_RATIONALE.md](Documentation/CUSTOM_VS_NATIVE_RATIONALE.md) - When to use our nodes vs native
- [Implementation Comparison](Documentation/IMPLEMENTATION_COMPARISON.md) - Feature comparison
- [Complete Node Tutorial](Documentation/COMPLETE_NODE_TUTORIAL.md) - Node usage guide
- [Reference Latents Explained](Documentation/REFERENCE_LATENTS_EXPLAINED.md) - Edit mode details
- [FUTURE_ENHANCEMENTS.md](Documentation/FUTURE_ENHANCEMENTS.md) - Potential improvements from DiffSynth

## Current Implementation (Simplified)

### Qwen Image Edit Nodes
ComfyUI nodes for the Qwen Image Edit model and Qwen2.5-VL text encoder. Does *NOT* use Wan.
1. Loads Qwen2.5-VL models from `models/text_encoders/` folder
2. Properly handles vision tokens for image editing
3. Builds on ComfyUI's internal CLIP infrastructure for compatibility

**Core Nodes:**
- `QwenVLCLIPLoader` - Loads Qwen2.5-VL model as CLIP
- `QwenVLTextEncoder` - Encodes text/images with proper vision token support

**Helper Nodes:**
- `QwenVLEmptyLatent` - Creates empty 16-channel latents
- `QwenVLImageToLatent` - Converts images to 16-channel latents

**Resolution Utilities:**
- `QwenOptimalResolution` - Auto-resize images to nearest Qwen resolution
- `QwenResolutionSelector` - Dropdown selector for Qwen resolutions

#### Workflow Usage
1. Place Qwen2.5-VL model in `models/text_encoders/`
2. Use `QwenVLCLIPLoader` to load the model
3. Use `QwenVLTextEncoder` with:
   - `text_to_image` mode for text-only generation
   - `image_edit` mode with an input image for editing

### Qwen Image Edit Architecture
- Uses **Qwen2.5-VL 7B** as text encoder (3584 dim embeddings)
- **16-channel VAE latents**
- Vision tokens: `<|vision_start|><|image_pad|><|vision_end|>`
- DiffSynth-Studio templates for consistency

### Key Implementation Points
- ComfyUI calls all text encoders "CLIP" internally
- Uses `CLIPType.QWEN_IMAGE` for proper model loading
- Templates are applied automatically by ComfyUI's tokenizer
- Reference latents added for image editing mode
