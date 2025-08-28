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
  - `qwen_text_to_image.json` - Pure text-to-image generation
  - `qwen_image_edit_semantic.json` - Semantic-aware editing (high denoise)
  - `qwen_image_edit_structure.json` - Structure-preserving edit (low denoise)
- `Documentation/` - Technical documentation and insights
  - [ISSUES.md](Documentation/ISSUES.md) - Detailed issue tracking with root causes and solutions

## Key Technical Documentation

**Investigation & Debugging:**
- [INITIAL_ANALYSIS.md](Documentation/INITIAL_ANALYSIS.md) - Original investigation into architecture issues
- [ENCODER_INVESTIGATION.md](Documentation/ENCODER_INVESTIGATION.md) - Discovery of random noise bug
- [ISSUES.md](Documentation/ISSUES.md) - Comprehensive issue tracking with solutions

**Implementation Guides:**
- [WORKFLOW_EXPLANATION.md](Documentation/WORKFLOW_EXPLANATION.md) - Understanding image edit workflows
- [CUSTOM_VS_NATIVE_RATIONALE.md](Documentation/CUSTOM_VS_NATIVE_RATIONALE.md) - When to use our nodes vs native
- [Implementation Comparison](Documentation/IMPLEMENTATION_COMPARISON.md) - Feature comparison
- [Complete Node Tutorial](Documentation/COMPLETE_NODE_TUTORIAL.md) - Node usage guide
- [Reference Latents Explained](Documentation/REFERENCE_LATENTS_EXPLAINED.md) - Edit mode details
- [FUTURE_ENHANCEMENTS.md](Documentation/FUTURE_ENHANCEMENTS.md) - Potential improvements from DiffSynth

## Current Implementation (December 2024)

### Qwen Image Edit Nodes
ComfyUI nodes for the Qwen Image Edit model and Qwen2.5-VL text encoder. Does *NOT* use Wan.
1. Loads Qwen2.5-VL models from `models/text_encoders/` folder
2. Properly handles vision tokens for image editing
3. Builds on ComfyUI's internal CLIP infrastructure for compatibility
4. Includes fixes from DiffSynth-Studio and DiffSynth-Engine

**Core Nodes:**
- `QwenVLCLIPLoader` - Loads Qwen2.5-VL model with RoPE fix applied
- `QwenVLTextEncoder` - Text encoder with:
  - Default Qwen formatting or custom Template Builder input
  - Resolution optimization with aspect ratio preservation (36 Qwen resolutions)
  - Optional multi-reference input accepts `QwenMultiReferenceHandler` output
  - Debug mode for troubleshooting
- `QwenLowresFixNode` - Two-stage generation for quality improvement
- `QwenMultiReferenceHandler` - Combines up to 4 images with:
  - Six resize modes: keep_proportion, stretch, resize, pad, pad_edge, crop
  - Five upscale methods: nearest-exact, bilinear, area, bicubic, lanczos
  - Four combination methods: index, offset, concat, grid
  - Weighted blending for offset method

**Helper Nodes:**
- `QwenVLEmptyLatent` - Creates empty 16-channel latents
- `QwenVLImageToLatent` - Converts images to 16-channel latents

**Resolution Utilities:**
- `QwenOptimalResolution` - Auto-resize images to nearest Qwen resolution
- `QwenResolutionSelector` - Dropdown selector for Qwen resolutions

#### Workflow Usage

**Text-to-Image:**
1. `QwenVLCLIPLoader` → `QwenVLTextEncoder` (mode: text_to_image) → KSampler
2. Use `QwenVLEmptyLatent` or `EmptyLatentImage` for latent input
3. Denoise: 1.0

**Image Editing (Correct Approach):**
1. `LoadImage` → `QwenOptimalResolution` → `QwenVLTextEncoder` (edit_image input)
2. Set mode to `image_edit` 
3. Image information passes through **conditioning**, not latent
4. For latent input to KSampler, either:
   - `VAEEncode` the resized image (low denoise 0.3-0.7 for structure preservation)
   - `QwenVLEmptyLatent` (high denoise 0.9-1.0 for semantic-aware generation)
5. **Important**: High denoise (0.9-1.0) lets the model use its vision understanding. Low denoise defeats the purpose of vision tokens.

### Qwen Image Edit Architecture
- Uses **Qwen2.5-VL 7B** as text encoder (3584 dim embeddings)
- **16-channel VAE latents**
- Vision tokens: `<|vision_start|><|image_pad|><|vision_end|>`
- DiffSynth-Studio templates for consistency

### Key Implementation Points
- ComfyUI calls all text encoders "CLIP" internally
- Uses `CLIPType.QWEN_IMAGE` for proper model loading
- **RoPE position embedding fix applied automatically**
- **Reference latents properly passed through conditioning**
- **Multiple template styles for different use cases**
- **Optimal resolution support for better quality**

### What's Fixed vs Native ComfyUI
- RoPE batch processing bug (via monkey patch)
- Template Builder for custom system prompts (removed duplicate from encoder)
- Resolution optimization with aspect ratio preservation
- Two-stage refinement (Lowres Fix)
- Debug mode for troubleshooting
- Multi-reference image support (up to 4 images) integrated into main encoder
- Removed redundant reference_method parameter (auto-determined from multi-ref)
