*Despite the name, this is becoming more of a Qwen Image Edit repo.*

# Overview
ComfyUI nodes for Qwen2.5-VL and Qwen Image Edit models with enhanced template control.

## Core Nodes

### Text Encoding
- **QwenVLCLIPLoader**: Loads Qwen2.5-VL models from `models/text_encoders/` folder
- **QwenVLTextEncoder**: Enhanced encoder with flexible template control
  - `apply_template`: Enable/disable system templates
  - `token_removal`: Choose between auto (ComfyUI), diffsynth (fixed), or none
  - `mode`: Text-to-image or image_edit (defaults to image_edit)

### Template Building
- **QwenTemplateBuilder**: Interactive template builder with presets
  - 9 presets including DiffSynth defaults, artistic, photorealistic, minimal edit
  - Custom template mode with system prompt control
  - Automatic vision token handling for image editing
- **QwenTemplateBuilderV2**: Simplified Python-only version
  - All options visible, no JavaScript required
  - custom_t2i and custom_edit modes
- **QwenTemplateConnector**: Bridges template builder to encoder

### Helper Nodes
- **QwenOptimalResolution**: Auto-resizes images to nearest Qwen-preferred resolution
- **QwenResolutionSelector**: Dropdown selector for Qwen resolutions
- **QwenTokenInfo**: Reference display for special tokens
