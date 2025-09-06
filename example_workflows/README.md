# Qwen-Image-Edit Workflow Examples

This directory contains example workflows demonstrating the Qwen-Image-Edit nodes.

## 🔴 CRITICAL: Understanding Latent Input Choice

### The Most Important Decision in Image Editing

When doing image editing, you have TWO fundamentally different approaches based on what you connect to KSampler's `latent_image` input:

#### **Option 1: VAE Encode Input Image → KSampler (Structure Preservation)**
```
LoadImage → VAEEncode → KSampler.latent_image
```
- **Purpose**: Keep composition, change details
- **Denoise**: 0.3-0.7 (lower = more preservation)
- **Template**: Use `minimal_edit`
- **Example**: "Change car color to blue" - keeps exact car shape/position
- **When to use**: Small edits, color changes, style tweaks

#### **Option 2: Empty Latent → KSampler (Vision-Guided Generation)**
```
EmptyLatentImage → KSampler.latent_image
```
- **Purpose**: Complete transformation using vision understanding
- **Denoise**: 0.9-1.0 (must be high - generating from scratch!)
- **Template**: Use `creative` or `default`
- **Example**: "Transform into cyberpunk style" - full reimagining
- **When to use**: Major transformations, style changes, creative edits

**KEY INSIGHT**: The reference latents (passed via VAE in QwenVLTextEncoder) provide vision guidance through conditioning REGARDLESS of which method you choose. They don't provide structure - the latent input does!

## Workflow Categories

### 0. Enhanced Workflow (NEW!)
- `qwen_enhanced_workflow.json` - **RECOMMENDED STARTING POINT**
  - All fixes from DiffSynth-Studio/Engine
  - RoPE batch fix applied
  - Template styles (creative/minimal_edit/photorealistic)
  - Optimal resolution support
  - Lightning LoRA 4-step generation
  - Clear notes on VAE Encode vs Empty Latent choice

### 1. Basic Workflows
- `qwen_t2i_basic.json` - Simple text-to-image generation
- `qwen_i2i_edit_basic.json` - Basic image editing with vision tokens
- `qwen_simple_node.json` - Using the simplified all-in-one node

### 2. Advanced Editing
- `qwen_autoregressive_edit.json` - Sequential editing with context
- `qwen_entity_control.json` - Regional generation with EliGen
- `qwen_prompt_interpolation.json` - Semantic blending between prompts

### 3. Sampler Integration
- `qwen_sampler_wrapper.json` - Using proper reference latents
- `qwen_sampler_advanced.json` - Advanced sampling with entity masks
- `qwen_iterative_sampler.json` - Progressive refinement

### 4. Helper Workflows
- `qwen_resolution_helper.json` - Finding optimal dimensions
- `qwen_vae_normalizer.json` - Applying proper VAE normalization
- `qwen_model_validator.json` - Validating model configuration

### 5. Experimental
- `qwen_wan_bridge.json` - Attempting Qwen→WAN (experimental)
- `qwen_controlnet.json` - BlockWise ControlNet integration

## Node Descriptions

### Core Nodes

**QwenImageEditTextEncoder**
- Replaces ComfyUI's broken Qwen text encoder
- Handles vision tokens properly
- Supports T2I, Edit, and Autoregressive modes
- Template dropping for better quality

**QwenImageEditUnified**
- All-in-one workflow node
- Combines all features in single interface
- Auto-detects mode based on inputs
- Includes ControlNet support

**QwenImageEditSimple**
- Simplified version for basic use
- Auto mode detection
- Minimal inputs required

### Feature Nodes

**QwenEliGenController**
- Entity-level generation control
- Up to 4 entities with masks
- Regional attention control

**QwenPromptInterpolator**
- Learned semantic blending
- Smooth transitions between prompts
- Training capability for custom interpolations

**QwenAutoregressiveEditor**
- Sequential editing chains
- Context-aware generation
- Edit history tracking

### Helper Nodes

**QwenImageResolutionHelper**
- Calculates optimal dimensions
- Based on target pixel count
- Ensures 32-pixel alignment

**QwenVAENormalizer**
- Applies official normalization
- 16-channel mean/std values
- Required for proper generation

**QwenModelValidator**
- Verifies token IDs
- Checks VAE configuration
- Validates model dimensions

## Quick Start

1. Load the Qwen-Image-Edit model
2. Use `QwenImageEditSimple` for basic generation
3. For editing: provide an image and use edit mode
4. For advanced features: use the unified node

## Important Notes

- Always use 16-channel VAE (not SD VAE)
- Vision tokens only work in edit mode
- Autoregressive editing builds on previous edits
- Entity masks should match entity prompt count
- Low denoise (0.3-0.5) preserves structure better