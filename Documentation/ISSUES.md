# Issues and Resolutions

This document details the issues encountered during development, their root causes, and the solutions that fixed them.

## Issue 1: Random Noise Output Instead of Encoded Text

### Symptoms
- Generated images were pure noise regardless of prompt
- Conditioning tensors contained random values
- No correlation between text input and output

### Root Cause
In `nodes/qwen_vl_loader.py`, the `QwenVLModelWrapper.encode_text()` method was returning `torch.randn()`:
```python
def encode_text(self, text: str) -> torch.Tensor:
    # Create dummy embeddings of the right shape
    dummy_embeddings = torch.randn(batch_size, seq_len, embed_dim, 
                                  device=self.device, dtype=self.dtype)
    return dummy_embeddings  # THIS IS RANDOM NOISE!
```

This was a placeholder implementation that was never replaced with actual encoding logic.

### Discovery Process
1. Traced through the workflow to find where conditioning was created
2. Added debug logging to print tensor statistics (min/max/mean/std)
3. Noticed values were completely random each run
4. Found the `torch.randn()` call in the encoder

### Solution
Created new nodes (`QwenVLCLIPLoader` and `QwenVLTextEncoder`) that wrap ComfyUI's working CLIP infrastructure:
```python
# Use ComfyUI's internal CLIP loader with proper Qwen type
clip = comfy.sd.load_clip(
    ckpt_paths=[model_path],
    embedding_directory=folder_paths.get_folder_paths("embeddings"),
    clip_type=comfy.sd.CLIPType.QWEN_IMAGE
)
```

### Why This Works
- ComfyUI already has working Qwen support in `comfy/text_encoders/qwen_image.py`
- The CLIP infrastructure properly loads model weights and runs actual inference
- `CLIPType.QWEN_IMAGE` tells ComfyUI to use the Qwen-specific tokenizer and model

## Issue 2: Double Template Application Corrupting Prompts

### Symptoms
- Output images had "burnt" appearance (oversaturated, high contrast)
- Conditioning tensor values were abnormally high (>100 in some dimensions)
- Text prompts seemed to be ignored or misinterpreted

### Root Cause
We were applying the DiffSynth-Studio template twice:
1. First in our code when preparing the text
2. Second in ComfyUI's tokenizer which already applies the same template

This resulted in prompts like:
```
<|im_start|>system
Describe the image...
<|im_start|>system
Describe the image...
[actual prompt nested inside]
```

### Discovery Process
1. Added debug logging to print token counts and IDs
2. Noticed unusually high token counts for simple prompts
3. Examined ComfyUI's `qwen_image.py` tokenizer implementation
4. Found it already applies the exact DiffSynth templates in `tokenize_with_weights()`

### Solution
Removed our template application and let ComfyUI handle it:
```python
# Before (WRONG - double templating):
templated_text = self.llama_template.format(text)
tokens = clip.tokenize(templated_text, images=images)

# After (CORRECT - single templating by ComfyUI):
tokens = clip.tokenize(text, images=images)  # ComfyUI applies template internally
```

### Why This Works
- ComfyUI's tokenizer checks for images and applies appropriate template
- Templates are applied once at the correct layer
- Prompt structure remains intact for proper model interpretation

## Issue 3: Image Tensor Shape and Processing Mismatch

### Symptoms
- Error: `'Image' object has no attribute 'shape'`
- PIL Image objects being passed where tensors expected
- Dimension mismatches in VAE encoding

### Root Cause
Converting tensors to PIL Images broke the pipeline:
```python
# Wrong approach:
image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
tokens = clip.tokenize(text, images=[image_pil])  # Expects tensor!
```

### Discovery Process
1. Error traceback showed tokenizer expecting tensor operations
2. Examined official `TextEncodeQwenImageEdit` node
3. Found it keeps images as tensors throughout

### Solution
Keep images as tensors and match official node's processing:
```python
# Process image tensor like official node
samples = edit_image.movedim(-1, 1)  # [B, H, W, C] -> [B, C, H, W]

# Scale to target resolution (1024x1024 total pixels)
total = int(1024 * 1024)
scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
width = round(samples.shape[3] * scale_by)
height = round(samples.shape[2] * scale_by)

# Resize using ComfyUI utilities
s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
image = s.movedim(1, -1)  # [B, C, H, W] -> [B, H, W, C]

# Extract RGB channels
images = [image[:, :, :, :3]]
```

### Why This Works
- Maintains tensor format expected by tokenizer
- Preserves numerical precision (float32)
- Uses ComfyUI's optimized resize functions
- Matches exact processing of working official node

## Issue 4: Missing config.json When Loading Single Safetensors File

### Symptoms
- Error: "Model directory must contain config.json"
- Could not load models from single `.safetensors` files
- Required full model directory structure

### Root Cause
Initial implementation assumed transformers library format which requires:
```
model_directory/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── model.safetensors
```

But users wanted to load single safetensors files like other ComfyUI nodes.

### Discovery Process
1. User showed GGUF loader example that handles single files
2. Realized ComfyUI's CLIP loader can handle safetensors directly
3. Found ComfyUI abstracts away the config requirements

### Solution
Use ComfyUI's CLIP loader which handles single files:
```python
model_path = folder_paths.get_full_path("text_encoders", model_name)
clip = comfy.sd.load_clip(
    ckpt_paths=[model_path],  # Just pass the safetensors path
    embedding_directory=folder_paths.get_folder_paths("embeddings"),
    clip_type=comfy.sd.CLIPType.QWEN_IMAGE
)
```

### Why This Works
- ComfyUI's loader has embedded configs for known model types
- Safetensors format includes necessary metadata
- CLIP infrastructure handles model initialization internally

## Issue 5: Reference Latents Not Being Added Correctly

### Symptoms
- Image editing mode produced unrelated outputs
- Edit instructions ignored
- No connection between input image and output

### Root Cause
Reference latents were not being properly attached to conditioning:
```python
# Missing reference latent attachment
conditioning = clip.encode_from_tokens_scheduled(tokens)
# Reference latents never added!
```

### Discovery Process
1. Compared our node with official `TextEncodeQwenImageEdit`
2. Found it adds reference latents after encoding
3. Discovered `node_helpers.conditioning_set_values()` utility

### Solution
Add reference latents using ComfyUI's helper:
```python
if vae is not None and edit_image is not None:
    ref_latent = vae.encode(image[:, :, :, :3])
    conditioning = node_helpers.conditioning_set_values(
        conditioning, 
        {"reference_latents": [ref_latent]}, 
        append=True
    )
```

### Why This Works
- Reference latents provide spatial guidance for edits
- Attached as metadata to conditioning
- Sampler can access them during generation
- Maintains spatial coherence with input image

## Key Learnings

### 1. Always Check What ComfyUI Already Provides
- ComfyUI has extensive built-in support for many models
- Don't reinvent what already works
- Use internal infrastructure when possible

### 2. Debug with Concrete Data
- Log tensor statistics (shape, dtype, min/max, mean/std)
- Compare values between working and broken implementations
- "Burnt image" is not diagnostic - need numerical analysis

### 3. Match Official Node Patterns
- ComfyUI's official nodes are well-tested references
- Copy their tensor processing exactly
- Use same utility functions and helpers

### 4. Understand the Data Flow
- IMAGE format: `[B, H, W, C]` with values 0-1
- LATENT format: Model-specific channel counts and normalization
- CONDITIONING format: `[B, seq_len, embed_dim]` plus metadata dict

### 5. Test Incrementally
- Add debug logging at each step
- Verify tensor shapes and values match expected
- Compare with working reference implementation