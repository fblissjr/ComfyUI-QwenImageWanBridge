# QwenSmartCrop (Experimental)

**Category:** QwenImage/Experimental
**Status:** Experimental

Intelligent headshot/subject cropping with multiple detection strategies for automated headshot isolation in multi-image workflows.

## Purpose

Automates the "headshot cropping" technique discovered by the community for better headshot changes and composition results. Instead of manually cropping, this node uses multiple detection strategies to find and extract headshots automatically.

## Detection Modes

### 1. center_square
**Simple geometric center crop**
- Crops largest square from image center
- Zero dependencies, always works
- Best for: Centered compositions

### 2. portrait_auto
**Portrait-aware heuristic cropping**
- Portrait orientation: Upper 60% (assumes face at top)
- Landscape orientation: Center square
- Uses photography composition rules
- Best for: Standard portrait photos

### 3. saliency_crop
**Edge/variance-based detection**
- Detects regions with high detail (edges, texture)
- Assumes faces have more detail than backgrounds
- Pure PyTorch implementation (no ML models)
- Best for: Varied compositions where face isn't centered

**How it works:**
1. Converts to grayscale
2. Computes gradient magnitude (Sobel-like edge detection)
3. Finds regions with most edges
4. Expands to bounding box with padding
5. Makes square if requested

### 4. vlm_detect
**VLM-based intelligent detection**
- Uses Qwen3-VL via shrug-prompter API
- Model understands faces semantically
- Asks VLM: "Where is the face?" → Gets bbox coordinates
- Best for: Complex scenes, unusual compositions

**Requirements:**
- VLM_CONTEXT from shrug-prompter VLMProviderConfig node
- API server running (heylookitsanllm or similar)
- Custom prompt optional

### 5. auto_fallback (Recommended)
**Tries multiple strategies with fallback**
1. Try VLM detection (if context available)
2. Fall back to saliency if VLM fails
3. Ultimate fallback to geometric if all else fails

## Parameters

### Required

**image** (IMAGE)
- Input image to crop

**detection_mode** (dropdown)
- Choose detection strategy (see above)
- Default: `auto_fallback`

**padding** (FLOAT: 0.0-1.0)
- Padding around detected region
- 0.2 = 20% expansion beyond detected bounds
- Default: 0.2

**output_square** (BOOLEAN)
- Force output to square aspect ratio
- Recommended: True (for face crops)
- Default: True

**min_crop_size** (INT: 64-2048)
- Minimum crop dimension in pixels
- Prevents too-small crops
- Default: 256

### Optional

**vlm_context** (VLM_CONTEXT)
- Required for `vlm_detect` and `auto_fallback` modes
- Connect from shrug-prompter's VLMProviderConfig node
- Provides API access to Qwen3-VL

**vlm_prompt** (STRING)
- Custom prompt for VLM detection
- Default: "Detect the face in this image. Respond with only the bounding box coordinates in format: x1,y1,x2,y2 where values are percentages (0-100) of image width/height."
- Multiline, customizable

**debug_mode** (BOOLEAN)
- Output debug visualization
- Shows detection process
- Default: False

## Outputs

**cropped_image** (IMAGE)
- Cropped image ready for batching

**info** (STRING)
- Crop information and statistics
- Includes: mode used, dimensions, coordinates

**debug_viz** (IMAGE)
- Debug visualization (if debug_mode=True)
- Currently same as cropped_image

## Workflow Examples

### Basic Face Crop for Swapping

```
LoadImage (portrait) → QwenSmartCrop (detection_mode: saliency_crop)
                           ↓
                    Tight face crop
                           ↓
                    QwenImageBatch (with scene image)
                           ↓
                    QwenVLTextEncoder (multi_image_edit)
                    "Person from image 2 with face from image 1"
```

### VLM-Powered Detection

```
VLMProviderConfig ──────────┐
                            │
LoadImage → QwenSmartCrop ──┘ (detection_mode: vlm_detect)
                ↓
         Intelligent face crop
                ↓
         QwenImageBatch → ...
```

### Multi-Strategy Fallback

```
VLMProviderConfig (optional) ──┐
                               │
LoadImage → QwenSmartCrop ─────┘ (detection_mode: auto_fallback)
                ↓
         Tries VLM → Saliency → Geometric
                ↓
         Best available crop
```

## Implementation Details

### Zero-Dependency Modes

**center_square, portrait_auto, saliency_crop:**
- Pure PyTorch operations
- No external models or APIs
- Always available
- Fast execution

### VLM Mode

**vlm_detect:**
- Requires shrug-prompter API server
- Uses Qwen3-VL's vision understanding (or your VLM of choice)
- Parses bbox response: "x1,y1,x2,y2" percentages
- Falls back gracefully on failure

**API Call Pattern:**
```python
# Expected VLM context interface:
context.call(
    images=[base64_image],
    prompt=vlm_prompt,
    max_tokens=100,
    temperature=0.1
)
```

### Saliency Detection Algorithm

```
1. Convert RGB → Grayscale
2. Compute gradients (Sobel-like):
   dy = |gray[y+1] - gray[y]|
   dx = |gray[x+1] - gray[x]|
3. Create edge magnitude map
4. Threshold at 70th percentile
5. Find bounding box of high-edge regions
6. Expand by padding ratio
7. Make square if requested
8. Clamp to image bounds
```

## Known Limitations

### Current State
- **Experimental** - Expect things to break or not work or the feature to be pulled completely if not useful
- **VLM integration incomplete** - You're on your own on this one unless you use all the software mentioned, and even then there's work to do

### Detection Accuracy
- Saliency: Works best when face has clear edges (may fail on soft-lit portraits)
- VLM: Requires correct API setup, prone to parsing errors if response format varies
- Geometric: Assumes standard compositions (fails on off-center subjects)

## Technical Notes

### VLM Response Parsing

Expected response format:
```
"The face is located at: 25,15,75,85"
or
"x1=25, y1=15, x2=75, y2=85"
```

Parser looks for 4 numbers in range 0-100 (percentages of image dimensions).

### Coordinate Systems

- Input: ComfyUI IMAGE tensor [B,H,W,C]
- VLM: Percentages (0-100) of width/height
- Output: Cropped IMAGE tensor [B,H',W',C]

## Related Nodes

- **QwenImageBatch**: Batch cropped faces with scenes
- **QwenVLTextEncoder**: Multi-image composition
- **shrug-prompter VLMProviderConfig**: VLM API access
- **QwenMaskProcessor**: Alternative mask-based approach
