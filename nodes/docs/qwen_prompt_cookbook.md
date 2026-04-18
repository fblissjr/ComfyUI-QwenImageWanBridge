# Qwen Image Edit 2509 Prompt Cookbook

Community-tested and official prompts for Qwen-Image-Edit-2509. These commands are **user prompts** (the actual editing instructions), not system prompts.

## Camera & Viewpoint Controls

### Rotation & Angles

**Side View (90°)**
- Chinese: `从侧面90度观看场景`
- English: "view the scene from the side at 90°"
- Effect: Rotates entire scene, not just objects

**Three-Quarters View**
- English: "make a three-quarters camera view of [subject] in image1"
- English: "make three-quarters camera view of [subject] in image1"
- Note: Works well with brief subject descriptions
- Example: "make a three-quarters camera view of close view of woman screaming in image1"
- Credit: Reddit user Striking-Long-2960

**Camera Rotation**
- Chinese: `相机视角向左旋转45度`
- English: "rotate camera viewpoint 45° to left"
- Note: Doesn't work in every picture

**Low-Angle Perspective**
- Chinese: `低角度视角`
- English: "low-angle perspective"
- Note: Not a true worm's-eye view, but effective

**Worm's-Eye / Upward View**
- Chinese: `仰视视角`
- Effect: Lower angle, but not true worm's-eye
- Note: Only works on some pictures

**Bird's-Eye View**
- Effect: Instant success (opposite of worm's-eye which struggles)

**Behind-The-Head Perspective**
- Chinese: `从某人头后方的视角`
- English: "from the perspective behind someone's head"
- Chinese Alt: `从背后视角`
- English Alt: "from a behind-the-back perspective"
- Chinese Alt 2: `背后视点`
- English Alt 2: "viewpoint from behind"
- Note: Very hit or miss - sometimes just turns person around, sometimes rotates entire scene

**POV Workaround (Two-Step)**
1. Generate with behind-the-head perspective
2. Re-edit with "remove the person"
3. Result: First-person perspective view

### Zoom & Framing

**Zoom Out**
- Chinese: `镜头拉远，显示整个场景`
- English: "zoom out the camera, show the whole scene"

## Scene Transformations

### Time of Day

**Day to Night / Night to Day**
- English: "change the scene to day time"
- English: "change the scene to night time"
- Note: Auto-generates shadows/lighting without extra prompting

### Weather

**Weather Changes**
- English: "change the weather to heavy rain"
- English: "change the weather to [condition]"

### Color Grading

**Black & White to Color**
- English: "Add colours to the scene"
- Note: Works excellently on old photos (1800s+)
- Effect: Also removes sepia/monochrome tones

**Sepia Tone**
- English: "Change the scene to sepia tone"
- Warning: May produce black and white instead

**Color Changes**
- English: "Change [object]'s [attribute] to [color]"
- Example: "Change the man's suit to green"
- Supports hex codes (partial): #00FF00 works for lime green
- Note: Not all hex codes work - use descriptive color names as fallback
- Example: "muted bluish-purple" works, but some hex codes fail

## Multi-Image Editing

### Person Replacement (Character Swap)

**Proven Formula (Reddit)**
- "Replace the person in image 1 with the person from image 2, while keeping the same pose, lighting, background, and outfit from image 1. Preserve the facial features and body proportions of the person from image 2."
- Note: From Reddit line 346 - most reliable pattern
- Warning: Very sensitive to wording changes

**Face Replacement (Experimental)**
- "Replace the face of the woman from image 2, with the face of the man from image 1"
- Note: Partially working, very random success rate
- Tip: Be precise and sharp with wording

### Multi-Image Combinations

**Official Patterns (from Qwen examples)**

1. **Person + Person → New Scene**
   - Chinese: `根据这图 1 中女性和图 2 中的男性，生成一组结婚照，并遵循以下描述：穿着中式红色婚服的两个人，在中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠，背景是传统的木质庭院，外面是蓝蓝的天空。光线明亮柔和，烘图观标，整体营庆并庄重。`
   - Effect: Generates wedding photos from two separate portraits

2. **Person + Scene → Placement**
   - Chinese: `生成一张图像：图 2 中的女生在图 1 躺椅上晒太阳`
   - English: "Generate image: woman from image 2 on beach chair from image 1 sunbathing"
   - Effect: Places person into scene naturally

3. **Person + Object → Product Showcase**
   - Chinese: `图 2 中的女生肩膀上挂着图 1 中的包`
   - English: "Woman from image 2 with bag from image 1 on shoulder"

4. **Person + ControlNet Keypoint**
   - Chinese: `图 2 中的女生改变为图 1 的姿势`
   - English: "Woman from image 2 changes to pose from image 1"
   - Note: Works with keypoint maps for pose transfer

### Three-Image Combinations

**Official Examples:**
- Wedding photos with custom bride/groom/scene
- Character placement in coffee shops
- Model showcase with person + product + scene

**Best Practices:**
- 1-3 images optimal (4+ may cause OOM)
- Reference images as "Picture 1:", "Picture 2:", etc. when using multi_image_edit mode
- Be explicit about which elements come from which image

## Portrait & Person Editing

### Style Changes

**Avatar Creator / ID Photo Styles**
- Chinese: `修改为蓝底证件照，人物穿上白色衬衫，黑色西装，打着条纹领带`
- English: "Blue background ID photo, person wearing white shirt, black suit, striped tie"
- Note: Excellent identity preservation across styles

### Pose Changes

**Hand Gestures**
- Chinese: `她双手举起，手掌朝向镜头，手指张开，做出一个俏皮的姿势`
- English: "Both hands raised, palms facing camera, fingers spread, making a playful pose"

**Heart Shape Gesture**
- Chinese: `她两只手摆出一个爱心的形状`
- English: "Both hands make heart shape"

**Holding Sign/Object**
- Chinese: `她两只手拿起一个黑板，上面写着"欢迎来到云栖大会"`
- English: "Both hands holding blackboard with text 'Welcome to Yunqi Conference'"

### Old Photo Restoration

**Restoration Prompt**
- Simple: "restore old photo" or "colorize and restore"
- Note: Model excels at maintaining identity while updating quality
- Works on very old photos (tested on 1800s portraits)

### Meme Creation

**Text Overlays**
- Leverages Qwen's text rendering capability
- Maintains identity while adding text
- Works with longer text passages

## Product & Object Editing

### Product Photography

**Product to Poster**
- Effect: Generates marketing posters from plain product images
- Example: Handbag on neutral background → styled model shoot

**Logo Generation**
- Input: Simple logo or product
- Effect: Creates branded marketing materials

### Text Editing

**Font Type Changes**
- Chinese example from official docs (slide 10)
- Effect: Maintains text content, changes typography

**Font Color Changes**
- Chinese example from official docs (slide 11)
- Effect: Recolors text while preserving layout

**Font Material Changes**
- Chinese example from official docs (slide 12)
- Effect: Applies textures/materials to text (metallic, wood, etc.)

**Text Content Editing**
- Supports precise text replacement
- Can combine with image editing (e.g., poster edits)

## Known Limitations

### What Doesn't Work Well

**Blur/Focus Issues**
- Cannot sharpen blurry images
- Cannot remove blur from out-of-focus subjects
- Foreground/background selective focus is difficult
- Prompts like "in sharp focus" or "crisp details" don't help

**True Worm's-Eye View**
- Model struggles with extreme low angles
- Low-angle perspective works, but not true worm's-eye

**Eye-Level from Low-Angle**
- Cannot reliably change from low angle to straight-on perspective
- Can do: low → lower, low → bird's-eye, low → side
- Cannot do: low → eye-level

### Prompt Sensitivity

**Face Swapping**
- Very inconsistent results
- Extremely sensitive to exact wording
- Better results: Use person replacement formula instead

**Chinese vs English**
- Chinese often provides finer control
- English works well for most commands
- Test both if results aren't satisfactory
- Use ChatGPT/LLM for Chinese translations (more accurate than Google Translate)

## Tips & Best Practices

### General Prompting

1. **Be Specific**: "Woman from image 2" not "the woman"
2. **Describe What You Want**: Not what you don't want (avoid pink elephant syndrome)
3. **Use Positive Prompts**: Put desired attributes in positive, undesired in negative
4. **Combine Creatively**: Reddit shows users mixing multiple commands successfully

### Resolution & Quality

1. **Recommended Settings**:
   - Resolution: 1024×1024 or 512×512
   - Steps: 8 (Lightning LoRA) or 20-30 (base)
   - CFG: 7.0-9.0
   - Denoise: 1.0 (T2I), 0.5-0.7 (edit)

2. **For Photorealism**:
   - Add: "Realistic, photorealistic, highly detailed"
   - Consider: lenovo.safetensors LoRA (reduces glossy/perfect look)

### Multi-Image Workflows

1. **Image Limits**: 1-3 optimal, 4+ risks OOM
2. **VRAM Relief**: Use `max_dimension_1024` scaling mode
3. **Batching**: Use QwenImageBatch for aspect-ratio preservation

### Experimentation

- Most commands require trial and error
- Same seed + different prompts for A/B testing
- ChatGPT can suggest prompt variations
- Community (Reddit) is actively discovering new patterns

## Sources

- Reddit: r/StableDiffusion - "Qwen Image Edit 2509 Helpful Commands"
- Official Qwen-Image-Edit-2509 announcement and examples
- Community testing and experimentation
- DiffSynth-Studio implementation patterns

## Face Replacement Analysis

Based on Reddit and official examples, face replacement patterns:

### What Works (Sort Of)

**Person Replacement** (Most Reliable)
- Full body/pose replacement WITH face
- Formula: "Replace the person in image 1 with the person from image 2, while keeping the same pose, lighting, background, and outfit from image 1. Preserve the facial features and body proportions of the person from image 2."
- Success rate: Variable but highest of all methods

**Face-Only Replacement** (Experimental)
- "Replace the face of [person A] from image 2, with the face of [person B] from image 1"
- Success rate: Very low, random
- Requires exact wording

### Why Face Swapping Is Hard

The model is trained for:
1. **Identity preservation** (keeping the same face)
2. **Style transfer** (changing everything BUT the face)
3. **Multi-image composition** (placing people in scenes)

Face swapping is the INVERSE of identity preservation, which explains poor results.

### Recommended Approach

Instead of face swapping, use:
1. **Multi-image person placement** for new scenes
2. **Pose transfer with keypoints** to change body position
3. **Style transfer** to change everything around the face
4. **Person replacement** to swap entire person+pose+face

### Automated Face Cropping (Experimental)

**QwenSmartCrop node** automates the tight face crop technique:

```
LoadImage (portrait) → QwenSmartCrop (detection_mode: saliency_crop)
                           ↓ (auto detects and crops face)
LoadImage (scene)      ────┼──→ QwenImageBatch
                               ↓
                        QwenVLTextEncoder (multi_image_edit)
                        "Person from image 2 with face from image 1"
```

**Detection modes:**
- `saliency_crop`: Edge detection (no dependencies, works well)
- `vlm_detect`: Uses Qwen2.5-VL for semantic understanding (requires shrug-prompter API)
- `auto_fallback`: Tries VLM → saliency → geometric (recommended)

**Benefits:**
- No manual cropping required
- Consistent crop sizes
- Multiple fallback strategies
- Adjustable padding (default 20%)

See [QwenSmartCrop documentation](QwenSmartCrop.md) for full details.

### Alternative Tools

For true face swapping, consider:
- Dedicated face swap models
- ComfyUI IPAdapter face workflows
- ReActor or similar face swap nodes
