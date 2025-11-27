# HunyuanVideo 1.5 Nodes

Custom text encoding nodes for HunyuanVideo 1.5 with template system and proper embedding alignment.

---

## HunyuanVideoCLIPLoader

**Category:** `HunyuanVideo/Loaders`

Loads Qwen2.5-VL + optional byT5 for multilingual text rendering.

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `qwen_model` | dropdown | Qwen2.5-VL model file (NOT Qwen3) |
| `byt5_model` | dropdown | Optional byT5/Glyph model for multilingual text |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `clip` | CLIP | Combined encoder for HunyuanVideoTextEncoder |

### Notes

- Warns if you accidentally select Qwen3 (incompatible)
- byT5 auto-encodes quoted text like `"Hello World"` or `"Hallo Welt"`
- byT5 adds ~2-3GB VRAM - only enable if you need multilingual text rendering

---

## HunyuanVideoTextEncoder

**Category:** `HunyuanVideo/Encoding`

The main text encoder with 39 video templates and dual conditioning output. This is the primary node for HunyuanVideo 1.5 text-to-video workflows.

### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `clip` | CLIP | Yes | - | From HunyuanVideoCLIPLoader |
| `text` | STRING | No | "" | Positive prompt - describe what you want |
| `negative_prompt` | STRING | No | "low quality, blurry..." | Negative prompt - what to avoid |
| `template_input` | HUNYUAN_TEMPLATE | No | None | Optional dict with `system_prompt`, `prompt`, `template_name` keys |
| `template_preset` | dropdown | No | "none" | Select from 39 built-in video templates |
| `custom_system_prompt` | STRING | No | "" | Manual system prompt override |
| `additional_instructions` | STRING | No | "" | Extra instructions appended to any template |
| `debug_mode` | BOOLEAN | No | False | Show encoding details in debug_output |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `positive` | CONDITIONING | Positive conditioning for KSampler |
| `negative` | CONDITIONING | Negative conditioning for KSampler |
| `debug_output` | STRING | Debug info (when debug_mode=True) |

### How It Works

1. **System prompt determines video style** - The template/system prompt shapes how your text prompt is interpreted
2. **Dual output for CFG** - Both positive and negative use the same system template for consistent embedding space
3. **byT5 passthrough** - Quoted text (`"Hello"`) automatically triggers byT5 encoding for text rendering

### Input Priority

The system prompt is determined by this priority (first match wins):

```
1. template_input (if connected)     - External template source (dict input)
2. custom_system_prompt (if set)     - Manual override
3. template_preset dropdown          - Built-in templates
4. (none)                            - ComfyUI default behavior
```

### Additional Instructions

The `additional_instructions` input is **always appended** to whatever system prompt is used. This allows you to:
- Add style modifiers: `"always use noir lighting"`
- Focus on specific elements: `"emphasize hand movements"`
- Override template defaults: `"ignore any animation references, use realistic style"`

Example: Using `hunyuan_video_cinematic` template + additional_instructions `"focus on facial expressions, subtle emotional beats"`

### Examples

**Basic T2V (no template):**
```
text: "A cat sits on a windowsill watching birds outside. Soft afternoon light."
negative_prompt: "blurry, low quality, watermark"
template_preset: none
```

**With Cinematic Template:**
```
text: "A detective walks through rain-soaked streets at night"
template_preset: hunyuan_video_cinematic
additional_instructions: "film noir style, high contrast shadows"
```

**With Custom System Prompt:**
```
text: "Ocean waves crash on rocky shore"
custom_system_prompt: "You are a nature documentary cinematographer. Describe scenes with scientific accuracy and cinematic beauty. Focus on lighting, texture, and natural movement patterns."
```

**With byT5 Text Rendering:**
```
text: 'A neon sign flickers with the text "OPEN 24 HOURS" in a foggy alley'
template_preset: hunyuan_video_urban
```
Note: The quoted text triggers byT5 to render readable text in the video.

---

## Available Templates (39 total)

### Core Templates

| Template | Focus |
|----------|-------|
| `hunyuan_video_t2v` | Official HunyuanVideo format (5-aspect description) |
| `hunyuan_video_cinematic` | Dramatic narrative, professional cinematography |
| `hunyuan_video_animation` | Character design, motion principles, timing |

### Genre Templates

| Template | Focus |
|----------|-------|
| `hunyuan_video_action` | Fast-paced, dynamic movement |
| `hunyuan_video_horror` | Tension, atmosphere, dread |
| `hunyuan_video_comedy` | Timing, visual gags |
| `hunyuan_video_scifi` | Futuristic, technology |
| `hunyuan_video_fantasy` | Magical, otherworldly |

### Subject Templates

| Template | Focus |
|----------|-------|
| `hunyuan_video_nature` | Landscapes, natural phenomena |
| `hunyuan_video_wildlife` | Animal behavior, documentary style |
| `hunyuan_video_sports` | Athletic movement, competition |
| `hunyuan_video_urban` | City life, street scenes |
| `hunyuan_video_underwater` | Aquatic environments |
| `hunyuan_video_aerial` | Drone/bird's eye views |

### Production Templates

| Template | Focus |
|----------|-------|
| `hunyuan_video_product` | Product showcase, commercial |
| `hunyuan_video_commercial` | Advertising, brand content |
| `hunyuan_video_documentary` | Factual, observational |
| `hunyuan_video_educational` | Instructional, explanatory |
| `hunyuan_video_music` | Music video aesthetics |
| `hunyuan_video_interview` | Talking head, dialogue |

### Technical Templates

| Template | Focus |
|----------|-------|
| `hunyuan_video_timelapse` | Accelerated time |
| `hunyuan_video_slowmo` | Slow motion |
| `hunyuan_video_abstract` | Non-representational |

### Experimental Templates

| Template | Focus |
|----------|-------|
| `hunyuan_video_structured_realism` | Full structured format testing |
| `hunyuan_video_minimal_structure` | Minimal effective structure |
| `hunyuan_video_temporal_only` | Timeline-focused |
| `hunyuan_video_camera_focused` | Camera movement emphasis |
| `hunyuan_video_lighting_focused` | Lighting emphasis |
| `hunyuan_video_style_spam` | Style descriptor testing |
| `hunyuan_video_anti_pattern` | What NOT to do (baseline) |
| `hunyuan_video_self_expand` | LLM-style prompt expansion |

### Fun/Creative Templates

| Template | Focus |
|----------|-------|
| `hunyuan_video_drunk_cameraman` | Chaotic handheld |
| `hunyuan_video_80s_music_video` | Retro MTV aesthetic |
| `hunyuan_video_majestic_pigeon` | Dramatic mundane subjects |
| `hunyuan_video_wes_anderson_fever` | Symmetry, pastels, deadpan |
| `hunyuan_video_michael_bay_mundane` | Explosions for everyday things |
| `hunyuan_video_excited_dog_pov` | First-person dog perspective |
| `hunyuan_video_infomercial_disaster` | "But wait, there's more!" |
| `hunyuan_video_romcom_lighting` | Romantic comedy aesthetics |

---

## Debug Mode

Enable `debug_mode=True` to see:
- Template being used
- Character counts for prompts
- Embedding tensor shapes
- Full prompt text (positive, negative, system)
- byT5 trigger detection

Example debug output:
```
Template: hunyuan_video_cinematic
Additional instructions: 45 chars
Positive: 892 chars
Positive shape: torch.Size([1, 256, 3584])
Negative: 52 chars
Negative shape: torch.Size([1, 32, 3584])
Quoted text found - byT5 will encode
```

---

## Basic Workflow

```
HunyuanVideoCLIPLoader
        |
        | clip
        v
HunyuanVideoTextEncoder
        |
        +-- positive --> KSampler (positive input)
        |
        +-- negative --> KSampler (negative input)
                                |
                                | latent
                                v
                           VAEDecode
                                |
                                | IMAGE
                                v
                           SaveImage
```

---

## Model Requirements

- **Text encoder**: `Qwen2.5-VL-7B-Instruct` (NOT Qwen3)
- **byT5** (optional): `byt5-small` or `Glyph-SDXL-v2`
- **HunyuanVideo**: DiT transformer + VAE (ComfyUI native)

---

## Tips

1. **Start with templates** - Pick a template that matches your content type before writing custom prompts
2. **Use additional_instructions** - Layer modifications on top of templates instead of writing from scratch
3. **Check debug output** - Verify your template is being applied and embeddings look correct
4. **byT5 for text** - Put any text you want rendered in quotes
5. **Negative prompts matter** - Default negative covers common issues, but add specific things to avoid

---

## Model Downloads

- **Qwen2.5-VL:** huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- **byT5:** huggingface.co/google/byt5-small
- **Glyph-SDXL-v2:** huggingface.co/AI-ModelScope/Glyph-SDXL-v2

---

**Last Updated:** 2025-11-27
