# Web Directory

JavaScript extensions for ComfyUI custom nodes.

## Files

### `js/template_autofill.js`

Unified template auto-fill extension that updates widget fields when template presets are selected.

**Supported Nodes:**
- `ZImageTextEncoder` - Full Z-Image encoder
- `ZImageTextEncoderSimple` - Simplified Z-Image encoder
- `HunyuanVideoTextEncoder` - HunyuanVideo encoder

**How It Works:**
1. Fetches template data from Python API (`/api/z_image_templates`)
2. Caches templates per-encoder type
3. When user selects a template preset:
   - Fills `system_prompt` with template body
   - Fills `add_think_block`, `thinking_content`, `assistant_content` if template specifies them
4. User can edit any field after template selection

**Key Features:**
- Single source of truth (templates in Python, JS just displays)
- Backward compatible with legacy string templates
- Preserves user customizations on workflow load
- Per-encoder caching to avoid duplicate API calls

## Developer Documentation

See `internal/COMFYUI_JAVASCRIPT_PYTHON_DEVELOPER_GUIDE.md` for:
- Architecture patterns
- How to add new encoder support
- Common pitfalls and best practices
- Code review checklist

## Template Format (v2.9.10+)

Templates can include extended fields in YAML frontmatter:

```yaml
---
name: z_image_json_structured
description: Parse JSON-structured prompts
model: z-image
category: structured
add_think_block: true
thinking_content: |
  Parsing the JSON structure...
---
System prompt body here...
```

When selected, JS fills all configured fields. User can edit before generation.
