# Changelog

## v2.9.5 - Z-Image UX Redesign

### Breaking Changes

**ZImageTextEncoder Input Renamed**
- `text` input renamed to `user_prompt` for clarity
- Existing workflows need to reconnect the prompt input

**ZImageMessageChain Removed**
- Replaced by `ZImageTurnBuilder` with different design
- Old workflows using ZImageMessageChain will break

### Added

**ZImageTurnBuilder Node** - Higher-level abstraction for multi-turn conversations
- Each node represents a complete turn (user message + optional assistant response)
- `previous` input required (connects from encoder or another TurnBuilder)
- `user_prompt` - the user's message for this turn
- `thinking_content` / `assistant_content` - optional assistant response
- `is_final` flag controls whether last message gets `<|im_end|>` (default: True)
- `debug_output` shows turn details and formatted messages

**New Outputs on ZImageTextEncoder**
- `debug_output` - detailed breakdown (mode, char counts, token estimate)
- `conversation` - chain to ZImageTurnBuilder for multi-turn workflows

### Changed

**ZImageTextEncoder Field Order** (top to bottom)
1. `clip` (required)
2. `user_prompt` (renamed from `text`)
3. `conversation_override` (optional)
4. `template_preset`
5. `system_prompt`
6. `add_think_block`
7. `thinking_content`
8. `assistant_content`
9. `raw_prompt` (at bottom for advanced use)

**format_conversation() Enhanced**
- Now respects `is_final` flag in conversation dict
- If `is_final=True` (default): last message has no `<|im_end|>`
- If `is_final=False`: all messages get `<|im_end|>` (more turns expected)

### Workflow Changes

**Before (3+ nodes for system+user+assistant):**
```
ZImageMessageChain (system) -> ZImageMessageChain (user) -> ZImageMessageChain (assistant) -> ZImageTextEncoder
```

**After (single encoder handles first turn):**
```
ZImageTextEncoder (system+user+assistant) -> ZImageTurnBuilder (additional turn) -> ZImageTextEncoder
```

Most users only need ZImageTextEncoder - it handles a complete first turn.

---

## v2.9.4 - Z-Image Cleanup

### Fixed

**Critical: Shallow Copy Bug in ZImageMessageChain**
- `list()` creates new list but message dicts were still shared references
- Now uses `copy.deepcopy()` to prevent conversation corruption

**Medium: Empty dict handling for conversation_override**
- Empty dict `{}` was passing `is not None` check
- Now properly checks `conversation_override.get("messages")`

### Removed

**ZImageTextEncoderSimple**
- Redundant - same functionality as ZImageTextEncoder with `template_preset="none"`
- Use ZImageTextEncoder instead (matches diffusers by default)

### Changed

**Unified JS Template Auto-fill**
- Merged `z_image_encoder.js` and `hunyuan_video_encoder.js` into `template_autofill.js`
- Single module handles both encoder types with shared caching logic

**Consolidated Z-Image Documentation**
- Merged `z_image_nodes.md`, `z_image_workflow_guide.md` into `z_image_encoder.md`
- Single comprehensive guide covering nodes, workflows, troubleshooting, experiments
- Moved `z_image_analysis.md` to `internal/docs/` (private reference)

---

## v2.9.3 - Multi-Turn Conversation Support

### Added

**ZImageMessageChain Node** - Build multi-turn conversations for Z-Image encoding
- Chain multiple messages together (system, user, assistant)
- `enable_thinking` flag (set on first node) applies to all assistant messages
- `thinking_content` field for custom thinking content (assistant role only)
- Output connects to ZImageTextEncoder's `conversation_override` input

**conversation_override Input** on ZImageTextEncoder
- Optional input that accepts ZIMAGE_CONVERSATION type
- When connected, overrides all other inputs (text, system_prompt, raw_prompt, etc.)
- Enables iterative/conversational image generation workflows

### Removed

**max_sequence_length Parameter** - Removed from ZImageTextEncoder
- ComfyUI natively handles unlimited context (`max_length=99999999`)
- Qwen3-4B supports 40K tokens (`max_position_embeddings: 40960`)
- The 512 limit was unnecessarily restrictive

### Workflow Example

```
ZImageMessageChain (system) --> ZImageMessageChain (user) --> ZImageMessageChain (assistant) --> ZImageTextEncoder
enable_thinking: True           previous: (connect)         thinking_content: ""                conversation_override: (connect)
```

---

## v2.9.2 - Z-Image Qwen3 Template Fix

### Fixed

**Correct Qwen3-4B Chat Template Format**
- Format now matches `coderef/Qwen3-4B/tokenizer_config.json` exactly
- `thinking_content` now correctly placed inside `<think>...</think>` tags
- Added `assistant_content` field for content AFTER `</think>` tags

**Template Loading Bug**
- Python now loads templates from files if `system_prompt` is empty (JS fallback)
- Previously, `template_preset` was received but never used - relied entirely on JS auto-fill
- Now works even if JS fails (browser cache, errors, etc.)

**Template Structure:**
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
<think>
{thinking_content}
</think>

{assistant_content}
```

### Changed
- `add_think_block` now auto-enables when `thinking_content` is provided
- Both `ZImageTextEncoder` and `ZImageTextEncoderSimple` updated with `assistant_content`
- JS uses `beforeRegisterNodeDef` pattern (matches qwen_template_builder.js)

### Architecture
- **Organized templates**: Z-Image templates moved to `nodes/templates/z_image/`
- **Single source of truth**: Templates only in `.md` files (no duplication)
- **API endpoint**: `/api/z_image_templates` serves templates to JS
- **JS fetches from API**: No hardcoded templates (was 134 duplicated in JS)
- **Python fallback**: Encoder loads from files if JS fails
- **Adding templates**: Just add `.md` file to `nodes/templates/z_image/`

---

## v2.9.1 - Z-Image Encoder Simplification

### Changed

**Simplified Z-Image Encoder Interface**
- Removed redundant `system_prompt_preset` (templates do the same thing)
- Renamed `custom_system_prompt` to `system_prompt` (clearer)
- Added `raw_prompt` input - bypass all formatting, write your own `<|im_start|>` tokens
- Added `formatted_prompt` output - see exactly what gets encoded (connect to ShowText)
- JS auto-fills `system_prompt` when `template_preset` changes (editable)

**New Input Priority:**
1. `raw_prompt` (if set, bypasses everything)
2. `system_prompt` (custom or auto-filled from template)
3. `template_preset` (auto-fills system_prompt via JS)

### Fixed
- JS callback now uses `beforeRegisterNodeDef` hook (matches qwen_template_builder.js pattern)

---

## v2.9.0 - Z-Image Text Encoder with Experimental Options

### Added

**Z-Image Text Encoder Nodes** - Experimental encoding options for Z-Image
- `ZImageTextEncoder` - Full-featured encoder with templates, raw mode, thinking content
- `ZImageTextEncoderSimple` - Drop-in replacement for CLIPTextEncode

**Key Finding (Corrected Analysis):**
After testing actual tokenizers, we found:
- ComfyUI and diffusers produce **identical templates** by default
- The `enable_thinking` parameter is counterintuitive:
  - `enable_thinking=True` = NO think block (this is what diffusers uses)
  - `enable_thinking=False` = ADD think block
- So ComfyUI is NOT missing anything - both produce the same output

**Features:**
- `template_preset` dropdown with JS auto-fill to `system_prompt`
- `raw_prompt` input for complete control with your own special tokens
- `formatted_prompt` output - see exactly what gets encoded
- `add_think_block` parameter (default: False, matches diffusers)
- `thinking_content` parameter - insert text between `<think>` tags (experimental)
- `max_sequence_length` parameter (default: 512, matches diffusers)
- Template files in `nodes/templates/z_image_*.md`

**Use case:** Experimentation with think blocks and system prompts to see if they improve output.

**Key insight:** Qwen3-4B (no suffix) is the INSTRUCT model, not base. Naming is opposite of convention.

### Known Gaps vs Diffusers (Cannot Fix)
- **Embedding extraction**: Diffusers filters to valid tokens only; ComfyUI returns full padded sequence
- **Tokenizer difference**: ComfyUI bundles Qwen2.5-VL tokenizer; `<think>` becomes subwords not special token

### Documentation
- `nodes/docs/z_image_analysis.md` - Corrected ComfyUI vs Diffusers comparison
- `nodes/docs/z_image_encoder.md` - Main encoder documentation
- `nodes/docs/z_image_workflow_guide.md` - Setup and experiments

---

## v2.8.3 - C2C Vision Bridge Archived

### Removed

**C2C Vision Bridge nodes archived**
- Moved to `nodes/archive/c2c/` (3 nodes: QwenC2CBridgeLoader, QwenC2CCacheExtractor, QwenC2CVisionEnhancer)
- Feature was experimental and not providing practical value
- Related documentation and setup scripts also archived

---

## v2.8.2 - Debug Patches Opt-In, Wrapper Nodes Archived

### Fixed

**Debug patches no longer applied by default** ([#11](https://github.com/fredbliss/ComfyUI-QwenImageWanBridge/issues/11))
- Removed automatic monkey-patching of `QwenImageTransformer2DModel._forward`
- Debug patches are now opt-in via `QWEN_ENABLE_DEBUG_PATCHES=true` environment variable
- Eliminates wrapper overhead on forward passes for all users
- Verbose tracing still available via `QWEN_DEBUG_VERBOSE=true` when patches enabled

**Wrapper nodes archived** ([#12](https://github.com/fredbliss/ComfyUI-QwenImageWanBridge/issues/12))
- Moved all wrapper nodes to `nodes/archive/wrapper/` (11 nodes)
- Moved custom DiT implementation (`models/`) to archive
- Addresses VRAM leak concerns (unbounded RoPE cache, unmanaged model memory)
- Main workflow uses ComfyUI's native Qwen support - no wrapper nodes needed

**Debug controller RAM leak fixed** ([#12](https://github.com/fredbliss/ComfyUI-QwenImageWanBridge/issues/12))
- Replaced unbounded lists with `deque` ring buffers (auto-evict oldest entries)
- Limits: 1000 execution logs, 500 memory snapshots, 200 errors, 100 perf entries/component

---

## v2.8.1 - HunyuanVideo Template Connection Fix

### Changed

**HunyuanVideoTextEncoder Template Input**
- Added optional `template_input` (HUNYUAN_TEMPLATE) to accept Template Builder connection
- Priority order: template_input > custom_system_prompt > template_preset dropdown
- Template Builder `template_output` can now connect to Encoder `template_input`
- Encoder `text` moved to optional (can be provided by Template Builder)

### Added

**HunyuanVideoPromptExpander Node** - not working, likely not going to pursue due to better alternatives

**8 Structure Test Templates** (for prompt quality experiments)
- `hunyuan_video_structured_realism` - Full official structure with 5-second constraint
- `hunyuan_video_minimal_structure` - Tests minimum viable structure
- `hunyuan_video_temporal_only` - Only temporal markers (Initially/Then/Next/Finally)
- `hunyuan_video_camera_focused` - Detailed camera movement and angle specs
- `hunyuan_video_lighting_focused` - Rembrandt, golden hour, practical lighting emphasis
- `hunyuan_video_style_spam` - Tests if repeating "cinematic realistic style" helps
- `hunyuan_video_anti_pattern` - Everything the official rewriter avoids (baseline test)
- `hunyuan_video_self_expand` - System prompt that instructs internal expansion

**8 Experimental Fun Templates** (marked `experimental: true`)
- `hunyuan_video_drunk_cameraman` - Wobbly, off-center documentary footage
- `hunyuan_video_80s_music_video` - Dramatic lighting, lens flares, wet surfaces
- `hunyuan_video_majestic_pigeon` - David Attenborough energy for mundane subjects
- `hunyuan_video_wes_anderson_fever` - Symmetrical pastel fever dream
- `hunyuan_video_michael_bay_mundane` - Heroic angles for grocery shopping
- `hunyuan_video_excited_dog_pov` - Chaotic joyful golden retriever camera
- `hunyuan_video_infomercial_disaster` - Everything goes wrong dramatically
- `hunyuan_video_romcom_lighting` - Soft glowing romantic comedy visuals

---

## v2.8.0 - HunyuanVideo 1.5 Text-to-Video Support

### Added

**HunyuanVideo 1.5 T2V Nodes**
- `HunyuanVideoCLIPLoader` - Load Qwen2.5-VL (byT5 optional for multilingual)
- `HunyuanVideoTextEncoder` - T2V with 23 templates, dual output (positive, negative)
  - Default negative: "low quality, blurry, distorted, artifacts, watermark, text, logo"
  - 23 video templates in `nodes/templates/hunyuan_video_*.md`
  - For basic encoding without templates, use CLIPTextEncode directly

**Workflow Options**
- KSampler: Connect positive/negative directly (CFG built-in)
- SamplerCustomAdvanced: Route through CFGGuider first

### Technical Details
- Uses ComfyUI's native HunyuanVideo sampler and VAE
- byT5 auto-triggered by quoted text for multilingual rendering
- ComfyUI handles crop_start correction automatically

## BREAKING CHANGE (v2.7.0+)

**Existing workflows will break.** You must:
1. Delete and re-add, or Right Click -> Recreate Template Builder, Encoder, and Encoder Advanced nodes from your workflow
2. Connect: Template Builder `template_output` → Encoder `template_output` (single connection only)

Old multi-connection system (mode + system_prompt) no longer works as it was getting convoluted and confusing. One connection from template builder now handles everything (prompt, mode, system_prompt).

**Updated Workflows Available** [here](example_workflows/nunchaku_qwen_image_edit_2509.json)
1. [Multi image edit workflow](example_workflows/nunchaku_qwen_image_edit_2509.json)
2. [Single image edit workflow](example_workflows/qwen_edit_2509_single_image_edit.json)

---

## v2.7.0 - Template System Refactor & Simplified Connection

### BREAKING CHANGES

**Simplified Template Connection**
- Template Builder now outputs single `template_output` (contains everything)
- Connect ONE output: Template Builder `template_output` → Encoder `template_output`
- Old system (separate mode + system_prompt connections) removed
- **You must recreate Template Builder and Encoder nodes in existing workflows**

### Changed

**File-Based Template System**
- Templates now stored as `.md` files in `nodes/templates/` directory (this now means you can make your own templates super easily)
- Single source of truth for all templates (no hardcoded dicts)
- YAML frontmatter for metadata (mode, vision, experimental flags)
- Easy to add/edit templates without code changes
- Shared `template_loader.py` used by all nodes (cached on load)

**Template Builder UI Enhancement**
- JavaScript extension now populates `custom_system` field when selecting presets
- Select a preset → see/edit the system prompt before using it
- Fixed widget name mismatch (was looking for `system_prompt`, now correctly uses `custom_system`)

### Technical Details
- `nodes/template_loader.py` - Shared loader with singleton pattern
- Templates use YAML frontmatter: `mode`, `vision`, `use_picture_format`, `experimental`, `no_template`
- Python node reads templates on module import (cached)
- JavaScript keeps hardcoded copy for UI responsiveness (synced manually)

## v2.6.2 - Multi-Image Batch Node

### Added

**QwenImageBatch Node** - Smart multi-image batching node to handle images with control over strategy and scaling
- Auto-detects up to 10 images (no manual inputcount)
- Skips empty inputs (no black images)
- Two batching strategies: `max_dimensions` (minimal distortion) and `first_image` (hero-driven)
- Automatic double-scaling prevention (marks images as pre-scaled, encoders skip VAE scaling)
- Enhanced debug logging (aspect ratios, scale factors, dimension adjustments)

**multi_image_edit Mode** - DiffSynth-compatible multi-reference encoding
- Vision tokens inside prompt (not before) - matches DiffSynth `encode_prompt_edit_multi`
- Automatic "Picture X:" labeling
- Available in both standard and advanced encoders

**Template Builder → Encoder Auto-Sync**
- Template Builder outputs `mode` (3rd output)
- Encoders accept mode as STRING input (can be connected or typed manually)
- Prevents vision token placement/drop index mismatches

### Fixed
- KJNodes ImageBatchMulti black image issue (empty inputs) - resolution was to create a new node that handles our use case exactly as needed
- Aspect ratio cropping in standard batch nodes
- Double-scaling when using batch node → encoder
- Multi-image template mode mismatch

### Technical Notes
- QwenImageBatch uses v2.6.1 scaling modes + batch strategy
- Metadata propagation: `qwen_pre_scaled`, `qwen_scaling_mode`, `qwen_batch_strategy`
- Vision encoder always scales (384×384 target), VAE skips if pre-scaled
- 32px alignment maintained throughout pipeline

## v2.6.1 - Resolution Scaling Fix

### Fixed
- **Zoom-out issue in image editing** - Large images were being aggressively downscaled
  - Previous behavior: 1477×2056 scaled to 864×1216 (0.59x, causing zoom-out)
  - New default: `preserve_resolution` keeps original dimensions with 32px alignment
  - Example: 1477×2056 → 1472×2048 (minimal crop, no zoom-out)

### Added
- **scaling_mode parameter** in QwenVLTextEncoder and QwenVLTextEncoderAdvanced with three modes:
  - `preserve_resolution` (default) - Keeps input size, only applies 32px alignment
  - `max_dimension_1024` - Scales largest side to 1024px (good for 4K images)
  - `area_1024` - Legacy behavior, scales to ~1024×1024 area
- Improved tooltips with detailed tradeoffs for each mode
- Advanced encoder applies scaling_mode as base, then applies resolution_mode weights on top

### Changed
- Vision encoder always uses 384×384 area scaling (unchanged)
- VAE encoder respects new scaling_mode setting
- Debug output shows which scaling mode is active

### Scaling Mode Comparison

**preserve_resolution (default, recommended for most use cases)**
- ✓ No zoom-out effect, subjects stay full-size
- ✓ Best quality output, minimal cropping (only 32px alignment)
- ✓ Works perfectly for typical image sizes (512px-2048px)
- ✗ May use significant VRAM with very large images (4K+)
- Use when: You want maximum quality and your images are under 2500px

**max_dimension_1024 (recommended for 4K and very large images)**
- ✓ Reduces VRAM usage significantly on large images
- ✓ Balanced quality vs performance tradeoff
- ✓ Prevents OOM errors on 4K images
- ✗ Some zoom-out effect on images larger than 1024px
- Use when: Working with 4K images or hitting VRAM limits

**area_1024 (legacy, not recommended)**
- ✓ Consistent ~1 megapixel output size
- ✗ Aggressive zoom-out on large images (major quality loss)
- ✗ Upscales small images unnecessarily (wastes quality)
- ✗ Poor behavior across different input sizes
- Use when: You need exact 1024×1024 area behavior for specific workflows

### Examples by Resolution

**Small portrait (512×768 = 0.4MP)**
- preserve: 512×768 (1.00x) - unchanged, perfect
- max_dimension: 672×1024 (1.31x) - upscaled
- area: 832×1248 (1.62x) - upscaled too much

**Large portrait (1477×2056 = 3MP, the reported issue)**
- preserve: 1472×2048 (1.00x) - no zoom-out, FIXED
- max_dimension: 736×1024 (0.50x) - some zoom-out
- area: 864×1216 (0.58x) - old broken behavior

**4K landscape (3840×2160 = 8MP)**
- preserve: 3840×2176 (1.00x) - may OOM on lower VRAM
- max_dimension: 1024×576 (0.27x) - RECOMMENDED for 4K
- area: 1376×768 (0.36x) - inconsistent scaling

## v2.6 - Mask-Based Inpainting

### Added
- **QwenMaskProcessor** - Mask preprocessing node
  - Base64 mask input from spatial editor
  - Blur, expand, feather controls
  - Preview overlays showing inpaint areas
- **QwenInpaintSampler** - Inpainting sampler node
  - Implements exact diffusers blending: `(1-mask)*original + mask*generated`
  - Strength control for partial/full regeneration
  - 4-channel to 16-channel auto-conversion
- **Inpainting mode** in QwenVLTextEncoder
  - New `inpaint_mask` optional parameter
  - Mask passed to conditioning for spatial control
- **Example workflows**
  - `qwen_edit_2509_mask_inpainting.json` - Standard workflow
  - `nunchaku_qwen_mask_inpainting.json` - Nunchaku variant

### Technical Details
- Follows DiffSynth-Studio mask-based approach (not coordinate tokens)
- Spatial tokens marked as experimental based on DiffSynth analysis
- JavaScript interface already supports both systems

## v2.5.1 - Reorganized example workflows
- Example workflows have been renamed and reorganized, with a new one as well for Nunchaku 2509

## v2.5 update - Wrapper Nodes (NOT WORKING YET - Experimental)

### Added
- **Wrapper Node System** - Independent implementation (11 nodes):
  - `QwenImageDiTLoaderWrapper` - Load Qwen DiT using transformers
  - `QwenVLTextEncoderLoaderWrapper` - Load Qwen2.5-VL using transformers
  - `QwenImageVAELoaderWrapper` - Load 16-channel VAE using diffusers/ComfyUI
  - `QwenModelManagerWrapper` - Unified pipeline loader
  - `QwenProcessorWrapper` - Process text/images with Qwen2VL processor
  - `QwenProcessedToEmbedding` - Convert processed tokens to conditioning
  - `QwenImageEncodeWrapper` - Encode images to edit latents (batch support via Image Batch node)
  - `QwenImageModelWrapper` - DiffSynth-style forward pass with 2x2 packing
  - `QwenImageSamplerNode` - FlowMatch sampler with built-in scheduling
  - `QwenImageModelWithEdit` - Inject edit latents into model
  - `QwenImageSamplerWithEdit` - Alternative sampler with edit support
  - `QwenDebugLatents` - Debug latent dimensions and flow

### Technical Implementation
- **No DiffSynth dependency** - Uses only transformers, diffusers, torch
- 2x2 patch packing operation following DiffSynth model_fn_qwen_image
- Edit latent concatenation in sequence dimension (not conditioning)
- FlowMatch scheduler with resolution-aware dynamic shift
- Automatic padding for odd dimensions to enable patch processing
- Direct edit latent injection bypassing ComfyUI's conditioning system
- ComfyUI's standard VAE loader for 16-channel VAE (auto-detects config)

### Fixed
- Removed all DiffSynth imports from wrapper loaders
- Fixed type mismatch: text encoder loader outputs `QWEN_TEXT_ENCODER` not `CLIP`
- Removed redundant `QwenImageCombineLatents` node (use Image Batch instead)
- Removed redundant `QwenSchedulerNode` (sampler has built-in scheduling)
- VAE loader now uses ComfyUI's auto-config detection

### Status
- **Not fully tested** - Wrapper nodes are complete but haven't been validated with actual models
- **Use Standard Nodes** - Recommended for production use (QwenVLCLIPLoader + QwenVLTextEncoder)

## v2.5 Power User Features & Auto-Labeling

### Added
- **QwenVLTextEncoderAdvanced** for power users
  - Per-image resolution weighting (hero/reference modes)
  - Memory budget management (VRAM limits)
  - Custom resolution targets (vision & VAE separate)
  - Progressive and memory-optimized modes
- **Configurable auto-labeling**
  - auto_label parameter (on/off)
  - label_format choice (Picture/Image)
- **Documentation**
  - Advanced encoder test guide (10 configurations)
  - Model limits (512 images, token constraints)
  - Power user guide

### Changed
- Auto-labeling now optional (was always on)
- Debug shows auto_label and label_format status
- Advanced encoder simplified - removed confusing validation_mode parameter
- Added verbose_log parameter to control console logging separately from UI debug
- Enhanced debug patch with comprehensive tracing (timestamps, memory stats, NaN/Inf detection)

### Fixed
- Wan21 latent format requires 5D tensors - now properly handled
- **Dimension mismatch auto-handling** - No more errors from mismatched resolutions!
  - Automatically pads latents to even dimensions for patch processing
  - Wraps ComfyUI's QwenImage model to handle any resolution
  - Users can now use any resolution, not just 1024x1024
- Clear debug messages showing dimension adjustments
- Documentation updated with technical explanation

## v2.4 DiffSynth Alignment & Better Debug

### Added
- **100% DiffSynth-Studio alignment** verified for all components
- **Better debug output** showing full prompts without truncation
- **New face replacement templates** aligned with Qwen-Image-Edit-2509:
  - `qwen_face_swap` - Simple face swap following model's training
  - `qwen_identity_merge` - Identity transfer with full scene preservation
- **Character counts** in debug output for tracking token usage

### Fixed
- Face replacement templates now generate full images, not just face crops
- Debug output shows complete formatted text being encoded
- Templates use "generate an image" structure matching model training

### Changed
- Updated face replacement templates to preserve full scene context
- Debug output sections reorganized for better clarity

## v2.3 Debug Controller & Silent Patches

### Added
- **QwenDebugController node** - Comprehensive debugging system
  - Multi-level debugging: off/basic/verbose/trace
  - Performance profiling and memory tracking
  - Component filtering and regex pattern matching
  - Export debug sessions to JSON/text
  - System info display (GPU, CPU, memory)
- **Silent debug patches** - No console spam by default
  - Set `QWEN_DEBUG_VERBOSE=true` or use Debug Controller to enable
  - Integrates with encoder's debug_mode parameter

### Fixed
- Debug patches now run silently unless explicitly enabled
- No more console spam during normal operation

## v2.2 Token Dropping & N-Image Support

### Added
- **Proper token dropping implementation** matching DiffSynth/Diffusers behavior
  - System prompt included during encoding for context
  - First 34/64 embeddings dropped after encoding
  - Prevents text contamination while maintaining quality
- **N-image support** in Template Builder (0-100 images)
  - Automatic warnings at 4+ and 10+ images
  - 0 = auto-detect mode for future implementation
- **Simplified architecture** following DRY principles
  - Template Builder: Single source for system prompts
  - Processor: Minimal formatter (80 lines)
  - Encoder: Handles all technical encoding/dropping

### Fixed
- System prompts now provide proper context without appearing in images
- Token dropping happens after encoding (was missing entirely)

## v2.1 System Prompt Separation

### Fixed
- **System prompt appearing in generated images** - Template formatting issue
- **Template Builder and Encoder separation** - No more duplicated logic

## v2.0 Qwen-Image-Edit-2509 Multi-Image Support

### Added
- **Full Qwen-Image-Edit-2509 support** with native multi-image editing capabilities
- **"Picture X:" reference formatting** for addressing specific images in prompts
- **Automatic detection** of Picture references in prompts for smart formatting
- **Template system enhancements**:
  - `custom_system` field works as override for ANY template mode
  - `show_all_prompts` mode displays all available system prompts
  - Uses exact templates from DiffSynth-Studio repository
- **Example workflows** with comprehensive documentation
  - Multi-image 2509 workflow with detailed notes
  - Text-to-image workflow with proper connections

### Fixed
- **Dimension mismatch in native_multi mode** causing VAE latent size errors
- **QwenMultiReferenceHandler** now enforces uniform dimensions automatically
- **Template system corrections** using proper DiffSynth-Studio prompts
- **Workflow parameter alignment** in example JSON files

### Changed
- **Multi-image workflow**: Use ComfyUI's Image Batch node for multiple images
- **Template consistency**: All templates use official DiffSynth-Studio system prompts
- **Dimension handling**: Automatic fixes prevent model forward pass errors

## v1.6+ Experimental Features (Archived)

### Spatial Token Generator
- Visual spatial editing interface with region drawing
- Multiple output formats (JSON, XML, natural language, traditional tokens)
- Coordinate system handling and validation
- **Status**: Experimental, low priority - spatial tokens effectiveness unclear

### EliGen Entity Control
- Entity-level spatial generation with masks
- Up to 4 regions with individual prompts and weights
- Based on DiffSynth implementation
- **Status**: Experimental, untested with current models

### Earlier Versions
- Multi-reference image processing improvements
- Resolution optimization and aspect ratio handling
- Debug mode enhancements
- Template builder separation from encoder

## Current Working State

### Production Ready
- QwenVLTextEncoder with proper template separation
- QwenTemplateBuilder with DiffSynth-Studio templates
- Multi-image support via Image Batch node
- Dimension-aware multi-reference processing
- Debug modes for troubleshooting

### Experimental
- Spatial token generation (effectiveness unclear)
- Entity control nodes (untested)
- Advanced spatial editing features

### Removed/Archived
- WAN keyframe nodes
- Video-to-video nodes
- Dual encoding architecture (overcomplicated)
- Native implementation attempts (wrapper approach preferred)
- Inpainting nodes (broken, low priority)
