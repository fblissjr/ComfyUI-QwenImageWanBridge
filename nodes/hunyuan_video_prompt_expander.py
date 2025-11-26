"""
HunyuanVideo Prompt Expander

Uses the already-loaded Qwen2.5-VL weights for LLM-based prompt expansion.
Extracts weights from ComfyUI's CLIP and adds only the lm_head for generation.

This approach reuses ~14GB of already-loaded weights, only loading ~50MB extra for lm_head.
"""

import os
import re
import json
import logging
from typing import Tuple, Optional, Dict, Any

import torch
import folder_paths

logger = logging.getLogger(__name__)

# Generation config matching official HunyuanVideo rewriter
GENERATION_CONFIG = {
    "temperature": 0.01,  # Near-deterministic
    "top_k": 1,
    "top_p": 0.001,
    "max_new_tokens": 512,
    "do_sample": True,
    "repetition_penalty": 1.0,
}

# Self-expand system prompt - instructs LLM to expand simple prompts
SELF_EXPAND_SYSTEM = """You are a video description structuring assistant. Expand the user's simple prompt into a detailed description for video generation.

OUTPUT STRUCTURE (follow this order):

1. SUMMARY: One sentence combining subject + action + setting

2. SUBJECT DETAILS: Observable characteristics only
   - For people: age range, build, hair, clothing
   - For animals: species, coloring, size
   - For objects: material, color, condition

3. TEMPORAL SEQUENCE: Four beats totaling 5 seconds
   - Initially, [first moment]
   - Then, [second development]
   - Next, [third progression]
   - Finally, [concluding state]

4. ENVIRONMENT: Background, secondary elements, atmosphere

5. CAMERA: Infer from content:
   - Action = dynamic tracking, handheld energy
   - Portrait = stable framing, subtle push
   - Landscape = slow pan, wide establishing
   - Default = eye-level, medium shot, subtle movement

6. LIGHTING: Infer from setting:
   - Outdoor day = natural sunlight
   - Indoor = practical sources, ambient
   - Night = artificial, contrast
   - Default = soft natural light

7. STYLE: Infer from prompt keywords:
   - "anime", "cartoon" = anime animation style
   - "painting", "artistic" = oil painting style
   - "noir", "dark" = film noir style
   - "documentary" = documentary style
   - Default = cinematic realistic style

RULES:
- Present tense only
- Observable actions only (no internal states)
- All actions must fit within 5 seconds
- Maximum 3 subjects unless specified
- Target length: 300-400 words
- End with: "The video is [inferred style] style."

Output ONLY the expanded description, no headers or labels."""


class HunyuanVideoPromptExpander:
    """
    LLM-based prompt expander using the already-loaded Qwen2.5-VL.

    Extracts weights from ComfyUI's CLIP model and adds lm_head for generation.
    This reuses the ~14GB model already in VRAM, only adding ~50MB for lm_head.
    """

    # Cache for the generator model (created once, reused)
    _generator_cache: Dict[str, Any] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Simple prompt to expand (e.g., 'a cat watching birds')"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Custom system prompt (leave empty for default self-expand)"
                }),
                "max_tokens": ("INT", {
                    "default": 512,
                    "min": 100,
                    "max": 1024,
                    "tooltip": "Maximum tokens for expanded prompt"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Generation temperature (0.01 = near-deterministic)"
                }),
                "bypass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip expansion, pass prompt through unchanged"
                }),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING", "STRING")
    RETURN_NAMES = ("clip", "expanded_prompt", "debug_info")
    FUNCTION = "expand"
    CATEGORY = "HunyuanVideo/Utilities"
    TITLE = "HunyuanVideo Prompt Expander"
    DESCRIPTION = "LLM-based prompt expansion using loaded Qwen2.5-VL weights. Reuses CLIP model, adds only lm_head."

    def _get_model_path(self, clip) -> Optional[str]:
        """Try to find the original model path from CLIP."""
        # Check if we can find the model path from clip's loading info
        try:
            # ComfyUI stores some info about loaded models
            if hasattr(clip, 'patcher') and hasattr(clip.patcher, 'model_options'):
                opts = clip.patcher.model_options
                if 'model_path' in opts:
                    return opts['model_path']
        except:
            pass

        # Fallback: search in text_encoders folder for Qwen models
        try:
            text_encoders = folder_paths.get_filename_list("text_encoders")
            for name in text_encoders:
                if "qwen" in name.lower() and "2.5" in name.lower():
                    path = folder_paths.get_full_path("text_encoders", name)
                    if path:
                        # Return path whether it's a file or directory
                        return path
        except:
            pass

        return None

    def _load_lm_head(self, model_path: str, device: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Load only the lm_head.weight from safetensors."""
        try:
            from safetensors import safe_open

            # If model_path is a file, try to load lm_head from it directly
            if os.path.isfile(model_path) and model_path.endswith('.safetensors'):
                with safe_open(model_path, framework="pt", device=device) as f:
                    if "lm_head.weight" in f.keys():
                        logger.info(f"Found lm_head in: {model_path}")
                        return f.get_tensor("lm_head.weight").to(dtype)
                    else:
                        logger.info(f"lm_head not in {os.path.basename(model_path)} (encoder-only model)")
                return None

            # If it's a directory, check for model index (sharded model)
            if os.path.isdir(model_path):
                index_path = os.path.join(model_path, "model.safetensors.index.json")
                if os.path.exists(index_path):
                    with open(index_path, 'r') as f:
                        index = json.load(f)
                    lm_head_file = index.get("weight_map", {}).get("lm_head.weight")
                    if lm_head_file:
                        shard_path = os.path.join(model_path, lm_head_file)
                        with safe_open(shard_path, framework="pt", device=device) as f:
                            logger.info(f"Found lm_head in shard: {lm_head_file}")
                            return f.get_tensor("lm_head.weight").to(dtype)

                # Single file in directory
                single_path = os.path.join(model_path, "model.safetensors")
                if os.path.exists(single_path):
                    with safe_open(single_path, framework="pt", device=device) as f:
                        if "lm_head.weight" in f.keys():
                            return f.get_tensor("lm_head.weight").to(dtype)

        except Exception as e:
            logger.warning(f"Failed to load lm_head: {e}")

        return None

    def _get_generator(self, clip) -> Optional[Any]:
        """Get or create the generator model from CLIP weights."""
        # Check cache first
        cache_key = id(clip)
        if cache_key in self._generator_cache:
            return self._generator_cache[cache_key]

        try:
            # Use text-only Qwen2 model - we don't need vision for prompt expansion
            from transformers import Qwen2ForCausalLM, Qwen2Config

            # Get the underlying model from CLIP
            # Structure varies by model type:
            # - HunyuanVideo 1.5: clip.cond_stage_model.qwen25_7b.transformer
            # - HunyuanVideo 1.0: clip.cond_stage_model.llama.transformer
            # - QwenImage: clip.cond_stage_model.transformer
            cond_model = clip.cond_stage_model

            # Find the Qwen transformer - try various paths
            transformer = None
            transformer_path = None

            # HunyuanVideo 1.5 / HunyuanImage (Qwen2.5-VL + byT5)
            if hasattr(cond_model, 'qwen25_7b') and hasattr(cond_model.qwen25_7b, 'transformer'):
                transformer = cond_model.qwen25_7b.transformer
                transformer_path = "cond_stage_model.qwen25_7b.transformer"
            # HunyuanVideo 1.0 (LLaMA)
            elif hasattr(cond_model, 'llama') and hasattr(cond_model.llama, 'transformer'):
                transformer = cond_model.llama.transformer
                transformer_path = "cond_stage_model.llama.transformer"
            # Direct transformer (some models)
            elif hasattr(cond_model, 'transformer'):
                transformer = cond_model.transformer
                transformer_path = "cond_stage_model.transformer"
            # clip_l path
            elif hasattr(cond_model, 'clip_l') and hasattr(cond_model.clip_l, 'transformer'):
                transformer = cond_model.clip_l.transformer
                transformer_path = "cond_stage_model.clip_l.transformer"

            if transformer is None:
                # Debug: list available attributes
                attrs = [a for a in dir(cond_model) if not a.startswith('_')]
                logger.warning(f"Could not find Qwen transformer in CLIP model. Available attrs: {attrs[:10]}")
                return None

            logger.info(f"Found transformer at: {transformer_path}")

            # Get state dict from ComfyUI's model
            comfy_state_dict = transformer.state_dict()
            device = next(transformer.parameters()).device
            dtype = next(transformer.parameters()).dtype

            logger.info(f"Extracted {len(comfy_state_dict)} weights from ComfyUI model")
            logger.info(f"Device: {device}, dtype: {dtype}")

            # Find model path and load lm_head
            model_path = self._get_model_path(clip)
            lm_head_weight = None
            if model_path:
                lm_head_weight = self._load_lm_head(model_path, str(device), dtype)
                if lm_head_weight is not None:
                    logger.info(f"Loaded lm_head.weight: {lm_head_weight.shape}")

            # Create text-only config matching Qwen2.5-VL-7B language model
            # We skip vision entirely since we're only doing text generation
            config = Qwen2Config(
                vocab_size=152064,
                hidden_size=3584,
                intermediate_size=18944,
                num_hidden_layers=28,
                num_attention_heads=28,
                num_key_value_heads=4,
                max_position_embeddings=128000,
                rms_norm_eps=1e-6,
                rope_theta=1000000.0,
                tie_word_embeddings=False,
            )

            # Create text-only model shell
            model = Qwen2ForCausalLM(config)

            # Map ComfyUI state dict to transformers format
            # ComfyUI uses: model.layers.X.*, visual.*
            # Transformers uses: model.model.layers.X.*
            # Skip vision weights - not needed for text generation
            hf_state_dict = {}
            for key, value in comfy_state_dict.items():
                # Skip vision weights entirely
                if key.startswith("visual."):
                    continue
                if key.startswith("model."):
                    # Language model: model.* -> model.model.*
                    new_key = "model." + key
                else:
                    new_key = key
                hf_state_dict[new_key] = value

            # Add lm_head
            if lm_head_weight is not None:
                hf_state_dict["lm_head.weight"] = lm_head_weight
            else:
                # Tie to embeddings as fallback (less accurate but works)
                if "model.model.embed_tokens.weight" in hf_state_dict:
                    hf_state_dict["lm_head.weight"] = hf_state_dict["model.model.embed_tokens.weight"]
                    logger.warning("Using tied embeddings for lm_head (lm_head.weight not found)")

            # Load state dict
            missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
            if missing:
                # Filter out expected missing keys for logging
                important_missing = [k for k in missing if not k.startswith("model.visual")]
                if important_missing:
                    logger.warning(f"Missing keys: {len(important_missing)} (first 5: {important_missing[:5]})")
            if unexpected:
                logger.warning(f"Unexpected keys: {len(unexpected)}")

            model.to(device)
            model.eval()

            # Cache and return
            self._generator_cache[cache_key] = model
            logger.info("Created text-only generator model from CLIP weights")
            return model

        except Exception as e:
            logger.error(f"Failed to create generator: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate(
        self,
        model,
        tokenizer,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate expanded prompt using the model."""
        # Build chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.001),  # Avoid div by zero
                top_k=1 if temperature < 0.1 else 50,
                top_p=0.001 if temperature < 0.1 else 0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        result = tokenizer.decode(generated, skip_special_tokens=True)

        return result.strip()

    def expand(
        self,
        clip,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.01,
        bypass: bool = False
    ) -> Tuple[Any, str, str]:
        """
        Expand a simple prompt into a structured one using LLM generation.
        """
        if bypass or not prompt.strip():
            return (clip, prompt, "Bypass enabled or empty prompt")

        # Check if prompt is already expanded (heuristic: >200 words)
        word_count = len(prompt.split())
        if word_count > 200:
            return (clip, prompt, f"Prompt already expanded ({word_count} words)")

        debug_info = []
        debug_info.append(f"Input: {prompt[:100]}..." if len(prompt) > 100 else f"Input: {prompt}")
        debug_info.append(f"Word count: {word_count}")

        try:
            # Get or create generator
            generator = self._get_generator(clip)
            if generator is None:
                debug_info.append("ERROR: Could not create generator from CLIP weights")
                debug_info.append("Returning original prompt")
                return (clip, prompt, "\n".join(debug_info))

            # Get tokenizer
            from transformers import AutoTokenizer
            model_path = self._get_model_path(clip)
            if model_path:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    trust_remote_code=True
                )

            # Use custom or default system prompt
            sys_prompt = system_prompt.strip() if system_prompt.strip() else SELF_EXPAND_SYSTEM

            debug_info.append(f"System prompt: {len(sys_prompt)} chars")
            debug_info.append(f"Temperature: {temperature}")
            debug_info.append(f"Max tokens: {max_tokens}")

            # Generate
            expanded = self._generate(
                generator,
                tokenizer,
                prompt,
                sys_prompt,
                max_tokens,
                temperature
            )

            debug_info.append(f"Expanded word count: {len(expanded.split())}")
            debug_info.append(f"\n--- EXPANDED PROMPT ---\n{expanded}\n--- END ---")

            return (clip, expanded, "\n".join(debug_info))

        except Exception as e:
            logger.error(f"Expansion failed: {e}")
            import traceback
            traceback.print_exc()
            debug_info.append(f"ERROR: {e}")
            debug_info.append("Returning original prompt")
            return (clip, prompt, "\n".join(debug_info))


# Node registration
NODE_CLASS_MAPPINGS = {
    "HunyuanVideoPromptExpander": HunyuanVideoPromptExpander,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanVideoPromptExpander": "HunyuanVideo Prompt Expander (LLM)",
}
