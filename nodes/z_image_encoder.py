"""
Z-Image Text Encoder for ComfyUI

Transparent encoder - see exactly what gets sent to the model.
Templates auto-fill the system prompt field so you can edit them.
Use raw_prompt for complete control with your own special tokens.
Use ZImageTurnBuilder for multi-turn conversations.
"""

import os
import re
import copy
import logging
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)


# Custom type for conversation chains
ZIMAGE_CONVERSATION_TYPE = "ZIMAGE_CONVERSATION"

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_z_image_templates() -> Dict[str, str]:
    """Load Z-Image templates from nodes/templates/z_image/*.md files."""
    templates = {}
    templates_dir = os.path.join(os.path.dirname(__file__), "templates", "z_image")

    if not os.path.exists(templates_dir):
        return templates

    for filename in os.listdir(templates_dir):
        if filename.endswith('.md'):
            template_name = filename[:-3]  # Remove '.md' suffix
            template_path = os.path.join(templates_dir, filename)
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        templates[template_name] = parts[2].strip()
            except Exception as e:
                logger.warning(f"Failed to load template {filename}: {e}")

    return templates


# Global template cache
_TEMPLATE_CACHE = None

def get_templates() -> Dict[str, str]:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = load_z_image_templates()
    return _TEMPLATE_CACHE


def format_conversation(conversation: Dict[str, Any]) -> str:
    """
    Format a conversation dict into Qwen3-4B chat template format.

    Conversation structure:
    {
        "enable_thinking": bool,
        "is_final": bool,  # If True (default), last message has no <|im_end|>
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "...", "thinking": "..."},
            ...
        ]
    }

    Output format:
    <|im_start|>system
    {content}<|im_end|>
    <|im_start|>user
    {content}<|im_end|>
    <|im_start|>assistant
    <think>
    {thinking}
    </think>

    {content}<|im_end|>
    ...

    Note: If is_final=True (default), last message does NOT get <|im_end|> (still being generated).
    If is_final=False, all messages get <|im_end|> (more turns expected).
    """
    enable_thinking = conversation.get("enable_thinking", False)
    is_final = conversation.get("is_final", True)  # Default to final
    messages = conversation.get("messages", [])

    if not messages:
        return ""

    parts = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        thinking = msg.get("thinking", "").strip()
        is_last = (i == len(messages) - 1)

        # Only skip <|im_end|> on last message if is_final=True
        skip_end = is_last and is_final

        if role == "system":
            if content:
                if skip_end:
                    parts.append(f"<|im_start|>system\n{content}")
                else:
                    parts.append(f"<|im_start|>system\n{content}<|im_end|>")

        elif role == "user":
            if skip_end:
                parts.append(f"<|im_start|>user\n{content}")
            else:
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")

        elif role == "assistant":
            if enable_thinking:
                # With thinking: <|im_start|>assistant\n<think>\n{think}\n</think>\n\n{content}
                think_inner = thinking if thinking else ""
                if skip_end:
                    parts.append(f"<|im_start|>assistant\n<think>\n{think_inner}\n</think>\n\n{content}")
                else:
                    parts.append(f"<|im_start|>assistant\n<think>\n{think_inner}\n</think>\n\n{content}<|im_end|>")
            else:
                # Without thinking: <|im_start|>assistant\n{content}
                if skip_end:
                    parts.append(f"<|im_start|>assistant\n{content}")
                else:
                    parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

    return "\n".join(parts)


class ZImageTurnBuilder:
    """
    Add a conversation turn for multi-turn Z-Image encoding.

    Each turn represents a complete exchange:
    - User message (required)
    - Assistant response (optional - with thinking if enabled in conversation)

    Workflow options:
    1. WITHOUT clip: Outputs conversation only - chain to more TurnBuilders or back to encoder
    2. WITH clip: Outputs conditioning directly - use for final turn without extra encoder

    The is_final flag indicates whether this is the last turn in the chain.
    When is_final=True, the last message won't get <|im_end|> (model continues generating).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "previous": (ZIMAGE_CONVERSATION_TYPE, {
                    "tooltip": "Previous conversation (from ZImageTextEncoder or another TurnBuilder)"
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "User's message for this turn"
                }),
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "Connect to encode and output conditioning directly (skip chaining back to encoder)"
                }),
                "thinking_content": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Assistant's thinking (only if conversation has enable_thinking=True)"
                }),
                "assistant_content": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Assistant's response after thinking"
                }),
                "is_final": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Is this the last turn? If True, last message has no <|im_end|>"
                }),
                "strip_key_quotes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove double quotes from JSON-style keys in prompts. Converts \"subject\": to subject: to prevent key names appearing as text in image."
                }),
            }
        }

    RETURN_TYPES = (ZIMAGE_CONVERSATION_TYPE, "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conversation", "conditioning", "formatted_prompt", "debug_output")
    FUNCTION = "add_turn"
    CATEGORY = "ZImage/Conversation"
    TITLE = "Z-Image Turn Builder"

    def add_turn(
        self,
        previous: Dict[str, Any],
        user_prompt: str,
        clip=None,
        thinking_content: str = "",
        assistant_content: str = "",
        is_final: bool = True,
        strip_key_quotes: bool = False,
    ) -> Tuple[Dict[str, Any], Any, str, str]:

        # Apply key quote filtering if enabled
        if strip_key_quotes:
            user_prompt = re.sub(r'"([^"]+)":', r'\1:', user_prompt)
            thinking_content = re.sub(r'"([^"]+)":', r'\1:', thinking_content)
            assistant_content = re.sub(r'"([^"]+)":', r'\1:', assistant_content)

        # Deep copy to avoid shared references
        conversation = {
            "enable_thinking": previous.get("enable_thinking", False),
            "messages": copy.deepcopy(previous.get("messages", [])),
            "is_final": is_final,
        }

        enable_thinking = conversation["enable_thinking"]
        prev_msg_count = len(conversation["messages"])

        # Add user message
        user_msg = {
            "role": "user",
            "content": user_prompt.strip(),
        }
        conversation["messages"].append(user_msg)

        # Always add assistant message (even if empty) to match encoder behavior
        # This ensures we get <|im_start|>assistant\n even with empty content
        assistant_msg = {
            "role": "assistant",
            "content": assistant_content.strip() if assistant_content else "",
        }
        # Add thinking if conversation has thinking enabled
        if enable_thinking:
            assistant_msg["thinking"] = thinking_content.strip() if thinking_content else ""
        conversation["messages"].append(assistant_msg)

        # Format conversation to text
        formatted_text = format_conversation(conversation)

        # Encode if clip provided
        conditioning = None
        if clip is not None:
            tokens = clip.tokenize(formatted_text)
            conditioning = clip.encode_from_tokens_scheduled(tokens)
            # Log formatted prompt to console (same as encoder)
            print(f"\n[Z-Image TurnBuilder] Formatted prompt:\n{formatted_text}\n")

        # Build debug output
        turn_num = (prev_msg_count // 2) + 1  # Rough turn number estimate
        debug_lines = [
            "=== Z-Image Turn Builder Debug ===",
        ]
        if strip_key_quotes:
            debug_lines.append("Key quote filter: ON (removing quotes from JSON keys)")
        debug_lines.extend([
            f"Turn #{turn_num} (appending to conversation with {prev_msg_count} messages)",
            f"User prompt: {len(user_prompt.strip())} chars",
            f"Enable thinking (inherited): {enable_thinking}",
            f"Clip connected: {clip is not None}",
        ])

        if enable_thinking:
            debug_lines.append(f"Thinking content: {len(thinking_content.strip()) if thinking_content else 0} chars")
        debug_lines.append(f"Assistant content: {len(assistant_content.strip()) if assistant_content else 0} chars")
        debug_lines.append(f"Is final: {is_final}")

        # Show this turn's formatted output
        debug_lines.append("")
        debug_lines.append("=== This Turn's Messages ===")
        debug_lines.append("<|im_start|>user")
        debug_lines.append(user_prompt.strip())
        debug_lines.append("<|im_end|>")  # User always gets end tag (assistant follows)

        debug_lines.append("<|im_start|>assistant")
        if enable_thinking:
            debug_lines.append("<think>")
            debug_lines.append(thinking_content.strip() if thinking_content else "")
            debug_lines.append("</think>")
            debug_lines.append("")
        debug_lines.append(assistant_content.strip() if assistant_content else "")
        if not is_final:
            debug_lines.append("<|im_end|>")

        if clip is not None:
            debug_lines.append("")
            debug_lines.append("=== Full Formatted Prompt ===")
            # Escape for HTML preview
            debug_lines.append(formatted_text.replace("<", "&lt;").replace(">", "&gt;"))

        debug_output = "\n".join(debug_lines)

        return (conversation, conditioning, formatted_text, debug_output)


class ZImageTextEncoder:
    """
    Z-Image Text Encoder - transparent and editable.

    Most users only need this node - it handles a complete first turn.
    For multi-turn conversations, connect conversation output to ZImageTurnBuilder,
    then connect that back to another encoder's conversation_override.

    Outputs:
    - conditioning: For KSampler
    - formatted_prompt: See exactly what gets encoded
    - debug_output: Detailed breakdown for troubleshooting
    - conversation: Chain to ZImageTurnBuilder for multi-turn
    """

    @classmethod
    def INPUT_TYPES(cls):
        templates = get_templates()
        template_names = ["none"] + sorted(templates.keys())

        return {
            "required": {
                "clip": ("CLIP",),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Your prompt - what you want the model to generate"
                }),
            },
            "optional": {
                "conversation_override": (ZIMAGE_CONVERSATION_TYPE, {
                    "tooltip": "Connect from ZImageTurnBuilder - uses this conversation instead of building one"
                }),
                "template_preset": (template_names, {
                    "default": "none",
                    "tooltip": "Select template - auto-fills system_prompt field (editable)"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System prompt - auto-filled by template, edit freely"
                }),
                "add_think_block": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add <think></think> block (auto-enabled if thinking_content provided)"
                }),
                "thinking_content": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Content inside <think>...</think> tags (auto-enables think block)"
                }),
                "assistant_content": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Content AFTER </think> tags (what assistant says after thinking)"
                }),
                "raw_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "RAW MODE: Bypass ALL formatting. Write your own <|im_start|> tokens. All other fields ignored when set."
                }),
                "strip_key_quotes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove double quotes from JSON-style keys in prompts. Converts \"subject\": to subject: to prevent key names appearing as text in image."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING", ZIMAGE_CONVERSATION_TYPE)
    RETURN_NAMES = ("conditioning", "formatted_prompt", "debug_output", "conversation")
    FUNCTION = "encode"
    CATEGORY = "ZImage/Encoding"
    TITLE = "Z-Image Text Encoder"

    def encode(
        self,
        clip,
        user_prompt: str,
        conversation_override: Optional[Dict[str, Any]] = None,
        template_preset: str = "none",
        system_prompt: str = "",
        add_think_block: bool = False,
        thinking_content: str = "",
        assistant_content: str = "",
        raw_prompt: str = "",
        strip_key_quotes: bool = False,
    ) -> Tuple[Any, str, str, Dict[str, Any]]:

        # Apply key quote filtering if enabled
        if strip_key_quotes:
            user_prompt = re.sub(r'"([^"]+)":', r'\1:', user_prompt)
            thinking_content = re.sub(r'"([^"]+)":', r'\1:', thinking_content)
            assistant_content = re.sub(r'"([^"]+)":', r'\1:', assistant_content)

        # Track mode for debug output
        mode = "direct"
        effective_system = ""
        use_think_block = False
        conversation_out = None

        # CONVERSATION OVERRIDE: multi-turn from ZImageTurnBuilder
        if conversation_override is not None and conversation_override.get("messages"):
            mode = "conversation_override"
            formatted_text = format_conversation(conversation_override)
            # Pass through the conversation (already complete)
            conversation_out = conversation_override

        # RAW MODE: complete control
        elif raw_prompt.strip():
            mode = "raw"
            formatted_text = raw_prompt
            # No conversation output for raw mode (user has full control)
            conversation_out = None

        else:
            # Determine system prompt: use provided, or fallback to template file
            effective_system = system_prompt.strip()
            if not effective_system and template_preset and template_preset != "none":
                # JS didn't fill system_prompt - load from template file as fallback
                templates = get_templates()
                if template_preset in templates:
                    effective_system = templates[template_preset]
                    logger.debug(f"Loaded template '{template_preset}' from file (JS fallback)")

            # Auto-enable think block if thinking_content provided
            use_think_block = add_think_block or bool(thinking_content.strip())

            # Build formatted prompt with chat template
            formatted_text = self._format_prompt(user_prompt, effective_system, use_think_block, thinking_content, assistant_content)

            # Build conversation for chaining
            conversation_out = {
                "enable_thinking": use_think_block,
                "messages": []
            }

            # Add system message if present
            if effective_system:
                conversation_out["messages"].append({
                    "role": "system",
                    "content": effective_system
                })

            # Add user message
            conversation_out["messages"].append({
                "role": "user",
                "content": user_prompt.strip()
            })

            # Add assistant message (with thinking if enabled)
            assistant_msg = {
                "role": "assistant",
                "content": assistant_content.strip() if assistant_content else ""
            }
            if use_think_block:
                assistant_msg["thinking"] = thinking_content.strip() if thinking_content else ""
            conversation_out["messages"].append(assistant_msg)

        # Build debug output
        debug_lines = ["=== Z-Image Text Encoder Debug ==="]
        if strip_key_quotes:
            debug_lines.append("Key quote filter: ON (removing quotes from JSON keys)")
        if mode == "conversation_override":
            msg_count = len(conversation_override.get("messages", []))
            debug_lines.append(f"Mode: Conversation Override ({msg_count} messages)")
        elif mode == "raw":
            debug_lines.append("Mode: Raw (bypassing all formatting)")
        else:
            debug_lines.append("Mode: Direct")
            debug_lines.append(f"Template: {template_preset}")
            debug_lines.append(f"System prompt: {len(effective_system)} chars")
            debug_lines.append(f"User prompt: {len(user_prompt.strip())} chars")
            debug_lines.append(f"Think block: {'enabled' if use_think_block else 'disabled'}")
            if use_think_block:
                debug_lines.append(f"Thinking content: {len(thinking_content.strip())} chars")
            debug_lines.append(f"Assistant content: {len(assistant_content.strip())} chars")

        debug_lines.append("")
        debug_lines.append("=== Formatted Prompt ===")
        # Escape angle brackets for HTML preview to show think tags
        debug_lines.append(formatted_text.replace("<", "&lt;").replace(">", "&gt;"))
        debug_lines.append("")
        debug_lines.append("=== Think Tag Check ===")
        debug_lines.append(f"Contains '<think>': {'<think>' in formatted_text}")
        debug_lines.append(f"Contains '</think>': {'</think>' in formatted_text}")

        # Estimate tokens (rough: ~4 chars per token for English)
        debug_lines.append("")
        debug_lines.append("=== Token Estimate ===")
        est_tokens = len(formatted_text) // 4
        debug_lines.append(f"~{est_tokens} tokens (estimate)")

        debug_output = "\n".join(debug_lines)

        # Log formatted prompt to console
        print(f"\n[Z-Image] Formatted prompt:\n{formatted_text}\n")

        # Encode
        tokens = clip.tokenize(formatted_text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        return (conditioning, formatted_text, debug_output, conversation_out)

    def _format_prompt(self, user_prompt: str, system_prompt: str = "", add_think_block: bool = False, thinking_content: str = "", assistant_content: str = "") -> str:
        """
        Format prompt following Qwen3-4B chat template from tokenizer_config.json.

        Structure:
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {user_prompt}<|im_end|>
        <|im_start|>assistant
        <think>
        {thinking_content}
        </think>

        {assistant_content}
        """
        parts = []

        if system_prompt.strip():
            parts.append(f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>")

        parts.append(f"<|im_start|>user\n{user_prompt}<|im_end|>")

        # Build assistant section - match Qwen3 template exactly
        # If assistant_content provided, close with <|im_end|>. Otherwise leave open (diffusers default).
        if add_think_block:
            think_inner = thinking_content.strip() if thinking_content.strip() else ""
            assistant_inner = assistant_content.strip() if assistant_content.strip() else ""
            if assistant_inner:
                # User provided assistant content - close the message
                parts.append(f"<|im_start|>assistant\n<think>\n{think_inner}\n</think>\n\n{assistant_inner}<|im_end|>")
            else:
                # No assistant content - leave open (matches diffusers)
                parts.append(f"<|im_start|>assistant\n<think>\n{think_inner}\n</think>\n\n")
        else:
            assistant_inner = assistant_content.strip() if assistant_content.strip() else ""
            if assistant_inner:
                # User provided assistant content - close the message
                parts.append(f"<|im_start|>assistant\n{assistant_inner}<|im_end|>")
            else:
                # No assistant content - leave open (matches diffusers)
                parts.append(f"<|im_start|>assistant\n")

        return "\n".join(parts)


# Export templates for JS
def get_z_image_template_systems() -> Dict[str, str]:
    """Get template systems for JS auto-fill. Called by __init__.py for API."""
    return get_templates()


class PromptKeyFilter:
    """
    Filter double quotes from JSON-style keys in prompts.

    Problem: When prompts contain "key": "value" format, the quoted key names
    can appear as visible text in the generated image.

    Solution: Strip quotes from keys only (before the colon), leaving values intact.

    Before: {"subject": "a man juggling", "style": "photo"}
    After:  {subject: "a man juggling", style: "photo"}

    Two ways to use:
    1. Connect text from another node (LLM output, etc.) to text_input
    2. Paste/type directly into the text field

    If text_input is connected, it takes priority over the text field.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "Paste or type text here. If text_input is connected, this is ignored."}),
                "strip_key_quotes": ("BOOLEAN", {"default": True, "tooltip": "Remove double quotes from JSON-style keys (e.g., \"subject\": becomes subject:). Prevents key names appearing as text in image."}),
            },
            "optional": {
                "text_input": ("STRING", {"forceInput": True, "tooltip": "Connect text from another node. Takes priority over the text field."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "filter_text"
    CATEGORY = "QwenImage/Utilities"

    DESCRIPTION = "Filter JSON-style key quotes from prompts. Converts \"key\": to key: to prevent label text appearing in generated images. Paste directly or connect from another node."

    def filter_text(self, text: str, strip_key_quotes: bool = True, text_input: Optional[str] = None) -> Tuple[str]:
        # Use connected input if provided, otherwise use the text field
        source_text = text_input if text_input is not None else text

        if not strip_key_quotes:
            return (source_text,)

        # Match "key": pattern - double quote, non-quote chars, double quote, colon
        # Replace with just the key name and colon (no quotes)
        filtered = re.sub(r'"([^"]+)":', r'\1:', source_text)

        return (filtered,)


NODE_CLASS_MAPPINGS = {
    "ZImageTextEncoder": ZImageTextEncoder,
    "ZImageTurnBuilder": ZImageTurnBuilder,
    "PromptKeyFilter": PromptKeyFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageTextEncoder": "Z-Image Text Encoder",
    "ZImageTurnBuilder": "Z-Image Turn Builder",
    "PromptKeyFilter": "Prompt Key Filter",
}
