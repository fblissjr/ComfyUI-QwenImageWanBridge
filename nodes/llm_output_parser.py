"""
LLM Output Parser for ComfyUI

Parses structured LLM output (JSON/YAML) or plain text and exposes fields
for direct wiring to Z-Image encoder inputs.

Works with any LLM node - not tied to any specific implementation.
"""

import json
import re
import copy
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Import conversation type from z_image_encoder
from .z_image_encoder import ZIMAGE_CONVERSATION_TYPE

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def strip_code_fences(text: str) -> str:
    """
    Remove markdown code fences from LLM output.

    Handles: ```json ... ```, ```yaml ... ```, ``` ... ```
    """
    text = text.strip()

    # Handle ```json or ```yaml
    for lang in ["json", "yaml", "yml"]:
        marker = f"```{lang}"
        if marker in text.lower():
            # Find the marker (case insensitive)
            lower_text = text.lower()
            start_idx = lower_text.find(marker)
            if start_idx != -1:
                # Skip past the marker and newline
                content_start = start_idx + len(marker)
                remainder = text[content_start:]
                # Find closing ```
                end_idx = remainder.find("```")
                if end_idx != -1:
                    return remainder[:end_idx].strip()
                return remainder.strip()

    # Handle plain ``` ... ```
    if text.startswith("```") and "```" in text[3:]:
        # Remove opening ```
        content = text[3:]
        # Skip language identifier if on first line
        first_newline = content.find("\n")
        if first_newline != -1 and first_newline < 20:  # Reasonable lang identifier length
            first_line = content[:first_newline].strip()
            if first_line.isalnum() or first_line == "":
                content = content[first_newline + 1:]
        # Find closing ```
        end_idx = content.rfind("```")
        if end_idx != -1:
            return content[:end_idx].strip()

    return text


def extract_field(data: Dict[str, Any], key: str) -> str:
    """
    Extract a field from a dict, supporting dot notation for nested keys.

    Examples:
        extract_field({"user_prompt": "test"}, "user_prompt") -> "test"
        extract_field({"result": {"prompt": "test"}}, "result.prompt") -> "test"
    """
    if not key or not data:
        return ""

    parts = key.split(".")
    value = data

    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return ""

    # Convert complex types to string
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2)

    return str(value) if value is not None else ""


def unwrap_nested(data: Any) -> Dict[str, Any]:
    """
    Unwrap common LLM output patterns.

    Handles:
        {"result": {...}} -> {...}
        {"data": {...}} -> {...}
        {"output": {...}} -> {...}
        [{"prompt": ...}] -> {"prompt": ...} (first item)
    """
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            return data[0]
        return {}

    if isinstance(data, dict):
        # Common wrapper keys
        for wrapper_key in ["result", "data", "output", "response"]:
            if wrapper_key in data and isinstance(data[wrapper_key], dict):
                return data[wrapper_key]

    return data if isinstance(data, dict) else {}


class LLMOutputParser:
    """
    Parse LLM output (JSON/YAML/text) and expose fields for Z-Image encoder.

    Supports auto-detection, explicit parsing modes, and multi-turn conversation building.
    Handles markdown code fences, nested structures, and common LLM output patterns.

    Works with any LLM node - connect the text output and wire fields to Z-Image inputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {
                    "forceInput": True,
                    "tooltip": "LLM output text to parse (connect from any LLM node)"
                }),
                "parse_mode": (["auto", "passthrough", "json", "yaml"], {
                    "default": "auto",
                    "tooltip": "auto: try JSON then YAML; passthrough: use raw text as user_prompt"
                }),
            },
            "optional": {
                "user_prompt_key": ("STRING", {
                    "default": "user_prompt",
                    "tooltip": "JSON/YAML key for user_prompt (supports dot notation: result.prompt)"
                }),
                "system_prompt_key": ("STRING", {
                    "default": "system_prompt",
                    "tooltip": "JSON/YAML key for system_prompt"
                }),
                "thinking_key": ("STRING", {
                    "default": "thinking",
                    "tooltip": "JSON/YAML key for thinking_content"
                }),
                "assistant_key": ("STRING", {
                    "default": "assistant",
                    "tooltip": "JSON/YAML key for assistant_content"
                }),
                "fallback_to_passthrough": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If parsing fails, output raw text as user_prompt instead of empty"
                }),
                "strip_quotes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove quotes from extracted fields (for JSON that may render as text)"
                }),
                "previous_conversation": (ZIMAGE_CONVERSATION_TYPE, {
                    "tooltip": "Connect from Z-Image encoder/turn builder to build multi-turn conversation"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", ZIMAGE_CONVERSATION_TYPE)
    RETURN_NAMES = ("user_prompt", "system_prompt", "thinking_content", "assistant_content", "raw_text", "parse_status", "conversation")
    FUNCTION = "parse"
    CATEGORY = "ZImage/Utilities"
    TITLE = "LLM Output Parser"

    DESCRIPTION = "Parse LLM output (JSON/YAML/text) to Z-Image encoder fields. Handles code fences, nested keys, and multi-turn conversations."

    def parse(
        self,
        text_input: str,
        parse_mode: str,
        user_prompt_key: str = "user_prompt",
        system_prompt_key: str = "system_prompt",
        thinking_key: str = "thinking",
        assistant_key: str = "assistant",
        fallback_to_passthrough: bool = True,
        strip_quotes: bool = False,
        previous_conversation: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str, str, str, str, Optional[Dict[str, Any]]]:

        # Edge case: empty input
        if not text_input or not text_input.strip():
            return ("", "", "", "", "", "error: empty input", None)

        raw_text = text_input
        user_prompt = ""
        system_prompt = ""
        thinking_content = ""
        assistant_content = ""
        parse_status = ""
        conversation = None

        # PASSTHROUGH MODE
        if parse_mode == "passthrough":
            user_prompt = text_input.strip()
            parse_status = "passthrough"

        # JSON MODE
        elif parse_mode == "json":
            try:
                cleaned = strip_code_fences(text_input)
                data = json.loads(cleaned)
                data = unwrap_nested(data)

                user_prompt = extract_field(data, user_prompt_key)
                system_prompt = extract_field(data, system_prompt_key)
                thinking_content = extract_field(data, thinking_key)
                assistant_content = extract_field(data, assistant_key)

                # If no user_prompt found with key, check common alternatives
                if not user_prompt:
                    for alt_key in ["prompt", "text", "content", "message"]:
                        user_prompt = extract_field(data, alt_key)
                        if user_prompt:
                            break

                parse_status = "success: json"

            except json.JSONDecodeError as e:
                if fallback_to_passthrough:
                    user_prompt = text_input.strip()
                    parse_status = f"fallback: json parse failed ({str(e)[:50]})"
                else:
                    parse_status = f"error: json parse failed ({str(e)[:50]})"

        # YAML MODE
        elif parse_mode == "yaml":
            if not YAML_AVAILABLE:
                if fallback_to_passthrough:
                    user_prompt = text_input.strip()
                    parse_status = "fallback: yaml module not installed"
                else:
                    parse_status = "error: yaml module not installed"
            else:
                try:
                    cleaned = strip_code_fences(text_input)
                    data = yaml.safe_load(cleaned)
                    data = unwrap_nested(data)

                    user_prompt = extract_field(data, user_prompt_key)
                    system_prompt = extract_field(data, system_prompt_key)
                    thinking_content = extract_field(data, thinking_key)
                    assistant_content = extract_field(data, assistant_key)

                    # If no user_prompt found with key, check common alternatives
                    if not user_prompt:
                        for alt_key in ["prompt", "text", "content", "message"]:
                            user_prompt = extract_field(data, alt_key)
                            if user_prompt:
                                break

                    parse_status = "success: yaml"

                except Exception as e:
                    if fallback_to_passthrough:
                        user_prompt = text_input.strip()
                        parse_status = f"fallback: yaml parse failed ({str(e)[:50]})"
                    else:
                        parse_status = f"error: yaml parse failed ({str(e)[:50]})"

        # AUTO MODE: try JSON, then YAML, then passthrough
        elif parse_mode == "auto":
            parsed = False
            cleaned = strip_code_fences(text_input)

            # Try JSON first
            try:
                data = json.loads(cleaned)
                data = unwrap_nested(data)

                user_prompt = extract_field(data, user_prompt_key)
                system_prompt = extract_field(data, system_prompt_key)
                thinking_content = extract_field(data, thinking_key)
                assistant_content = extract_field(data, assistant_key)

                # If no user_prompt found with key, check common alternatives
                if not user_prompt:
                    for alt_key in ["prompt", "text", "content", "message"]:
                        user_prompt = extract_field(data, alt_key)
                        if user_prompt:
                            break

                parse_status = "success: json (auto)"
                parsed = True

            except json.JSONDecodeError:
                pass

            # Try YAML if JSON failed
            if not parsed and YAML_AVAILABLE:
                try:
                    data = yaml.safe_load(cleaned)
                    if isinstance(data, dict):
                        data = unwrap_nested(data)

                        user_prompt = extract_field(data, user_prompt_key)
                        system_prompt = extract_field(data, system_prompt_key)
                        thinking_content = extract_field(data, thinking_key)
                        assistant_content = extract_field(data, assistant_key)

                        # If no user_prompt found with key, check common alternatives
                        if not user_prompt:
                            for alt_key in ["prompt", "text", "content", "message"]:
                                user_prompt = extract_field(data, alt_key)
                                if user_prompt:
                                    break

                        parse_status = "success: yaml (auto)"
                        parsed = True
                except Exception:
                    pass

            # Fallback to passthrough
            if not parsed:
                if fallback_to_passthrough:
                    user_prompt = text_input.strip()
                    parse_status = "passthrough (auto fallback)"
                else:
                    parse_status = "error: auto parse failed"

        # Apply quote stripping if enabled
        if strip_quotes:
            user_prompt = re.sub(r'"([^"]+)":', r'\1:', user_prompt)
            user_prompt = user_prompt.replace('"', '')
            system_prompt = re.sub(r'"([^"]+)":', r'\1:', system_prompt)
            system_prompt = system_prompt.replace('"', '')
            thinking_content = re.sub(r'"([^"]+)":', r'\1:', thinking_content)
            thinking_content = thinking_content.replace('"', '')
            assistant_content = re.sub(r'"([^"]+)":', r'\1:', assistant_content)
            assistant_content = assistant_content.replace('"', '')

        # Build conversation if previous_conversation provided
        if previous_conversation is not None and previous_conversation.get("messages"):
            # Deep copy to avoid modifying original
            conversation = {
                "enable_thinking": previous_conversation.get("enable_thinking", False),
                "is_final": True,  # Default to final
                "messages": copy.deepcopy(previous_conversation.get("messages", [])),
            }

            enable_thinking = conversation["enable_thinking"]

            # Add user message (if we have user_prompt)
            if user_prompt:
                conversation["messages"].append({
                    "role": "user",
                    "content": user_prompt,
                })

                # Add assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": assistant_content,
                }
                if enable_thinking:
                    assistant_msg["thinking"] = thinking_content
                conversation["messages"].append(assistant_msg)

        # Log for debugging
        logger.debug(f"[LLM Parser] Status: {parse_status}")
        logger.debug(f"[LLM Parser] user_prompt: {len(user_prompt)} chars")

        return (user_prompt, system_prompt, thinking_content, assistant_content,
                raw_text, parse_status, conversation)


NODE_CLASS_MAPPINGS = {
    "LLMOutputParser": LLMOutputParser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMOutputParser": "LLM Output Parser",
}
