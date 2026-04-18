"""
Template Loader for Qwen Image Nodes
Single source of truth for all system prompt templates
"""

import os
import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class TemplateLoader:
    """Loads and caches template files from nodes/templates/"""

    def __init__(self):
        self.templates: Dict[str, Dict[str, Any]] = {}
        self.template_dir = Path(__file__).parent / "templates"
        self._load_templates()

    def _parse_frontmatter(self, content: str) -> tuple[Dict[str, Any], str]:
        """Parse YAML frontmatter from markdown file"""
        lines = content.split('\n')

        if not lines or lines[0].strip() != '---':
            return {}, content

        # Find closing ---
        end_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                end_idx = i
                break

        if end_idx is None:
            return {}, content

        # Parse frontmatter (simple key: value parser)
        metadata = {}
        for line in lines[1:end_idx]:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                # Convert booleans
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False

                metadata[key] = value

        # System prompt is everything after frontmatter
        system_prompt = '\n'.join(lines[end_idx + 1:]).strip()

        return metadata, system_prompt

    def _load_templates(self):
        """Load all template files from templates/ directory"""
        if not self.template_dir.exists():
            logger.warning(f"Templates directory not found: {self.template_dir}")
            return

        for template_file in sorted(self.template_dir.glob("*.md")):
            template_name = template_file.stem

            try:
                content = template_file.read_text(encoding='utf-8')
                metadata, system_prompt = self._parse_frontmatter(content)

                # Build template dict
                template = {
                    "system": system_prompt,
                    "mode": metadata.get("mode", "text_to_image"),
                    "vision": metadata.get("vision", False),
                }

                # Optional metadata
                if "use_picture_format" in metadata:
                    template["use_picture_format"] = metadata["use_picture_format"]
                if "experimental" in metadata:
                    template["experimental"] = metadata["experimental"]
                if "no_template" in metadata:
                    template["no_template"] = metadata["no_template"]

                self.templates[template_name] = template
                logger.debug(f"Loaded template: {template_name} (mode: {template['mode']})")

            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")

    def get_template(self, name: str) -> Dict[str, Any]:
        """Get template by name"""
        return self.templates.get(name, {})

    def get_template_names(self) -> List[str]:
        """Get list of all template names in order"""
        return list(self.templates.keys())

    def get_modes(self) -> List[str]:
        """Get list of unique modes from all templates"""
        modes = set()
        for template in self.templates.values():
            modes.add(template["mode"])
        return sorted(modes)


# Global instance - loaded once on module import
_template_loader = None

def get_template_loader() -> TemplateLoader:
    """Get singleton template loader instance"""
    global _template_loader
    if _template_loader is None:
        _template_loader = TemplateLoader()
        logger.info(f"Loaded {len(_template_loader.templates)} templates")
    return _template_loader
