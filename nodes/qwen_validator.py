"""
Qwen Reference Validator
Phase 1 of Smart Labeling Implementation
Validates Picture references in prompts
"""

import re
from typing import List, Tuple, Optional, Dict
from .qwen_config import QwenConfig
from .qwen_logger import QwenLogger

class ReferenceValidator:
    """Validates Picture references in prompts"""

    def __init__(self):
        self.logger = QwenLogger()
        self.patterns = QwenConfig.get_picture_patterns()

    def extract_references(self, text: str) -> List[int]:
        """
        Extract all Picture number references from text

        Args:
            text: Prompt text to scan

        Returns:
            Sorted list of unique picture numbers referenced
        """
        if not text:
            return []

        # Limit scan length for performance
        scan_text = text[:QwenConfig.MAX_PROMPT_SCAN_LENGTH]

        refs = []
        for pattern in self.patterns:
            matches = pattern.finditer(scan_text)
            for match in matches:
                # Extract number from match
                num_match = re.search(r'\d+', match.group())
                if num_match:
                    refs.append(int(num_match.group()))

        unique_refs = sorted(set(refs))

        if unique_refs:
            self.logger.debug(f"Found Picture references: {unique_refs}")

        return unique_refs

    def validate(
        self,
        text: str,
        num_images: int,
        mode: str = "off"
    ) -> Tuple[bool, Optional[str], Dict]:
        """
        Validate references match available images

        Args:
            text: Prompt text containing Picture references
            num_images: Number of images actually available
            mode: Validation mode (off, warn, error, verbose)

        Returns:
            Tuple of (is_valid, error_message, details_dict)
        """
        details = {
            "num_images": num_images,
            "mode": mode
        }

        if mode == "off":
            return True, None, details

        refs = self.extract_references(text)
        details["references_found"] = refs

        if not refs:
            return True, None, details

        max_ref = max(refs)
        min_ref = min(refs)
        details["max_ref"] = max_ref
        details["min_ref"] = min_ref

        issues = []

        # Check if max reference exceeds available images
        if max_ref > num_images:
            issue = f"Picture {max_ref} referenced but only {num_images} images available"
            issues.append(issue)
            details["over_reference"] = True

        # Check for invalid references (< 1)
        if min_ref < 1:
            issue = f"Invalid Picture {min_ref} reference (must be >= 1)"
            issues.append(issue)
            details["invalid_reference"] = True

        # Check for gaps in references (verbose mode only)
        if mode == "verbose" and num_images > 0:
            expected = set(range(1, num_images + 1))
            referenced = set(refs)
            unused = expected - referenced

            if unused:
                issue = f"Unused images: Picture {sorted(unused)}"
                issues.append(issue)
                details["unused_images"] = sorted(unused)

        # Handle issues based on mode
        if issues:
            message = "; ".join(issues)

            if mode == "warn" or mode == "verbose":
                self.logger.log_validation("warn", message, details)
                return True, message, details  # Valid but with warnings
            elif mode == "error":
                self.logger.log_validation("error", message, details)
                return False, message, details  # Invalid

        return True, None, details

    def suggest_corrections(self, text: str, num_images: int) -> str:
        """
        Suggest corrections for invalid references

        Args:
            text: Original prompt text
            num_images: Number of available images

        Returns:
            Suggested corrected prompt
        """
        refs = self.extract_references(text)
        if not refs:
            return text

        corrected = text
        for ref in refs:
            if ref > num_images:
                # Suggest using last available image
                old_pattern = f"Picture {ref}"
                new_pattern = f"Picture {num_images}"
                corrected = corrected.replace(old_pattern, new_pattern)

                self.logger.info(
                    f"Suggestion: Replace '{old_pattern}' with '{new_pattern}'"
                )

        return corrected

    def analyze_prompt(self, text: str) -> Dict:
        """
        Analyze prompt for Picture reference patterns

        Args:
            text: Prompt to analyze

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "has_references": False,
            "reference_count": 0,
            "reference_numbers": [],
            "reference_patterns": [],
            "suggested_num_images": 0
        }

        refs = self.extract_references(text)

        if refs:
            analysis["has_references"] = True
            analysis["reference_count"] = len(refs)
            analysis["reference_numbers"] = refs
            analysis["suggested_num_images"] = max(refs)

            # Find which patterns matched
            for pattern in self.patterns:
                if pattern.search(text[:QwenConfig.MAX_PROMPT_SCAN_LENGTH]):
                    analysis["reference_patterns"].append(pattern.pattern)

        return analysis