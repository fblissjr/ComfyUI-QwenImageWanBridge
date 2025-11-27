"""
Template Influence Analyzer Node for ComfyUI

This node tests whether system prompts actually affect embeddings.
Connect to a loaded Qwen CLIP model to analyze embedding differences.

Usage:
1. Load Qwen2.5-VL CLIP model
2. Connect to this node
3. Provide test prompt and templates to compare
4. View analysis output

This uses the already-loaded model in ComfyUI, avoiding duplicate loading.
"""

import torch
import logging
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)


class TemplateInfluenceAnalyzer:
    """
    Analyze how different system prompts affect text embeddings.

    This node helps validate whether the template system is effective
    by comparing embeddings from the same user prompt with different templates.
    """

    def __init__(self):
        """Initialize with default templates for comparison"""
        self.default_templates = {
            "empty": "",
            "minimal": "Generate an image.",
            "default_t2i": "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:",
            "horror": "You are a horror film director assistant. Describe the video by detailing ominous settings, unsettling elements, deep shadows, and suspenseful atmosphere.",
            "comedy": "You are a comedy director assistant. Focus on timing, humor, lighthearted moments, and playful visual elements.",
            "cinematic": "You are a cinematic video director assistant. Describe with emphasis on dramatic narrative, cinematic composition, dynamic movement, and professional camera techniques.",
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "test_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A warrior walking through an ancient temple, golden light streaming through pillars",
                    "tooltip": "The user prompt to test with different templates"
                }),
            },
            "optional": {
                "template_a_name": ("STRING", {
                    "default": "empty",
                    "tooltip": "Name for first template (for display)"
                }),
                "template_a": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "First system prompt template to compare"
                }),
                "template_b_name": ("STRING", {
                    "default": "cinematic",
                    "tooltip": "Name for second template (for display)"
                }),
                "template_b": ("STRING", {
                    "multiline": True,
                    "default": "You are a cinematic video director assistant. Describe with emphasis on dramatic narrative, cinematic composition, dynamic movement, and professional camera techniques.",
                    "tooltip": "Second system prompt template to compare"
                }),
                "drop_idx": ("INT", {
                    "default": 34,
                    "min": 0,
                    "max": 128,
                    "tooltip": "Number of tokens to drop (34 for T2I, 64 for edit modes)"
                }),
                "run_full_comparison": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Compare against all built-in templates (slower)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("analysis_output",)
    FUNCTION = "analyze"
    CATEGORY = "QwenImage/Utilities"
    TITLE = "Template Influence Analyzer"
    DESCRIPTION = "Test whether system prompts affect embeddings"

    def format_template(self, user_prompt: str, system_prompt: str) -> str:
        """Format with DiffSynth-style chat markers"""
        if not system_prompt:
            return user_prompt
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def get_embedding(
        self,
        clip,
        user_prompt: str,
        system_prompt: str,
        drop_idx: int
    ) -> torch.Tensor:
        """Get embedding for a prompt with system template"""
        formatted = self.format_template(user_prompt, system_prompt)

        # Tokenize using ComfyUI's clip.tokenize
        tokens = clip.tokenize(formatted)

        # Encode
        cond = clip.encode_from_tokens_scheduled(tokens)

        # Extract embedding tensor
        if cond and len(cond) > 0:
            emb = cond[0][0]  # [batch, seq, hidden]
            return emb
        return None

    def compute_metrics(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor
    ) -> Dict[str, float]:
        """Compute similarity metrics between embeddings"""
        # Ensure same length
        min_len = min(emb_a.shape[1], emb_b.shape[1])
        emb_a = emb_a[:, :min_len, :].float()
        emb_b = emb_b[:, :min_len, :].float()

        # Flatten for overall comparison
        flat_a = emb_a.reshape(-1)
        flat_b = emb_b.reshape(-1)

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            flat_a.unsqueeze(0),
            flat_b.unsqueeze(0)
        ).item()

        # L2 distance
        l2_dist = torch.norm(flat_a - flat_b).item()

        # Per-token analysis
        token_diffs = torch.norm(emb_a - emb_b, dim=-1).squeeze()
        n_tokens = len(token_diffs)

        first_quarter = token_diffs[:n_tokens//4].mean().item() if n_tokens >= 4 else 0
        last_quarter = token_diffs[-n_tokens//4:].mean().item() if n_tokens >= 4 else 0

        return {
            "cosine_similarity": cos_sim,
            "l2_distance": l2_dist,
            "sequence_length": min_len,
            "first_quarter_diff": first_quarter,
            "last_quarter_diff": last_quarter,
        }

    def interpret_results(self, cos_sim: float) -> str:
        """Interpret cosine similarity value"""
        if cos_sim > 0.995:
            return "IDENTICAL (>0.995) - Templates have NO effect"
        elif cos_sim > 0.99:
            return "VERY SIMILAR (0.99-0.995) - Minimal template effect"
        elif cos_sim > 0.95:
            return "DIFFERENT (0.95-0.99) - Templates DO affect embeddings"
        else:
            return "VERY DIFFERENT (<0.95) - Strong template effect"

    def analyze(
        self,
        clip,
        test_prompt: str,
        template_a_name: str = "empty",
        template_a: str = "",
        template_b_name: str = "cinematic",
        template_b: str = "",
        drop_idx: int = 34,
        run_full_comparison: bool = False
    ) -> Tuple[str]:
        """Analyze template influence on embeddings"""

        output_lines = []
        output_lines.append("=" * 60)
        output_lines.append("TEMPLATE INFLUENCE ANALYSIS")
        output_lines.append("=" * 60)
        output_lines.append(f"Test prompt: {test_prompt[:50]}...")
        output_lines.append(f"Drop index: {drop_idx}")
        output_lines.append("")

        if run_full_comparison:
            # Compare all built-in templates
            output_lines.append("Running full comparison against built-in templates...")
            output_lines.append("")

            embeddings = {}
            for name, template in self.default_templates.items():
                emb = self.get_embedding(clip, test_prompt, template, drop_idx)
                if emb is not None:
                    embeddings[name] = emb
                    output_lines.append(f"  {name}: shape {tuple(emb.shape)}")

            output_lines.append("")
            output_lines.append("-" * 60)
            output_lines.append(f"{'Pair':<30} | {'Cosine':<10} | {'L2 Dist':<10}")
            output_lines.append("-" * 60)

            names = list(embeddings.keys())
            results = []
            for i, name_a in enumerate(names):
                for name_b in names[i+1:]:
                    metrics = self.compute_metrics(embeddings[name_a], embeddings[name_b])
                    pair = f"{name_a} vs {name_b}"
                    output_lines.append(
                        f"{pair:<30} | {metrics['cosine_similarity']:<10.6f} | {metrics['l2_distance']:<10.2f}"
                    )
                    results.append(metrics)

            # Summary
            if results:
                cos_sims = [r["cosine_similarity"] for r in results]
                min_sim = min(cos_sims)
                max_sim = max(cos_sims)

                output_lines.append("")
                output_lines.append("=" * 60)
                output_lines.append("SUMMARY")
                output_lines.append("=" * 60)
                output_lines.append(f"Similarity range: {min_sim:.6f} - {max_sim:.6f}")
                output_lines.append(f"Interpretation: {self.interpret_results(min_sim)}")

        else:
            # Compare just the two provided templates
            output_lines.append(f"Comparing: {template_a_name} vs {template_b_name}")
            output_lines.append("")

            emb_a = self.get_embedding(clip, test_prompt, template_a, drop_idx)
            emb_b = self.get_embedding(clip, test_prompt, template_b, drop_idx)

            if emb_a is None or emb_b is None:
                output_lines.append("ERROR: Failed to get embeddings")
                return ("\n".join(output_lines),)

            output_lines.append(f"{template_a_name} embedding shape: {tuple(emb_a.shape)}")
            output_lines.append(f"{template_b_name} embedding shape: {tuple(emb_b.shape)}")
            output_lines.append("")

            metrics = self.compute_metrics(emb_a, emb_b)

            output_lines.append("-" * 60)
            output_lines.append("METRICS")
            output_lines.append("-" * 60)
            output_lines.append(f"Cosine Similarity: {metrics['cosine_similarity']:.6f}")
            output_lines.append(f"L2 Distance: {metrics['l2_distance']:.4f}")
            output_lines.append(f"Sequence Length: {metrics['sequence_length']}")
            output_lines.append("")

            output_lines.append("Position Analysis:")
            output_lines.append(f"  First quarter avg diff: {metrics['first_quarter_diff']:.6f}")
            output_lines.append(f"  Last quarter avg diff: {metrics['last_quarter_diff']:.6f}")

            if metrics['first_quarter_diff'] > metrics['last_quarter_diff'] * 1.2:
                output_lines.append("  -> Position decay DETECTED (first tokens more affected)")
            else:
                output_lines.append("  -> No significant position decay")

            output_lines.append("")
            output_lines.append("=" * 60)
            output_lines.append("INTERPRETATION")
            output_lines.append("=" * 60)
            output_lines.append(self.interpret_results(metrics['cosine_similarity']))

            # Detailed advice
            output_lines.append("")
            if metrics['cosine_similarity'] > 0.995:
                output_lines.append("Advice: These templates produce nearly identical embeddings.")
                output_lines.append("The system prompt is likely not affecting generation.")
                output_lines.append("Consider testing with more divergent templates or")
                output_lines.append("investigating if token dropping is too aggressive.")
            elif metrics['cosine_similarity'] > 0.99:
                output_lines.append("Advice: Small differences detected. Template effect is minimal.")
                output_lines.append("May need output quality testing to confirm if meaningful.")
            else:
                output_lines.append("Advice: Meaningful differences detected!")
                output_lines.append("Templates ARE affecting embeddings. Test output quality")
                output_lines.append("to confirm this translates to generation differences.")

        return ("\n".join(output_lines),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "TemplateInfluenceAnalyzer": TemplateInfluenceAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TemplateInfluenceAnalyzer": "Template Influence Analyzer",
}
