"""
Pico Prompt Sampler Node

Samples prompts from the Pico-Banana-400K dataset for use with Z-Image
and other text-to-image encoders.

Requires:
- duckdb (pip install duckdb)
- PICO_DB_PATH environment variable pointing to image_edits.duckdb
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

# DuckDB import with fallback
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None
    print("[PicoPromptSampler] WARNING: duckdb not installed. Install with: pip install duckdb")

# Database path from environment variable (required)
PICO_DB_PATH = Path(os.environ.get("PICO_DB_PATH", ""))

# Debug: Log availability at import time
if PICO_DB_PATH:
    print(f"[PicoPromptSampler] PICO_DB_PATH is set")
    print(f"[PicoPromptSampler] Path exists: {PICO_DB_PATH.exists()}")
else:
    print("[PicoPromptSampler] PICO_DB_PATH not set - set environment variable to database path")
print(f"[PicoPromptSampler] DuckDB available: {DUCKDB_AVAILABLE}")

# Category cache
_category_cache = None
_prompt_cache = {}


def get_pico_categories() -> List[str]:
    """Get list of all Pico edit categories."""
    global _category_cache

    if _category_cache is not None:
        return _category_cache

    if not DUCKDB_AVAILABLE:
        return ["(duckdb not installed - pip install duckdb)"]

    if not PICO_DB_PATH or not PICO_DB_PATH.exists():
        return ["(set PICO_DB_PATH environment variable)"]

    try:
        con = duckdb.connect(str(PICO_DB_PATH), read_only=True)
        results = con.execute("""
            SELECT DISTINCT edit_category
            FROM single_turn_edits
            ORDER BY edit_category
        """).fetchall()
        con.close()

        _category_cache = [r[0] for r in results]
        return _category_cache
    except Exception as e:
        print(f"[PicoPromptSampler] Error loading categories: {e}")
        return ["(error loading categories)"]


def get_category_stats() -> Dict[str, int]:
    """Get prompt counts per category."""
    if not DUCKDB_AVAILABLE or not PICO_DB_PATH or not PICO_DB_PATH.exists():
        return {}

    try:
        con = duckdb.connect(str(PICO_DB_PATH), read_only=True)
        results = con.execute("""
            SELECT edit_category, COUNT(*) as cnt
            FROM single_turn_edits
            GROUP BY edit_category
            ORDER BY cnt DESC
        """).fetchall()
        con.close()

        return {r[0]: r[1] for r in results}
    except Exception:
        return {}


def sample_prompt(category: str, seed: int = -1) -> Tuple[str, str, str]:
    """
    Sample a prompt from the given category.

    Returns: (full_prompt, summary, uuid)
    """
    if not DUCKDB_AVAILABLE or not PICO_DB_PATH or not PICO_DB_PATH.exists():
        return ("Database not available - set PICO_DB_PATH", "", "")

    try:
        con = duckdb.connect(str(PICO_DB_PATH), read_only=True)

        if seed >= 0:
            # Deterministic sampling using seed
            # Get count first
            count = con.execute("""
                SELECT COUNT(*) FROM single_turn_edits WHERE edit_category = ?
            """, [category]).fetchone()[0]

            if count == 0:
                con.close()
                return (f"No prompts found for category: {category}", "", "")

            # Use seed to pick index
            random.seed(seed)
            offset = random.randint(0, count - 1)

            result = con.execute("""
                SELECT full_edit_prompt, prompt_summary, uuid
                FROM single_turn_edits
                WHERE edit_category = ?
                LIMIT 1 OFFSET ?
            """, [category, offset]).fetchone()
        else:
            # Random sampling
            result = con.execute("""
                SELECT full_edit_prompt, prompt_summary, uuid
                FROM single_turn_edits
                WHERE edit_category = ?
                ORDER BY RANDOM()
                LIMIT 1
            """, [category]).fetchone()

        con.close()

        if result:
            return (result[0], result[1] or "", str(result[2]))
        return (f"No prompts found for category: {category}", "", "")

    except Exception as e:
        return (f"Error sampling prompt: {e}", "", "")


class PicoPromptSampler:
    """
    Sample prompts from Pico-Banana-400K dataset.

    Outputs prompts for use with Z-Image, Qwen-Image-Edit, or other encoders.
    Connect the 'prompt' output to your encoder's user_prompt input.

    Requires PICO_DB_PATH environment variable to be set.
    """

    CATEGORY = "ZImage/Pico"
    FUNCTION = "sample"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "summary", "uuid")
    OUTPUT_TOOLTIPS = (
        "Full edit prompt from Pico dataset",
        "Condensed summary of the prompt",
        "UUID for reference"
    )

    @classmethod
    def INPUT_TYPES(cls):
        categories = get_pico_categories()

        return {
            "required": {
                "category": (categories, {
                    "default": categories[0] if categories else "(none)",
                    "tooltip": "Pico edit category to sample from"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0x7FFFFFFF,
                    "tooltip": "-1 for random, or set seed for reproducible sampling"
                }),
            },
            "optional": {
                "trigger": ("*", {
                    "tooltip": "Connect anything here to trigger resample on each run"
                }),
            }
        }

    def sample(self, category: str, seed: int, trigger=None):
        """Sample a prompt from the selected category."""
        prompt, summary, uuid = sample_prompt(category, seed)

        # Log for debugging
        print(f"[PicoPromptSampler] Category: {category}")
        print(f"[PicoPromptSampler] Seed: {seed}")
        print(f"[PicoPromptSampler] UUID: {uuid}")
        print(f"[PicoPromptSampler] Prompt: {prompt[:100]}...")

        return (prompt, summary, uuid)


class PicoPromptBatch:
    """
    Sample multiple prompts from Pico for batch experiments.

    Returns a list of prompts that can be used with batch processing nodes.
    """

    CATEGORY = "ZImage/Pico"
    FUNCTION = "sample_batch"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts_json",)
    OUTPUT_IS_LIST = (False,)

    @classmethod
    def INPUT_TYPES(cls):
        categories = get_pico_categories()

        return {
            "required": {
                "category": (categories, {
                    "default": categories[0] if categories else "(none)",
                }),
                "count": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "tooltip": "Number of prompts to sample"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0x7FFFFFFF,
                }),
            }
        }

    def sample_batch(self, category: str, count: int, seed: int):
        """Sample multiple prompts and return as JSON."""
        import json

        if not DUCKDB_AVAILABLE or not PICO_DB_PATH or not PICO_DB_PATH.exists():
            return ('{"error": "Database not available - set PICO_DB_PATH"}',)

        try:
            con = duckdb.connect(str(PICO_DB_PATH), read_only=True)

            if seed >= 0:
                # Deterministic - get all then sample
                results = con.execute("""
                    SELECT full_edit_prompt, prompt_summary, uuid
                    FROM single_turn_edits
                    WHERE edit_category = ?
                """, [category]).fetchall()

                random.seed(seed)
                if len(results) > count:
                    results = random.sample(results, count)
            else:
                # Random
                results = con.execute("""
                    SELECT full_edit_prompt, prompt_summary, uuid
                    FROM single_turn_edits
                    WHERE edit_category = ?
                    ORDER BY RANDOM()
                    LIMIT ?
                """, [category, count]).fetchall()

            con.close()

            prompts = [
                {
                    "prompt": r[0],
                    "summary": r[1] or "",
                    "uuid": str(r[2]),
                    "category": category
                }
                for r in results
            ]

            return (json.dumps(prompts, indent=2),)

        except Exception as e:
            return (f'{{"error": "{str(e)}"}}',)


class PicoCategoryInfo:
    """
    Display information about Pico categories.

    Shows category statistics and helps choose which category to sample from.
    """

    CATEGORY = "ZImage/Pico"
    FUNCTION = "get_info"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "show_all": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show all categories (True) or just top 10 (False)"
                }),
            }
        }

    def get_info(self, show_all: bool):
        """Get category statistics."""
        stats = get_category_stats()

        if not stats:
            return ("Database not available - set PICO_DB_PATH environment variable",)

        lines = ["Pico Edit Categories:", "=" * 60]

        sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)

        if not show_all:
            sorted_stats = sorted_stats[:10]

        for cat, count in sorted_stats:
            lines.append(f"{count:>8}  {cat}")

        total = sum(stats.values())
        lines.append("=" * 60)
        lines.append(f"{total:>8}  TOTAL ({len(stats)} categories)")

        info = "\n".join(lines)
        return (info,)


# Export functions for API endpoint
def get_pico_categories_for_api() -> Dict[str, Any]:
    """Return categories with stats for API endpoint."""
    categories = get_pico_categories()
    stats = get_category_stats()

    return {
        "categories": categories,
        "stats": stats,
        "db_available": DUCKDB_AVAILABLE and bool(PICO_DB_PATH) and PICO_DB_PATH.exists(),
    }
