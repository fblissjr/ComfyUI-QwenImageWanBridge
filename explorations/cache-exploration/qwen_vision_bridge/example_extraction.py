"""
Example: Extract and compare vision caches from Qwen3-VL and Qwen2.5-VL

This script demonstrates:
1. Loading both models
2. Extracting vision caches from the same image
3. Comparing the cache structures
4. Visualizing the differences

Run this to validate cache extraction before building the bridge.
"""

import torch
from PIL import Image
import requests
from io import BytesIO
from cache_extraction import VisionCacheExtractor, VisionCache


def load_example_image(url: str = None) -> Image.Image:
    """
    Load an example image for testing.

    Args:
        url: Optional URL to load image from

    Returns:
        PIL Image
    """
    if url is None:
        # Default: Use a simple test image
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

    print(f"Loading image from: {url}")
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    print(f"  Image size: {image.size}")
    return image


def example_basic_extraction():
    """
    Example 1: Basic cache extraction from both models.
    """
    print("\n" + "="*60)
    print("Example 1: Basic Cache Extraction")
    print("="*60 + "\n")

    # Initialize extractor
    extractor = VisionCacheExtractor(
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit=True,  # Use 8-bit for memory efficiency
    )

    # Load example image
    image = load_example_image()

    # Test prompt
    prompt = "Describe this image focusing on objects, colors, and spatial layout."

    # Extract from Qwen3-VL
    print("\n--- Extracting from Qwen3-VL ---")
    qwen3_cache = extractor.extract_qwen3_vision(
        image=image,
        text_prompt=prompt
    )

    # Extract from Qwen2.5-VL
    print("\n--- Extracting from Qwen2.5-VL ---")
    qwen25_cache = extractor.extract_qwen25_vision(
        image=image,
        text_prompt=prompt
    )

    # Compare caches
    print("\n--- Comparing Caches ---")
    comparison = extractor.compare_caches(qwen3_cache, qwen25_cache)

    return qwen3_cache, qwen25_cache, comparison


def example_deepstack_analysis(qwen3_cache: VisionCache):
    """
    Example 2: Analyze Qwen3's DeepStack multi-level features.
    """
    print("\n" + "="*60)
    print("Example 2: DeepStack Multi-Level Feature Analysis")
    print("="*60 + "\n")

    print("Qwen3-VL DeepStack Features:")
    print(f"  Early (low-level) features shape: {qwen3_cache.early_features.shape}")
    print(f"  Mid (mid-level) features shape: {qwen3_cache.mid_features.shape}")
    print(f"  Late (high-level) features shape: {qwen3_cache.late_features.shape}")

    # Compute feature statistics
    print("\nFeature Statistics:")

    # Early features (edges, textures)
    early_mean = qwen3_cache.early_features.mean().item()
    early_std = qwen3_cache.early_features.std().item()
    print(f"  Early features - Mean: {early_mean:.4f}, Std: {early_std:.4f}")

    # Mid features (objects, structure)
    mid_mean = qwen3_cache.mid_features.mean().item()
    mid_std = qwen3_cache.mid_features.std().item()
    print(f"  Mid features   - Mean: {mid_mean:.4f}, Std: {mid_std:.4f}")

    # Late features (semantics, context)
    late_mean = qwen3_cache.late_features.mean().item()
    late_std = qwen3_cache.late_features.std().item()
    print(f"  Late features  - Mean: {late_mean:.4f}, Std: {late_std:.4f}")

    # Compute cosine similarity between levels
    print("\nInter-Level Similarity (Cosine):")

    def cosine_sim(a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        return torch.nn.functional.cosine_similarity(
            a_flat.unsqueeze(0),
            b_flat.unsqueeze(0)
        ).item()

    early_mid_sim = cosine_sim(qwen3_cache.early_features, qwen3_cache.mid_features)
    mid_late_sim = cosine_sim(qwen3_cache.mid_features, qwen3_cache.late_features)
    early_late_sim = cosine_sim(qwen3_cache.early_features, qwen3_cache.late_features)

    print(f"  Early ↔ Mid:  {early_mid_sim:.4f}")
    print(f"  Mid ↔ Late:   {mid_late_sim:.4f}")
    print(f"  Early ↔ Late: {early_late_sim:.4f}")

    print("\nObservations:")
    if early_late_sim < 0.9:
        print("  ✓ Early and late features are distinct (good for multi-level fusion)")
    else:
        print("  ⚠ Early and late features are similar (may not benefit from multi-level)")


def example_dimension_analysis(qwen3_cache: VisionCache, qwen25_cache: VisionCache):
    """
    Example 3: Analyze dimension mismatches for bridge design.
    """
    print("\n" + "="*60)
    print("Example 3: Dimension Mismatch Analysis")
    print("="*60 + "\n")

    qwen3_dim = qwen3_cache.hidden_dim
    qwen25_dim = qwen25_cache.hidden_dim

    print(f"Dimension mismatch: {qwen3_dim} → {qwen25_dim}")
    print(f"Compression ratio: {qwen25_dim / qwen3_dim:.2f}x")

    if qwen3_dim > qwen25_dim:
        print(f"\nBridge needs to compress {qwen3_dim - qwen25_dim} dimensions")
        print("  Strategy: Use learned linear projection + fusion network")
    elif qwen3_dim < qwen25_dim:
        print(f"\nBridge needs to expand {qwen25_dim - qwen3_dim} dimensions")
        print("  Strategy: Use learned expansion + interpolation")
    else:
        print("\n✓ Dimensions match! Simple fusion possible.")

    # Estimate projector size
    print("\nProjector Network Size Estimate:")

    # For each DeepStack level
    num_levels = 3  # early, mid, late
    params_per_level = qwen3_dim * qwen25_dim

    print(f"  Per-level projection: {params_per_level:,} params")
    print(f"  Total for {num_levels} levels: {params_per_level * num_levels:,} params")

    # Fusion network (concatenated → fused)
    fusion_input_dim = qwen25_dim * num_levels
    fusion_hidden_dim = qwen25_dim * 2
    fusion_params = (
        fusion_input_dim * fusion_hidden_dim +  # First layer
        fusion_hidden_dim * qwen25_dim           # Output layer
    )

    print(f"  Fusion network: {fusion_params:,} params")

    total_params = params_per_level * num_levels + fusion_params
    print(f"\n  Total bridge parameters: {total_params:,} (~{total_params / 1e6:.1f}M)")

    if total_params < 10e6:
        print("  ✓ Bridge is lightweight (<10M params)")
    else:
        print("  ⚠ Bridge is heavy (>10M params) - consider dimensionality reduction")


def example_memory_usage():
    """
    Example 4: Estimate memory usage for cache extraction.
    """
    print("\n" + "="*60)
    print("Example 4: Memory Usage Estimation")
    print("="*60 + "\n")

    print("Estimated VRAM usage:")
    print("  Qwen3-VL 8B (8-bit): ~8-10 GB")
    print("  Qwen2.5-VL 7B (8-bit): ~7-9 GB")
    print("  Vision caches: ~0.5-1 GB")
    print("  Bridge projector: ~0.1 GB")
    print("\n  Total (sequential loading): ~10-15 GB")
    print("  Total (both loaded): ~15-20 GB")

    print("\nRecommendations:")
    print("  • 24GB VRAM (RTX 4090, A5000): Load both models")
    print("  • 16GB VRAM (RTX 4080): Use 8-bit, sequential loading")
    print("  • 12GB VRAM (RTX 4070 Ti): Use 8-bit + CPU offloading")


def main():
    """
    Run all examples.
    """
    print("\n" + "="*80)
    print(" Qwen Vision Cache Extraction Examples")
    print("="*80)

    try:
        # Example 1: Basic extraction
        qwen3_cache, qwen25_cache, comparison = example_basic_extraction()

        # Example 2: DeepStack analysis
        example_deepstack_analysis(qwen3_cache)

        # Example 3: Dimension analysis
        example_dimension_analysis(qwen3_cache, qwen25_cache)

        # Example 4: Memory usage
        example_memory_usage()

        print("\n" + "="*80)
        print(" All examples completed successfully!")
        print("="*80 + "\n")

        print("Next steps:")
        print("  1. Build VisionCacheBridge to project Qwen3 → Qwen2.5")
        print("  2. Integrate with Qwen-Image-Edit pipeline")
        print("  3. Test on OCR, spatial, and detail tasks")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
