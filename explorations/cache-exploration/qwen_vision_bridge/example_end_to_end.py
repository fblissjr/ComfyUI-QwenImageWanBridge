"""
End-to-End Example: Complete Vision Bridge Pipeline

This script demonstrates the full pipeline:
1. Extract vision caches from Qwen3 and Qwen2.5
2. Bridge caches with vision projector
3. Generate edited images with enhanced vision
4. Evaluate capabilities (OCR, spatial, detail)
5. Compare baseline vs enhanced

Run this to see the complete system in action.
"""

import torch
from PIL import Image
import os


def example_1_basic_comparison():
    """
    Example 1: Basic comparison of baseline vs enhanced.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Baseline vs Enhanced Comparison")
    print("=" * 60)

    from enhanced_pipeline import EnhancedQwenImageEdit
    from evaluation import CapabilityEvaluator

    # Initialize pipeline
    print("\nInitializing enhanced pipeline...")
    pipeline = EnhancedQwenImageEdit(
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit=True,  # Save memory
    )

    # Create synthetic test image with text
    print("\nCreating test image...")
    evaluator = CapabilityEvaluator(pipeline)
    test_image = evaluator._create_text_image("Hello World", lang="en")

    # Save test image
    os.makedirs("examples_output", exist_ok=True)
    test_image.save("examples_output/test_input.png")
    print("  Test image saved: examples_output/test_input.png")

    # Test prompt
    prompt = "Change the text to say 'Goodbye World'"

    # Generate comparison
    print(f"\nGenerating with prompt: '{prompt}'")
    baseline, enhanced, attention_maps = pipeline.compare(
        image=test_image,
        prompt=prompt,
        save_comparison="examples_output/comparison_basic.png"
    )

    print("\n✓ Example 1 complete!")
    print("  Outputs saved to: examples_output/")

    return pipeline, test_image


def example_2_ocr_capability():
    """
    Example 2: Test OCR capability with multi-lingual text.
    """
    print("\n" + "=" * 60)
    print("Example 2: OCR Capability Test")
    print("=" * 60)

    from enhanced_pipeline import EnhancedQwenImageEdit
    from evaluation import CapabilityEvaluator

    pipeline = EnhancedQwenImageEdit(
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit=True,
    )

    evaluator = CapabilityEvaluator(pipeline, output_dir="examples_output/ocr_tests")

    # Test OCR on different languages
    test_cases = [
        ("Hello", "en", "Change 'Hello' to 'Hi there'"),
        ("你好", "zh", "Change the Chinese text to '再见'"),
        ("Bonjour", "fr", "Remove all text"),
    ]

    results = []

    for text, lang, prompt in test_cases:
        print(f"\n--- Testing: {text} ({lang}) ---")
        print(f"    Prompt: {prompt}")

        # Create test image
        test_image = evaluator._create_text_image(text, lang)

        # Generate
        baseline = pipeline.generate(test_image, prompt, use_qwen3_vision=False)
        enhanced = pipeline.generate(test_image, prompt, use_qwen3_vision=True)

        results.append({
            'text': text,
            'lang': lang,
            'baseline': baseline,
            'enhanced': enhanced,
        })

        print(f"    ✓ Generated")

    print("\n✓ Example 2 complete!")
    print("  Expected: Enhanced performs better on text editing")

    return results


def example_3_spatial_reasoning():
    """
    Example 3: Test spatial reasoning capability.
    """
    print("\n" + "=" * 60)
    print("Example 3: Spatial Reasoning Test")
    print("=" * 60)

    from enhanced_pipeline import EnhancedQwenImageEdit
    from evaluation import CapabilityEvaluator

    pipeline = EnhancedQwenImageEdit(
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit=True,
    )

    evaluator = CapabilityEvaluator(pipeline, output_dir="examples_output/spatial_tests")

    # Test spatial manipulations
    print("\n--- Test: Move object left to right ---")
    spatial_image = evaluator._create_spatial_scene("left")
    prompt = "Move the red circle from left side to right side"

    baseline, enhanced, _ = pipeline.compare(
        image=spatial_image,
        prompt=prompt,
        save_comparison="examples_output/spatial_tests/move_left_right.png"
    )

    print("    ✓ Generated")
    print("\n✓ Example 3 complete!")
    print("  Expected: Enhanced better understands spatial instructions")


def example_4_detail_preservation():
    """
    Example 4: Test detail preservation.
    """
    print("\n" + "=" * 60)
    print("Example 4: Detail Preservation Test")
    print("=" * 60)

    from enhanced_pipeline import EnhancedQwenImageEdit
    from evaluation import CapabilityEvaluator

    pipeline = EnhancedQwenImageEdit(
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit=True,
    )

    evaluator = CapabilityEvaluator(pipeline, output_dir="examples_output/detail_tests")

    # Test detail preservation
    print("\n--- Test: Change color while preserving texture ---")
    textured_image = evaluator._create_textured_image()
    prompt = "Change the color to blue while keeping the checkerboard pattern"

    baseline, enhanced, _ = pipeline.compare(
        image=textured_image,
        prompt=prompt,
        save_comparison="examples_output/detail_tests/texture_preservation.png"
    )

    print("    ✓ Generated")
    print("\n✓ Example 4 complete!")
    print("  Expected: Enhanced preserves fine details better")


def example_5_full_evaluation():
    """
    Example 5: Run complete evaluation suite.
    """
    print("\n" + "=" * 60)
    print("Example 5: Full Evaluation Suite")
    print("=" * 60)

    from enhanced_pipeline import EnhancedQwenImageEdit
    from evaluation import CapabilityEvaluator

    pipeline = EnhancedQwenImageEdit(
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit=True,
    )

    evaluator = CapabilityEvaluator(
        pipeline,
        output_dir="examples_output/full_evaluation"
    )

    # Run all tests
    print("\nRunning complete evaluation...")
    results = evaluator.evaluate_all()

    print("\n✓ Example 5 complete!")
    print("  Full results saved to: examples_output/full_evaluation/")

    return results


def example_6_attention_visualization():
    """
    Example 6: Visualize attention weights from bridge.
    """
    print("\n" + "=" * 60)
    print("Example 6: Attention Visualization")
    print("=" * 60)

    from enhanced_pipeline import EnhancedQwenImageEdit
    from evaluation import CapabilityEvaluator
    import matplotlib.pyplot as plt
    import numpy as np

    pipeline = EnhancedQwenImageEdit(
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit=True,
    )

    evaluator = CapabilityEvaluator(pipeline)
    test_image = evaluator._create_text_image("ATTENTION", lang="en")

    prompt = "Change text color to red"

    print("\nGenerating with attention map extraction...")
    enhanced, attention_weights = pipeline.generate(
        image=test_image,
        prompt=prompt,
        use_qwen3_vision=True,
        return_attention_maps=True,
    )

    if attention_weights is not None:
        print("  Attention weights shape:", attention_weights.shape)

        # Visualize attention
        try:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(test_image)
            plt.title("Input Image")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            # Average attention across heads and tokens
            attn_map = attention_weights.mean(dim=0).mean(dim=0).cpu().numpy()
            plt.imshow(attn_map, cmap='hot')
            plt.title("Average Attention Weights")
            plt.colorbar()

            plt.tight_layout()
            plt.savefig("examples_output/attention_visualization.png", dpi=150)
            plt.close()

            print("  Attention visualization saved: examples_output/attention_visualization.png")
        except Exception as e:
            print(f"  (Visualization failed: {e})")

    else:
        print("  (Attention weights not available - may need bridge training)")

    print("\n✓ Example 6 complete!")


def main():
    """
    Run all examples.
    """
    print("\n" + "=" * 80)
    print(" Qwen Vision Bridge: End-to-End Examples")
    print("=" * 80)

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    if device == "cpu":
        print("WARNING: Running on CPU. This will be slow!")
        print("For best performance, use a CUDA-enabled GPU.")

    # Menu
    print("\nAvailable examples:")
    print("  1. Basic comparison (baseline vs enhanced)")
    print("  2. OCR capability test (multi-lingual)")
    print("  3. Spatial reasoning test")
    print("  4. Detail preservation test")
    print("  5. Full evaluation suite")
    print("  6. Attention visualization")
    print("  7. Run all examples")

    choice = input("\nSelect example (1-7): ").strip()

    try:
        if choice == "1":
            example_1_basic_comparison()
        elif choice == "2":
            example_2_ocr_capability()
        elif choice == "3":
            example_3_spatial_reasoning()
        elif choice == "4":
            example_4_detail_preservation()
        elif choice == "5":
            example_5_full_evaluation()
        elif choice == "6":
            example_6_attention_visualization()
        elif choice == "7":
            # Run all
            example_1_basic_comparison()
            example_2_ocr_capability()
            example_3_spatial_reasoning()
            example_4_detail_preservation()
            example_5_full_evaluation()
            example_6_attention_visualization()
        else:
            print("Invalid choice. Running example 1...")
            example_1_basic_comparison()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        print("\n" + "=" * 60)
        print("Common issues:")
        print("  1. VRAM: Reduce batch size or use 8-bit quantization")
        print("  2. Models not found: Check HuggingFace model IDs")
        print("  3. Missing dependencies: pip install -r requirements.txt")
        print("=" * 60)

    print("\n" + "=" * 80)
    print(" Examples Complete!")
    print("=" * 80)

    print("\nNext steps:")
    print("  • Review outputs in examples_output/")
    print("  • Compare baseline vs enhanced visually")
    print("  • Train bridge on task-specific data (optional)")
    print("  • Integrate with your own image editing workflows")


if __name__ == "__main__":
    main()
