"""
Example usage of Wan2.1 ↔ ChronoEdit cache bridging.

This script demonstrates how to:
1. Load Wan2.1 and ChronoEdit models
2. Create a bridge with learned gating
3. Generate video with fused caches
4. Train the projector (optional)
"""

import torch
from config import WanChronoEditConfig
from bridge import WanChronoEditBridge


def example_basic_fusion():
    """
    Basic example: Fuse Wan and ChronoEdit caches without training.
    """
    print("="*60)
    print("Example 1: Basic Cache Fusion (No Training)")
    print("="*60)

    # Create configuration
    config = WanChronoEditConfig(
        projector_type="weighted",  # Simple weighted fusion
        fusion_alpha=0.5,  # Equal weight
        learnable_alpha=False,  # Fixed weight
        fusion_direction="bidirectional",
    )

    # NOTE: In practice, you would load actual Wan2.1 and ChronoEdit models
    # For this example, we create mock models
    print("\n[1] Creating mock models (replace with actual model loading)")
    wan_model = create_mock_wan_model()
    chronoedit_model = create_mock_chronoedit_model()

    # Create bridge
    print("[2] Creating Wan-ChronoEdit bridge")
    bridge = WanChronoEditBridge(
        config=config,
        wan_model=wan_model,
        chronoedit_model=chronoedit_model,
    )

    # Prepare inputs (mock data)
    print("[3] Preparing inputs")
    wan_inputs = {
        "video_latents": torch.randn(1, 16, 10, 32, 32),  # (B, C, T, H, W)
        "text_embeddings": torch.randn(1, 512, 4096),  # (B, L, D)
    }

    chronoedit_inputs = wan_inputs.copy()  # Same inputs for both models

    # Generate with fusion
    print("[4] Extracting and fusing caches")
    fused_cache, generation_inputs = bridge.forward(
        wan_inputs,
        chronoedit_inputs,
        target_model="wan"
    )

    print(f"    - Fused cache layers: {len(fused_cache)}")
    print(f"    - Cache shape: {fused_cache.shape}")

    print("\n[5] Ready for generation with fused cache!")
    print("    Use: wan_model.generate(**generation_inputs)")

    print("\n✓ Basic fusion complete!\n")


def example_learned_gating():
    """
    Advanced example: Use learned gating for context-dependent fusion.
    """
    print("="*60)
    print("Example 2: Learned Gating Projector")
    print("="*60)

    # Create configuration with learned gating
    config = WanChronoEditConfig(
        projector_type="gating",
        projector_hidden_dim=256,
        gate_granularity="head",  # Per-head gating
        fusion_direction="wan_to_chrono",  # Wan as source
    )

    print("\n[1] Creating bridge with learned gating")
    bridge = WanChronoEditBridge(config=config)

    print("[2] Projector architecture:")
    print(f"    - Type: {config.projector_type}")
    print(f"    - Hidden dim: {config.projector_hidden_dim}")
    print(f"    - Gate granularity: {config.gate_granularity}")
    print(f"    - Trainable parameters: {count_parameters(bridge.key_projector):,}")

    print("\n[3] Training projector (mock)")
    print("    In practice, train on paired video generation data:")
    print("    - Input: Text prompt + reference frames")
    print("    - Wan output: High visual quality")
    print("    - ChronoEdit output: High temporal consistency")
    print("    - Objective: Combine both strengths")

    print("\n✓ Learned gating bridge created!\n")


def example_training_pipeline():
    """
    Example training pipeline for projector.
    """
    print("="*60)
    print("Example 3: Training Pipeline")
    print("="*60)

    config = WanChronoEditConfig(
        projector_type="gating",
        learning_rate=1e-4,
        weight_decay=0.01,
    )

    bridge = WanChronoEditBridge(config=config)

    print("\n[1] Training setup:")
    print(f"    - Learning rate: {config.learning_rate}")
    print(f"    - Weight decay: {config.weight_decay}")
    print(f"    - Trainable: {config.trainable_components}")

    print("\n[2] Training procedure:")
    print("    a) Prepare dataset:")
    print("       - Collect video prompts with desired outputs")
    print("       - Run both Wan2.1 and ChronoEdit")
    print("       - Extract KV caches")
    print()
    print("    b) Training loop:")
    print("       - Extract caches from both models")
    print("       - Fuse with current projector")
    print("       - Compute loss (MSE or generation quality)")
    print("       - Update projector parameters")
    print()
    print("    c) Validation:")
    print("       - Test on held-out video prompts")
    print("       - Measure quality + consistency metrics")

    print("\n[3] Pseudo-code:")
    print("""
    for epoch in range(num_epochs):
        for batch in dataloader:
            wan_cache = extract_wan_cache(batch)
            chrono_cache = extract_chronoedit_cache(batch)

            fused_cache = bridge.fuse_caches(wan_cache, chrono_cache)

            loss = compute_loss(fused_cache, target_cache)
            loss.backward()
            optimizer.step()
    """)

    print("\n[4] Saving trained projector:")
    print("    bridge.save_projectors('wan_chrono_projector.pt')")

    print("\n✓ Training pipeline explained!\n")


def example_use_cases():
    """
    Demonstrate specific use cases for Wan-ChronoEdit bridging.
    """
    print("="*60)
    print("Example 4: Real-World Use Cases")
    print("="*60)

    print("\nUse Case 1: Quality + Consistency")
    print("-" * 40)
    print("Scenario: Generate sunset video with smooth color transitions")
    print("Approach:")
    print("  - Wan2.1: Generates beautiful, high-quality frames")
    print("  - ChronoEdit: Ensures smooth temporal transitions")
    print("  - Fused: Best of both worlds")
    print()
    print("Configuration:")
    print("  config = WanChronoEditConfig(")
    print("      projector_type='gating',")
    print("      fusion_direction='bidirectional',")
    print("  )")

    print("\n\nUse Case 2: Two-Stage Generation")
    print("-" * 40)
    print("Scenario: Fast prototyping → High-quality refinement")
    print("Approach:")
    print("  - Stage 1: Wan2.1 quick generation (low steps)")
    print("  - Stage 2: Bridge cache → ChronoEdit refinement")
    print("  - Result: Fast iteration with quality finish")
    print()
    print("Configuration:")
    print("  config = WanChronoEditConfig(")
    print("      projector_type='gating',")
    print("      fusion_direction='wan_to_chrono',  # Wan → ChronoEdit")
    print("  )")

    print("\n\nUse Case 3: Temporal Reasoning Injection")
    print("-" * 40)
    print("Scenario: Physics-aware video generation")
    print("Approach:")
    print("  - ChronoEdit: Provides temporal reasoning patterns")
    print("  - Wan2.1: Uses patterns for physics-plausible generation")
    print("  - Example: Ball trajectories, water flow, smoke")
    print()
    print("Configuration:")
    print("  config = WanChronoEditConfig(")
    print("      projector_type='gating',")
    print("      fusion_direction='chrono_to_wan',  # ChronoEdit → Wan")
    print("      gate_granularity='token',  # Fine-grained control")
    print("  )")

    print("\n\n✓ Use cases demonstrated!\n")


# Helper functions

def create_mock_wan_model():
    """Create mock Wan2.1 model for demonstration."""
    class MockWanModel(torch.nn.Module):
        def forward(self, **kwargs):
            # Mock forward pass
            pass

        def generate(self, **kwargs):
            # Mock generation
            return torch.randn(1, 16, 10, 32, 32)

    return MockWanModel()


def create_mock_chronoedit_model():
    """Create mock ChronoEdit model for demonstration."""
    return create_mock_wan_model()  # Same architecture


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Wan2.1 ↔ ChronoEdit Cache Bridging Examples")
    print("="*60 + "\n")

    # Run all examples
    example_basic_fusion()
    print("\n" + "="*60 + "\n")

    example_learned_gating()
    print("\n" + "="*60 + "\n")

    example_training_pipeline()
    print("\n" + "="*60 + "\n")

    example_use_cases()

    print("\n" + "="*60)
    print("All examples complete!")
    print("="*60 + "\n")
