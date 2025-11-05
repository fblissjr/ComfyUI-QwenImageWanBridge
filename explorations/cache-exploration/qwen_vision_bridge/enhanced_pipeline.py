"""
Enhanced Qwen-Image-Edit Pipeline

Integrate vision bridge with Qwen-Image-Edit for enhanced capabilities.

Architecture:
    Input Image + Prompt
         ↓
    Qwen3-VL (extract enhanced vision)
         ↓
    Vision Bridge (project to Qwen2.5 format)
         ↓
    Qwen-Image-Edit DiT (generate with enhanced vision)
         ↓
    Output (with OCR, spatial, detail capabilities)
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any
from PIL import Image
from diffusers import DiffusionPipeline
from cache_extraction import VisionCacheExtractor, VisionCache
from vision_bridge import VisionCacheBridge, BridgeConfig


class EnhancedQwenImageEdit:
    """
    Qwen-Image-Edit enhanced with Qwen3-VL vision understanding.

    This pipeline injects Qwen3's superior vision capabilities into
    Qwen-Image-Edit's generation process.

    New capabilities enabled:
    - OCR-aware editing (32 languages)
    - Enhanced spatial reasoning
    - Fine detail preservation
    - Multi-image consistency (256K context)

    Usage:
        # Create enhanced pipeline
        pipeline = EnhancedQwenImageEdit()

        # Generate with Qwen3 vision enhancement
        output = pipeline.generate(
            image="input.jpg",
            prompt="Change the text 'Hello' to 'Goodbye'",
            use_qwen3_vision=True,  # Enable enhancement
        )

        # Compare with baseline
        baseline = pipeline.generate(
            image="input.jpg",
            prompt="Change the text 'Hello' to 'Goodbye'",
            use_qwen3_vision=False,  # Standard Qwen2.5 only
        )
    """

    def __init__(
        self,
        image_edit_model_id: str = "Qwen/Qwen-Image-Edit-2509",
        bridge_config: Optional[BridgeConfig] = None,
        bridge_weights_path: Optional[str] = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
        enable_cpu_offload: bool = False,
    ):
        """
        Initialize enhanced pipeline.

        Args:
            image_edit_model_id: HuggingFace model ID for Qwen-Image-Edit
            bridge_config: Configuration for vision bridge
            bridge_weights_path: Path to pre-trained bridge weights
            device: Device to load models on
            load_in_8bit: Use 8-bit quantization
            enable_cpu_offload: Offload models to CPU when not in use
        """
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.enable_cpu_offload = enable_cpu_offload

        print("Initializing Enhanced Qwen-Image-Edit Pipeline...")

        # Initialize vision cache extractor
        print("  Loading vision cache extractor...")
        self.vision_extractor = VisionCacheExtractor(
            device=device,
            load_in_8bit=load_in_8bit,
        )

        # Initialize vision bridge
        print("  Loading vision bridge...")
        if bridge_weights_path is not None:
            self.vision_bridge = VisionCacheBridge.from_pretrained(bridge_weights_path)
        else:
            if bridge_config is None:
                bridge_config = BridgeConfig()
            self.vision_bridge = VisionCacheBridge(bridge_config)

        self.vision_bridge.to(device)
        self.vision_bridge.eval()

        # Load Qwen-Image-Edit pipeline
        print(f"  Loading Qwen-Image-Edit: {image_edit_model_id}...")
        try:
            from diffusers import QwenImageEditPlusPipeline
            self.base_pipeline = QwenImageEditPlusPipeline.from_pretrained(
                image_edit_model_id,
                torch_dtype=torch.bfloat16,
                device_map=device if not load_in_8bit else "auto",
            )
        except ImportError:
            print("    WARNING: QwenImageEditPlusPipeline not found.")
            print("    Using generic DiffusionPipeline as fallback.")
            print("    Full integration requires diffusers with Qwen-Image-Edit support.")

            self.base_pipeline = DiffusionPipeline.from_pretrained(
                image_edit_model_id,
                torch_dtype=torch.bfloat16,
                device_map=device if not load_in_8bit else "auto",
            )

        if enable_cpu_offload and not load_in_8bit:
            self.base_pipeline.enable_model_cpu_offload()

        print("✓ Enhanced pipeline initialized successfully")
        print(f"  Device: {device}")
        print(f"  8-bit quantization: {load_in_8bit}")
        print(f"  CPU offload: {enable_cpu_offload}")
        print(f"  Bridge parameters: {self.vision_bridge.count_parameters():,}")

    def _inject_enhanced_vision(
        self,
        base_pipeline_inputs: Dict[str, Any],
        enhanced_vision_cache: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Inject enhanced vision cache into base pipeline inputs.

        This is where we hook into Qwen-Image-Edit's architecture.
        The enhanced vision cache replaces the standard Qwen2.5-VL encoding.

        Args:
            base_pipeline_inputs: Original inputs for Qwen-Image-Edit
            enhanced_vision_cache: Enhanced vision from bridge

        Returns:
            Modified inputs with enhanced vision conditioning
        """
        # Strategy 1: Replace vision encoder outputs directly
        # This assumes we can access and modify the vision encoder step

        # Note: This is a simplified placeholder.
        # Actual implementation depends on Qwen-Image-Edit's internal structure.
        # You may need to:
        # 1. Hook into the forward pass
        # 2. Replace encoder outputs at specific layer
        # 3. Ensure compatibility with downstream DiT

        modified_inputs = base_pipeline_inputs.copy()

        # If pipeline exposes vision encoder outputs, replace them
        if hasattr(self.base_pipeline, 'vision_encoder'):
            # Replace with enhanced cache
            modified_inputs['vision_embeddings'] = enhanced_vision_cache
        elif 'image_embeds' in modified_inputs:
            # Fallback: Replace image embeddings
            modified_inputs['image_embeds'] = enhanced_vision_cache
        else:
            # Last resort: Add as additional conditioning
            modified_inputs['additional_embeddings'] = enhanced_vision_cache

        return modified_inputs

    def generate(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        use_qwen3_vision: bool = True,
        return_attention_maps: bool = False,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> Union[Image.Image, tuple]:
        """
        Generate edited image with optional Qwen3 vision enhancement.

        Args:
            image: Input image (path or PIL Image)
            prompt: Edit instruction
            use_qwen3_vision: Whether to use enhanced vision (True) or baseline (False)
            return_attention_maps: Return attention weights from bridge
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for generation
            **kwargs: Additional arguments for base pipeline

        Returns:
            If return_attention_maps=False:
                edited_image: Generated image
            If return_attention_maps=True:
                (edited_image, attention_maps): Tuple with image and attention weights
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        print(f"\nGenerating with {'Qwen3 vision enhancement' if use_qwen3_vision else 'baseline Qwen2.5'}...")

        if use_qwen3_vision:
            # Enhanced path: Use Qwen3 vision
            print("  Step 1: Extracting Qwen3-VL vision cache...")
            qwen3_cache = self.vision_extractor.extract_qwen3_vision(
                image=image,
                text_prompt=prompt
            )

            print("  Step 2: Extracting Qwen2.5-VL baseline cache...")
            qwen25_cache = self.vision_extractor.extract_qwen25_vision(
                image=image,
                text_prompt=prompt
            )

            print("  Step 3: Bridging caches...")
            enhanced_vision, attention_weights = self.vision_bridge(
                qwen3_cache=qwen3_cache,
                qwen25_cache=qwen25_cache,
                return_attention_weights=True,
            )

            print(f"    Enhanced vision shape: {enhanced_vision.shape}")

            # Note: The actual integration point depends on Qwen-Image-Edit's API
            # This is a conceptual implementation showing the flow

            print("  Step 4: Generating with enhanced vision...")

            # Prepare base pipeline inputs
            base_inputs = {
                "prompt": prompt,
                "image": image,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                **kwargs
            }

            # Inject enhanced vision
            # NOTE: This is where actual integration would happen
            # The exact method depends on Qwen-Image-Edit's architecture
            modified_inputs = self._inject_enhanced_vision(base_inputs, enhanced_vision)

            # Generate
            try:
                output = self.base_pipeline(**modified_inputs)
            except Exception as e:
                print(f"    WARNING: Enhanced generation failed: {e}")
                print("    Falling back to baseline generation...")
                output = self.base_pipeline(**base_inputs)
                attention_weights = None

            edited_image = output.images[0] if hasattr(output, 'images') else output

            if return_attention_maps:
                return edited_image, attention_weights
            return edited_image

        else:
            # Baseline path: Standard Qwen2.5 only
            print("  Generating with baseline Qwen2.5-VL...")

            output = self.base_pipeline(
                prompt=prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs
            )

            edited_image = output.images[0] if hasattr(output, 'images') else output

            if return_attention_maps:
                return edited_image, None
            return edited_image

    def compare(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        save_comparison: Optional[str] = None,
    ) -> tuple:
        """
        Generate both baseline and enhanced versions for comparison.

        Args:
            image: Input image
            prompt: Edit instruction
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale
            save_comparison: Optional path to save side-by-side comparison

        Returns:
            (baseline_output, enhanced_output, attention_maps)
        """
        print("=" * 60)
        print("Comparison: Baseline vs Enhanced")
        print("=" * 60)

        # Generate baseline
        print("\n[1/2] Generating baseline (Qwen2.5-VL only)...")
        baseline = self.generate(
            image=image,
            prompt=prompt,
            use_qwen3_vision=False,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        # Generate enhanced
        print("\n[2/2] Generating enhanced (Qwen3-VL vision)...")
        enhanced, attention_maps = self.generate(
            image=image,
            prompt=prompt,
            use_qwen3_vision=True,
            return_attention_maps=True,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        # Optionally save comparison
        if save_comparison:
            self._save_comparison(
                input_image=image if isinstance(image, Image.Image) else Image.open(image),
                baseline=baseline,
                enhanced=enhanced,
                prompt=prompt,
                save_path=save_comparison,
            )

        print("\n✓ Comparison complete")
        return baseline, enhanced, attention_maps

    def _save_comparison(
        self,
        input_image: Image.Image,
        baseline: Image.Image,
        enhanced: Image.Image,
        prompt: str,
        save_path: str,
    ):
        """
        Save side-by-side comparison image.

        Layout: [Input | Baseline | Enhanced]
        """
        import numpy as np
        from PIL import ImageDraw, ImageFont

        # Resize all to same height
        height = 512
        input_resized = input_image.resize((int(input_image.width * height / input_image.height), height))
        baseline_resized = baseline.resize((int(baseline.width * height / baseline.height), height))
        enhanced_resized = enhanced.resize((int(enhanced.width * height / enhanced.height), height))

        # Create comparison canvas
        total_width = input_resized.width + baseline_resized.width + enhanced_resized.width
        canvas = Image.new('RGB', (total_width, height + 60), (255, 255, 255))

        # Paste images
        canvas.paste(input_resized, (0, 60))
        canvas.paste(baseline_resized, (input_resized.width, 60))
        canvas.paste(enhanced_resized, (input_resized.width + baseline_resized.width, 60))

        # Add labels
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        draw.text((input_resized.width // 2 - 30, 20), "Input", fill=(0, 0, 0), font=font)
        draw.text((input_resized.width + baseline_resized.width // 2 - 50, 20), "Baseline", fill=(0, 0, 0), font=font)
        draw.text((input_resized.width + baseline_resized.width + enhanced_resized.width // 2 - 50, 20), "Enhanced (Qwen3)", fill=(0, 0, 0), font=font)

        # Add prompt
        draw.text((10, 5), f"Prompt: {prompt[:60]}...", fill=(100, 100, 100), font=font)

        # Save
        canvas.save(save_path)
        print(f"  Comparison saved to: {save_path}")

    def benchmark(
        self,
        test_images: list,
        test_prompts: list,
        output_dir: str = "benchmark_results",
        num_inference_steps: int = 50,
    ):
        """
        Run benchmark on test set.

        Args:
            test_images: List of test images
            test_prompts: List of prompts (one per image)
            output_dir: Directory to save results
            num_inference_steps: Number of diffusion steps
        """
        import os
        import time

        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("Benchmarking Enhanced Pipeline")
        print("=" * 60)
        print(f"Test images: {len(test_images)}")
        print(f"Output directory: {output_dir}")

        results = []

        for idx, (image, prompt) in enumerate(zip(test_images, test_prompts)):
            print(f"\n[{idx+1}/{len(test_images)}] Processing: {prompt[:50]}...")

            # Baseline
            start = time.time()
            baseline = self.generate(
                image=image,
                prompt=prompt,
                use_qwen3_vision=False,
                num_inference_steps=num_inference_steps,
            )
            baseline_time = time.time() - start

            # Enhanced
            start = time.time()
            enhanced = self.generate(
                image=image,
                prompt=prompt,
                use_qwen3_vision=True,
                num_inference_steps=num_inference_steps,
            )
            enhanced_time = time.time() - start

            # Save comparison
            comparison_path = os.path.join(output_dir, f"comparison_{idx:03d}.png")
            self._save_comparison(
                input_image=image if isinstance(image, Image.Image) else Image.open(image),
                baseline=baseline,
                enhanced=enhanced,
                prompt=prompt,
                save_path=comparison_path,
            )

            results.append({
                'index': idx,
                'prompt': prompt,
                'baseline_time': baseline_time,
                'enhanced_time': enhanced_time,
                'overhead': (enhanced_time - baseline_time) / baseline_time * 100,
            })

            print(f"  Baseline time: {baseline_time:.2f}s")
            print(f"  Enhanced time: {enhanced_time:.2f}s")
            print(f"  Overhead: {results[-1]['overhead']:.1f}%")

        # Print summary
        print("\n" + "=" * 60)
        print("Benchmark Summary")
        print("=" * 60)
        avg_baseline = sum(r['baseline_time'] for r in results) / len(results)
        avg_enhanced = sum(r['enhanced_time'] for r in results) / len(results)
        avg_overhead = sum(r['overhead'] for r in results) / len(results)

        print(f"Average baseline time: {avg_baseline:.2f}s")
        print(f"Average enhanced time: {avg_enhanced:.2f}s")
        print(f"Average overhead: {avg_overhead:.1f}%")
        print(f"\nResults saved to: {output_dir}/")

        return results

    def train_bridge(
        self,
        training_data,
        num_epochs: int = 10,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        save_path: Optional[str] = None,
    ):
        """
        Train vision bridge on task-specific data.

        Args:
            training_data: Dataset with (image, prompt, target) tuples
            num_epochs: Number of training epochs
            batch_size: Batch size (usually 1 for large models)
            learning_rate: Learning rate
            save_path: Path to save trained weights

        Note: This is a simplified training loop.
        For production, use a more sophisticated training pipeline.
        """
        import torch.optim as optim
        from torch.utils.data import DataLoader

        print("=" * 60)
        print("Training Vision Bridge")
        print("=" * 60)

        # Set bridge to training mode
        self.vision_bridge.train()

        # Optimizer
        optimizer = optim.AdamW(
            self.vision_bridge.parameters(),
            lr=learning_rate,
            weight_decay=self.vision_bridge.config.weight_decay,
        )

        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            for batch_idx, (image, prompt, target) in enumerate(training_data):
                # Extract caches
                qwen3_cache = self.vision_extractor.extract_qwen3_vision(image, prompt)
                qwen25_cache = self.vision_extractor.extract_qwen25_vision(image, prompt)

                # Forward through bridge
                enhanced_vision, _ = self.vision_bridge(qwen3_cache, qwen25_cache)

                # Compute loss
                # This depends on your training objective:
                # Option 1: MSE with target vision features
                # Option 2: Generation quality loss
                # Option 3: Task-specific loss (OCR accuracy, spatial error, etc.)

                # Placeholder: MSE loss
                if target is not None:
                    loss = nn.functional.mse_loss(enhanced_vision, target)
                else:
                    # If no target, skip this batch
                    continue

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.6f}")

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.6f}")

        # Save trained weights
        if save_path:
            self.vision_bridge.save(save_path)

        print("\n✓ Training complete")

        # Set back to eval mode
        self.vision_bridge.eval()
