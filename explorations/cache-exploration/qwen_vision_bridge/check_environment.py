#!/usr/bin/env python
"""
Environment Checker for Qwen Vision Bridge

Run this script on your CUDA machine to verify all dependencies
and hardware requirements are met before running the main examples.
"""

import sys

def check_python_version():
    """Check Python version (3.8+)"""
    print("=" * 60)
    print("Python Version Check")
    print("=" * 60)
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print("✓ Python version OK")
    return True


def check_dependencies():
    """Check required packages"""
    print("\n" + "=" * 60)
    print("Dependency Check")
    print("=" * 60)

    required = [
        ("torch", "2.0.0"),
        ("transformers", "4.40.0"),
        ("diffusers", "0.27.0"),
        ("accelerate", "0.27.0"),
        ("PIL", None),  # Pillow
        ("numpy", "1.24.0"),
    ]

    optional = [
        ("bitsandbytes", "0.41.0", "8-bit quantization"),
        ("xformers", "0.0.20", "memory-efficient attention"),
        ("matplotlib", "3.7.0", "visualization"),
        ("cv2", None, "opencv-python - image processing"),
        ("skimage", None, "scikit-image - SSIM metrics"),
    ]

    all_ok = True

    # Check required
    print("\nRequired packages:")
    for package, min_version in required:
        try:
            if package == "PIL":
                import PIL
                version = PIL.__version__
                package_name = "Pillow"
            else:
                module = __import__(package)
                version = getattr(module, "__version__", "unknown")
                package_name = package

            print(f"  ✓ {package_name:20} {version}")

            if min_version and version != "unknown":
                from packaging import version as pkg_version
                if pkg_version.parse(version) < pkg_version.parse(min_version):
                    print(f"    ⚠️  Version {min_version}+ recommended")
        except ImportError:
            print(f"  ❌ {package:20} NOT FOUND")
            all_ok = False

    # Check optional
    print("\nOptional packages:")
    for item in optional:
        package = item[0]
        min_version = item[1]
        description = item[2] if len(item) > 2 else ""

        try:
            if package == "cv2":
                import cv2
                version = cv2.__version__
            elif package == "skimage":
                import skimage
                version = skimage.__version__
            else:
                module = __import__(package)
                version = getattr(module, "__version__", "unknown")

            print(f"  ✓ {package:20} {version:10} - {description}")
        except ImportError:
            print(f"  ⚠️  {package:20} not found   - {description}")

    return all_ok


def check_cuda():
    """Check CUDA availability and VRAM"""
    print("\n" + "=" * 60)
    print("CUDA Check")
    print("=" * 60)

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            device_count = torch.cuda.device_count()
            print(f"GPU count: {device_count}")

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                total_memory_gb = props.total_memory / (1024**3)
                print(f"\nGPU {i}: {props.name}")
                print(f"  Total VRAM: {total_memory_gb:.1f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")

                # Memory recommendations
                if total_memory_gb >= 20:
                    print("  ✓ Excellent - Can run both models simultaneously")
                elif total_memory_gb >= 14:
                    print("  ✓ Good - Use 8-bit quantization, sequential loading")
                elif total_memory_gb >= 10:
                    print("  ⚠️  Minimum - Use 8-bit + CPU offloading (slower)")
                else:
                    print("  ❌ Insufficient - Need 12GB+ for this project")

            return True
        else:
            print("⚠️  CUDA not available - will run on CPU (very slow)")
            print("   For best results, run on a machine with CUDA GPU")
            return False

    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False


def check_model_access():
    """Check if HuggingFace models are accessible"""
    print("\n" + "=" * 60)
    print("Model Access Check")
    print("=" * 60)

    print("\nThis will attempt to load model configs (not weights)...")

    models = [
        ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL (7B)"),
        ("Qwen/Qwen3-VL-8B-Instruct", "Qwen3-VL (9B)"),
    ]

    try:
        from transformers import AutoConfig

        all_ok = True
        for model_id, name in models:
            try:
                print(f"\n  Testing {name}...")
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                print(f"  ✓ {name} accessible")
            except Exception as e:
                print(f"  ❌ {name} not accessible: {e}")
                all_ok = False

        if not all_ok:
            print("\n⚠️  Some models not accessible. You may need to:")
            print("   1. Accept model licenses on HuggingFace")
            print("   2. Log in: huggingface-cli login")
            print("   3. Check internet connection")

        return all_ok

    except ImportError:
        print("❌ transformers not installed - skipping model check")
        return False


def check_disk_space():
    """Check available disk space for models"""
    print("\n" + "=" * 60)
    print("Disk Space Check")
    print("=" * 60)

    try:
        import shutil
        import os

        # Check cache directory
        cache_dir = os.path.expanduser("~/.cache/huggingface")

        if os.path.exists(cache_dir):
            stat = shutil.disk_usage(cache_dir)
            free_gb = stat.free / (1024**3)
            print(f"HuggingFace cache: {cache_dir}")
            print(f"Free space: {free_gb:.1f} GB")

            if free_gb >= 50:
                print("✓ Sufficient space for model downloads")
            elif free_gb >= 30:
                print("⚠️  Space is limited - monitor during downloads")
            else:
                print("❌ Insufficient space - need ~40GB for models")

            return free_gb >= 30
        else:
            print(f"⚠️  Cache directory doesn't exist yet: {cache_dir}")
            print("   Will be created on first model download")
            return True

    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True


def print_recommendations():
    """Print setup recommendations"""
    print("\n" + "=" * 60)
    print("Setup Recommendations")
    print("=" * 60)

    print("""
For optimal performance:

1. Hardware:
   - Recommended: RTX 4090 (24GB), A5000 (24GB), A6000
   - Good: RTX 4080 (16GB) with 8-bit quantization
   - Minimum: RTX 4070 Ti (12GB) with 8-bit + offloading

2. Software:
   - Install all optional packages for full functionality
   - Use CUDA 11.8+ or 12.1+ for best compatibility
   - Consider installing xformers for memory efficiency

3. Models:
   - Ensure HuggingFace account is set up
   - Log in: huggingface-cli login
   - Accept model licenses on HuggingFace website

4. Running:
   - Start with example_extraction.py (lighter)
   - Then try example_end_to_end.py (full pipeline)
   - Use 8-bit quantization if VRAM is limited

5. Troubleshooting:
   - OOM errors: Reduce batch size, enable 8-bit
   - Slow: Check CUDA is actually being used
   - Model download fails: Check huggingface-cli login
""")


def main():
    print("\n" + "=" * 80)
    print(" Qwen Vision Bridge - Environment Checker")
    print("=" * 80)

    results = {
        "Python version": check_python_version(),
        "Dependencies": check_dependencies(),
        "CUDA": check_cuda(),
        "Model access": check_model_access(),
        "Disk space": check_disk_space(),
    }

    print("\n" + "=" * 80)
    print(" Summary")
    print("=" * 80)

    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {check:20} {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All checks passed! Ready to run examples.")
        print("\nNext steps:")
        print("  1. cd qwen_vision_bridge")
        print("  2. python example_extraction.py")
        print("  3. python example_end_to_end.py")
    else:
        print("⚠️  Some checks failed. See recommendations below.")
        print_recommendations()

    print("=" * 80)


if __name__ == "__main__":
    main()
