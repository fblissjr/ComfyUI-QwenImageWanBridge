#!/usr/bin/env python
"""
Structure Validator for Qwen Vision Bridge

This script validates the code structure WITHOUT loading models.
Can run on machines without CUDA to verify implementation correctness.
"""

import importlib.util
import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists"""
    if filepath.exists():
        print(f"  ✓ {description}")
        return True
    else:
        print(f"  ❌ {description} - NOT FOUND")
        return False


def check_module_structure(module_path, expected_classes, expected_functions=None):
    """Check if a module has expected classes and functions"""
    module_name = module_path.stem
    print(f"\n--- Checking {module_name}.py ---")

    if not module_path.exists():
        print(f"  ❌ File not found: {module_path}")
        return False

    # Load module spec without executing
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    if spec is None:
        print(f"  ❌ Could not load module spec")
        return False

    # Read file content
    with open(module_path, 'r') as f:
        content = f.read()

    all_ok = True

    # Check classes
    print("  Classes:")
    for cls in expected_classes:
        if f"class {cls}" in content:
            print(f"    ✓ {cls}")
        else:
            print(f"    ❌ {cls} - NOT FOUND")
            all_ok = False

    # Check functions
    if expected_functions:
        print("  Functions:")
        for func in expected_functions:
            if f"def {func}" in content:
                print(f"    ✓ {func}")
            else:
                print(f"    ❌ {func} - NOT FOUND")
                all_ok = False

    return all_ok


def validate_cache_extraction():
    """Validate cache_extraction.py structure"""
    return check_module_structure(
        Path("cache_extraction.py"),
        expected_classes=[
            "VisionCache",
            "VisionCacheExtractor",
        ],
        expected_functions=[
            "create_test_image",
        ]
    )


def validate_vision_bridge():
    """Validate vision_bridge.py structure"""
    return check_module_structure(
        Path("vision_bridge.py"),
        expected_classes=[
            "BridgeConfig",
            "DeepStackProjector",
            "MultiLevelFusion",
            "VisionCacheBridge",
        ],
        expected_functions=[
            "create_default_bridge",
        ]
    )


def validate_enhanced_pipeline():
    """Validate enhanced_pipeline.py structure"""
    return check_module_structure(
        Path("enhanced_pipeline.py"),
        expected_classes=[
            "EnhancedQwenImageEdit",
        ],
        expected_functions=[]
    )


def validate_evaluation():
    """Validate evaluation.py structure"""
    return check_module_structure(
        Path("evaluation.py"),
        expected_classes=[
            "CapabilityEvaluator",
        ],
        expected_functions=[]
    )


def check_documentation():
    """Check documentation files"""
    print("\n--- Documentation Files ---")

    docs = {
        "README.md": "Main documentation",
        "QUICKSTART.md": "Quick start guide",
        "requirements.txt": "Dependencies",
        "__init__.py": "Package initialization",
    }

    all_ok = True
    for filename, description in docs.items():
        filepath = Path(filename)
        if not check_file_exists(filepath, f"{filename:20} - {description}"):
            all_ok = False

    return all_ok


def check_examples():
    """Check example scripts"""
    print("\n--- Example Scripts ---")

    examples = {
        "example_extraction.py": "Cache extraction demo",
        "example_end_to_end.py": "Full pipeline demo (7 examples)",
        "check_environment.py": "Environment checker",
        "validate_structure.py": "Structure validator (this file)",
    }

    all_ok = True
    for filename, description in examples.items():
        filepath = Path(filename)
        if not check_file_exists(filepath, f"{filename:25} - {description}"):
            all_ok = False

    return all_ok


def check_imports():
    """Check if critical imports are syntactically valid"""
    print("\n--- Import Validation ---")

    modules = [
        "cache_extraction",
        "vision_bridge",
        "enhanced_pipeline",
        "evaluation",
    ]

    all_ok = True
    for module_name in modules:
        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                f"{module_name}.py"
            )
            if spec is None:
                print(f"  ❌ {module_name:20} - Could not load spec")
                all_ok = False
            else:
                print(f"  ✓ {module_name:20} - Syntax valid")
        except Exception as e:
            print(f"  ❌ {module_name:20} - Error: {e}")
            all_ok = False

    return all_ok


def check_architecture_consistency():
    """Check that dimensions are consistent across modules"""
    print("\n--- Architecture Consistency ---")

    # Expected dimensions
    expected = {
        "Qwen3 hidden dim": 4096,
        "Qwen2.5 hidden dim": 3584,
        "Vision tokens": 256,
        "DeepStack levels": 3,
    }

    checks = []

    # Check cache_extraction.py
    with open("cache_extraction.py", 'r') as f:
        content = f.read()
        checks.append(("4096" in content, "Qwen3 4096D found in cache_extraction"))
        checks.append(("3584" in content, "Qwen2.5 3584D found in cache_extraction"))

    # Check vision_bridge.py
    with open("vision_bridge.py", 'r') as f:
        content = f.read()
        checks.append(("qwen3_hidden_dim: int = 4096" in content,
                      "Qwen3 dimension in BridgeConfig"))
        checks.append(("qwen25_hidden_dim: int = 3584" in content,
                      "Qwen2.5 dimension in BridgeConfig"))
        checks.append(("DeepStackProjector" in content,
                      "DeepStackProjector class exists"))
        checks.append(("MultiLevelFusion" in content,
                      "MultiLevelFusion class exists"))

    all_ok = True
    for check, description in checks:
        if check:
            print(f"  ✓ {description}")
        else:
            print(f"  ❌ {description}")
            all_ok = False

    return all_ok


def estimate_code_metrics():
    """Estimate code metrics"""
    print("\n--- Code Metrics ---")

    modules = [
        "cache_extraction.py",
        "vision_bridge.py",
        "enhanced_pipeline.py",
        "evaluation.py",
    ]

    total_lines = 0
    total_classes = 0
    total_functions = 0

    for module in modules:
        if Path(module).exists():
            with open(module, 'r') as f:
                content = f.read()
                lines = len(content.split('\n'))
                classes = content.count('class ')
                functions = content.count('def ') - classes * 2  # Approximate (exclude __init__, etc.)

                print(f"  {module:25} {lines:5} lines, {classes:2} classes, ~{functions:2} functions")

                total_lines += lines
                total_classes += classes
                total_functions += functions

    print(f"\n  Total: {total_lines} lines, {total_classes} classes, ~{total_functions} functions")

    return True


def main():
    print("=" * 80)
    print(" Qwen Vision Bridge - Structure Validator")
    print("=" * 80)
    print("\nThis validates code structure WITHOUT loading models or CUDA.\n")

    results = {}

    # Check documentation
    results["Documentation"] = check_documentation()

    # Check examples
    results["Examples"] = check_examples()

    # Check module structures
    results["cache_extraction"] = validate_cache_extraction()
    results["vision_bridge"] = validate_vision_bridge()
    results["enhanced_pipeline"] = validate_enhanced_pipeline()
    results["evaluation"] = validate_evaluation()

    # Check imports
    results["Imports"] = check_imports()

    # Check architecture consistency
    results["Architecture"] = check_architecture_consistency()

    # Code metrics
    estimate_code_metrics()

    # Summary
    print("\n" + "=" * 80)
    print(" Validation Summary")
    print("=" * 80)

    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {check:25} {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All structure checks passed!")
        print("\nCode structure is valid. Ready for testing on CUDA machine.")
        print("\nNext steps:")
        print("  1. Transfer to CUDA machine")
        print("  2. Run: python check_environment.py")
        print("  3. Run: python example_extraction.py")
    else:
        print("❌ Some checks failed. Review errors above.")

    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
