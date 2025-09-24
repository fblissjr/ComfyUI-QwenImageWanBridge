"""
Auto-apply debug patches when ComfyUI starts

Place this file in ComfyUI/custom_nodes/ or import it from your __init__.py
"""

def setup_debug_tracing():
    """Apply debug patches automatically"""
    import os
    import sys
    
    # Add the bridge directory to path
    bridge_path = "/Users/fredbliss/workspace/ComfyUI-QwenImageWanBridge"
    if bridge_path not in sys.path:
        sys.path.append(bridge_path)
    
    try:
        import debug_patch
        debug_patch.apply_debug_patches()
        print("✅ Debug patches applied - end-to-end tracing active")
    except Exception as e:
        print(f"❌ Failed to apply debug patches: {e}")

# Apply patches when this module is imported
setup_debug_tracing()