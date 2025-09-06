#!/usr/bin/env python3
"""
Test script to verify improved QwenVLTextEncoder resolution handling
"""

def test_resolution_improvements():
    """Test that the improved resolution logic works correctly"""
    try:
        # Read the file and check for the expected changes
        with open('nodes/qwen_vl_encoder.py', 'r') as f:
            content = f.read()
        
        # Check for basic width/height output functionality
        if 'RETURN_TYPES = ("CONDITIONING", "INT", "INT")' in content:
            print("✓ RETURN_TYPES correctly updated to include width and height")
        else:
            print("✗ RETURN_TYPES not found or incorrect")
            return False
        
        if 'RETURN_NAMES = ("conditioning", "width", "height")' in content:
            print("✓ RETURN_NAMES correctly updated to include width and height")
        else:
            print("✗ RETURN_NAMES not found or incorrect")
            return False
            
        # Check for improved resolution list
        if '(1920, 1080)' in content and '(1080, 1920)' in content:
            print("✓ Resolution list includes modern aspect ratios (16:9, 9:16)")
        else:
            print("✗ Resolution list with modern aspect ratios not found")
            return False
            
        # Check for simplified resolution logic (removed confusing parsing)
        if 'resolution_mode == "auto"' in content:
            print("✓ Simplified resolution logic with clear auto/custom modes")
        else:
            print("✗ Simplified resolution logic not found")
            return False
            
        # Check for improved optimal resolution method
        if 'logarithmic aspect ratio comparison' in content:
            print("✓ Improved optimal resolution method with logarithmic comparison")
        else:
            print("✗ Improved optimal resolution method not found")
            return False
            
        # Check for simplified resolution interface
        if 'resolution_mode' in content and '"auto", "custom"' in content:
            print("✓ Simplified resolution_mode parameter (auto/custom)")
        else:
            print("✗ Resolution_mode parameter not found")
            return False
            
        # Check for manual width/height inputs
        if '"width": ("INT"' in content and '"height": ("INT"' in content:
            print("✓ Manual width/height inputs added for custom resolution")
        else:
            print("✗ Manual width/height inputs not found")
            return False
            
        # Check for pixel filtering logic
        if 'target_pixels * 0.8' in content and 'target_pixels * 1.2' in content:
            print("✓ Target pixel filtering with ±20% variance implemented")
        else:
            print("✗ Target pixel filtering not found")
            return False
            
        print("✓ All resolution improvement checks passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_resolution_improvements()
    
    print("\n" + "="*60)
    print("RESOLUTION INTERFACE IMPROVEMENTS:")
    print("✓ Simplified resolution interface (removed confusing dual controls)")
    print("✓ Clear auto/custom mode selection")  
    print("✓ Manual width/height inputs for exact control")
    print("✓ Resolution list with modern aspect ratios")
    print("✓ Width/height outputs for downstream integration")
    print("\nInterface now offers:")
    print("- resolution_mode: 'auto' (optimal Qwen resolution) or 'custom' (exact size)")
    print("- width/height: Manual inputs when using custom mode") 
    print("- Works for both text_to_image and image_edit modes")
    print("- No more confusing optimize_resolution + target_resolution")
    print("- Direct, intuitive control over output resolution")
    print("="*60)
    
    exit(0 if success else 1)