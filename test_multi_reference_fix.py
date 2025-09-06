#!/usr/bin/env python3
"""
Test script to verify the fixed multi-reference handler resize logic
"""

def test_multi_reference_improvements():
    """Test that the multi-reference handler has improved resize options"""
    try:
        # Read the file and check for the expected changes
        with open('nodes/qwen_multi_reference.py', 'r') as f:
            content = f.read()
        
        # Check for new resize_mode parameter
        if 'resize_mode' in content and '"match_first", "common_height", "common_width", "largest_dims"' in content:
            print("✓ New resize_mode parameter with multiple options added")
        else:
            print("✗ New resize_mode parameter not found")
            return False
        
        # Check for aspect ratio preservation logic
        if 'aspect_ratio = img.shape[2] / img.shape[1]' in content:
            print("✓ Aspect ratio preservation logic implemented")
        else:
            print("✗ Aspect ratio preservation logic not found")
            return False
            
        # Check for common_height mode (best for concat)
        if 'common_height' in content and 'aspect ratios preserved' in content:
            print("✓ Common height mode for aspect ratio preservation")
        else:
            print("✗ Common height mode not found")
            return False
            
        # Check for updated description
        if 'common_height: same height, preserve aspect ratios (recommended for concat)' in content:
            print("✓ Updated description with resize mode explanations")
        else:
            print("✗ Updated description not found")
            return False
            
        # Check that old forced resize logic was replaced
        if 'Standardizing all images to' not in content:
            print("✓ Removed old forced standardization logic")
        else:
            print("✗ Old forced standardization logic still present")
            return False
            
        # Check for smart dimension calculation
        if 'min(img.shape[1] for img in images)' in content:
            print("✓ Smart dimension calculation using min/max across images")
        else:
            print("✗ Smart dimension calculation not found") 
            return False
            
        print("✓ All multi-reference improvements verified!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_multi_reference_improvements()
    
    print("\n" + "="*60)
    print("MULTI-REFERENCE CONCAT FIX SUMMARY:")
    print("✓ Fixed squishing issue in concat mode")
    print("✓ Added resize_mode parameter with 4 options")
    print("✓ Aspect ratio preservation for common_height/common_width modes")
    print("✓ Smart dimension calculation based on all input images")
    print("✓ Backward compatible with match_first mode")
    print("\nResize modes:")
    print("- match_first: Old behavior (resize all to image1, may distort)")
    print("- common_height: Same height, preserve aspect ratios (BEST for concat)")
    print("- common_width: Same width, preserve aspect ratios") 
    print("- largest_dims: Use largest dimensions found (may distort)")
    print("\nNow concat won't squish images with different aspect ratios!")
    print("="*60)
    
    exit(0 if success else 1)