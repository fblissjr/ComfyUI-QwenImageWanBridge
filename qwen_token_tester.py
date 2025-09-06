#!/usr/bin/env python3
"""
Qwen2.5-VL Token Testing Tool
Tests actual tokenization behavior with the special tokens.
"""

import json
import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoProcessor
    import torch
except ImportError:
    print("Required packages not found. Install with:")
    print("pip install transformers torch")
    sys.exit(1)

class QwenTokenTester:
    def __init__(self, model_path: str = "/Users/fredbliss/Storage/Qwen-Image-Edit"):
        self.model_path = model_path
        self.tokenizer = None
        self.processor = None
        self.load_tokenizer()
    
    def load_tokenizer(self):
        """Load the tokenizer from the local path"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            print(f"✓ Loaded tokenizer from {self.model_path}")
            print(f"  Vocab size: {self.tokenizer.vocab_size}")
            print(f"  Model max length: {self.tokenizer.model_max_length}")
            
        except Exception as e:
            print(f"✗ Failed to load tokenizer: {e}")
            print("Make sure the model path contains valid tokenizer files")
    
    def test_special_tokens(self):
        """Test how special tokens are tokenized"""
        if not self.tokenizer:
            return
        
        print("\n=== Special Token Testing ===")
        
        special_tokens = [
            "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
            "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>",
            "<|quad_start|>", "<|quad_end|>",
            "<|im_start|>", "<|im_end|>"
        ]
        
        for token in special_tokens:
            try:
                # Test tokenization
                encoded = self.tokenizer.encode(token, add_special_tokens=False)
                decoded = self.tokenizer.decode(encoded)
                
                # Check if it's a single token
                is_single = len(encoded) == 1
                token_id = encoded[0] if is_single else "MULTIPLE"
                
                print(f"{token:20} → {token_id:6} ({'✓' if is_single else '✗'}) → '{decoded}'")
                
                if not is_single:
                    print(f"  Full encoding: {encoded}")
                    
            except Exception as e:
                print(f"{token:20} → ERROR: {e}")
    
    def test_vision_sequence(self):
        """Test a complete vision processing sequence"""
        if not self.tokenizer:
            return
        
        print("\n=== Vision Sequence Testing ===")
        
        # Test different vision sequences
        sequences = [
            "<|vision_start|><|image_pad|><|vision_end|>",
            "<|im_start|>user\nDescribe this image: <|vision_start|><|image_pad|><|vision_end|><|im_end|>",
            "The <|object_ref_start|>red car<|object_ref_end|> is at <|box_start|>100,50,200,150<|box_end|>"
        ]
        
        for i, seq in enumerate(sequences, 1):
            print(f"\nSequence {i}: {seq}")
            try:
                encoded = self.tokenizer.encode(seq, add_special_tokens=False)
                decoded = self.tokenizer.decode(encoded)
                
                print(f"  Tokens: {len(encoded)}")
                print(f"  Token IDs: {encoded}")
                print(f"  Decoded: '{decoded}'")
                
                # Check if decoding matches original
                matches = decoded == seq
                print(f"  Round-trip: {'✓' if matches else '✗'}")
                
                if not matches:
                    print(f"  Difference: Original vs Decoded")
                    print(f"    Original: '{seq}'")
                    print(f"    Decoded:  '{decoded}'")
                
            except Exception as e:
                print(f"  ERROR: {e}")
    
    def test_spatial_coordinates(self):
        """Test how spatial coordinates are handled"""
        if not self.tokenizer:
            return
        
        print("\n=== Spatial Coordinates Testing ===")
        
        # Test different coordinate formats
        coord_tests = [
            "<|box_start|>100,50,200,150<|box_end|>",
            "<|box_start|>0.1,0.2,0.3,0.4<|box_end|>",  # Normalized coordinates
            "<|quad_start|>10,20 100,25 95,80 8,75<|quad_end|>",  # Polygon
            "Move the object from <|box_start|>100,100,200,200<|box_end|> to <|box_start|>300,100,400,200<|box_end|>"
        ]
        
        for test_str in coord_tests:
            print(f"\nTesting: {test_str}")
            try:
                encoded = self.tokenizer.encode(test_str, add_special_tokens=False)
                decoded = self.tokenizer.decode(encoded)
                
                print(f"  Tokens: {len(encoded)}")
                print(f"  Decoded: '{decoded}'")
                
                # Analyze the coordinate tokens
                coord_start = None
                coord_end = None
                for i, token_id in enumerate(encoded):
                    token_text = self.tokenizer.decode([token_id])
                    if '<|box_start|>' in token_text or '<|quad_start|>' in token_text:
                        coord_start = i
                    elif '<|box_end|>' in token_text or '<|quad_end|>' in token_text:
                        coord_end = i
                        break
                
                if coord_start is not None and coord_end is not None:
                    coord_tokens = encoded[coord_start+1:coord_end]
                    coord_text = self.tokenizer.decode(coord_tokens)
                    print(f"  Coordinate content: '{coord_text}' ({len(coord_tokens)} tokens)")
                
            except Exception as e:
                print(f"  ERROR: {e}")
    
    def test_template_formats(self):
        """Test different template formats that might be used"""
        if not self.tokenizer:
            return
            
        print("\n=== Template Format Testing ===")
        
        templates = [
            # Image editing templates
            "<|im_start|>system\nYou are an AI image editor.<|im_end|>\n<|im_start|>user\nEdit this image: <|vision_start|><|image_pad|><|vision_end|>\nMake it brighter.<|im_end|>",
            
            # Object detection template  
            "<|im_start|>user\nFind all cars in this image: <|vision_start|><|image_pad|><|vision_end|><|im_end|>",
            
            # Spatial editing template
            "<|im_start|>user\nEdit the region <|box_start|>100,100,200,200<|box_end|> in this image: <|vision_start|><|image_pad|><|vision_end|>\nChange it to a garden.<|im_end|>",
        ]
        
        for i, template in enumerate(templates, 1):
            print(f"\n--- Template {i} ---")
            print(f"Template: {template[:100]}...")
            
            try:
                encoded = self.tokenizer.encode(template, add_special_tokens=False)
                decoded = self.tokenizer.decode(encoded)
                
                print(f"Total tokens: {len(encoded)}")
                print(f"Round-trip match: {'✓' if decoded == template else '✗'}")
                
                # Count special tokens
                special_count = 0
                for token_id in encoded:
                    if token_id >= 151643:  # Special token range
                        special_count += 1
                
                print(f"Special tokens: {special_count}")
                
            except Exception as e:
                print(f"ERROR: {e}")
    
    def analyze_token_ids(self):
        """Analyze the token ID space and mappings"""
        if not self.tokenizer:
            return
            
        print("\n=== Token ID Analysis ===")
        
        # Get vocab size and special token range
        vocab_size = self.tokenizer.vocab_size
        print(f"Base vocabulary size: {vocab_size}")
        
        # Check added tokens
        if hasattr(self.tokenizer, 'added_tokens_decoder'):
            added_tokens = self.tokenizer.added_tokens_decoder
            print(f"Added tokens: {len(added_tokens)}")
            
            # Analyze the range
            token_ids = [int(k) for k in added_tokens.keys()]
            min_id, max_id = min(token_ids), max(token_ids)
            print(f"Token ID range: {min_id} - {max_id}")
            print(f"Range size: {max_id - min_id + 1}")
            
            # Check for gaps
            expected_ids = set(range(min_id, max_id + 1))
            actual_ids = set(token_ids)
            gaps = expected_ids - actual_ids
            
            if gaps:
                print(f"Gaps in sequence: {sorted(gaps)}")
            else:
                print("No gaps in token ID sequence")
        
        # Test edge cases around vocabulary boundary
        test_ids = [vocab_size - 1, vocab_size, 151643, 151664]
        print(f"\nEdge case testing:")
        for test_id in test_ids:
            try:
                decoded = self.tokenizer.decode([test_id])
                print(f"  ID {test_id}: '{decoded}'")
            except Exception as e:
                print(f"  ID {test_id}: ERROR - {e}")
    
    def interactive_test(self):
        """Interactive testing mode"""
        if not self.tokenizer:
            return
            
        print("\n=== Interactive Testing Mode ===")
        print("Enter text to tokenize (or 'quit' to exit):")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                # Tokenize
                encoded = self.tokenizer.encode(user_input, add_special_tokens=False)
                decoded = self.tokenizer.decode(encoded)
                
                print(f"Tokens: {len(encoded)}")
                print(f"Token IDs: {encoded}")
                print(f"Decoded: '{decoded}'")
                print(f"Round-trip: {'✓' if decoded == user_input else '✗'}")
                
                # Show individual tokens
                if len(encoded) <= 20:  # Don't spam for very long sequences
                    print("Individual tokens:")
                    for i, token_id in enumerate(encoded):
                        token_text = self.tokenizer.decode([token_id])
                        special_mark = " [SPECIAL]" if token_id >= 151643 else ""
                        print(f"  {i:2d}: {token_id:6d} → '{token_text}'{special_mark}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"ERROR: {e}")
        
        print("Goodbye!")
    
    def run_all_tests(self):
        """Run all automated tests"""
        print("=== Qwen2.5-VL Token Testing Suite ===")
        
        if not self.tokenizer:
            print("Cannot run tests without a valid tokenizer")
            return
        
        self.test_special_tokens()
        self.test_vision_sequence()
        self.test_spatial_coordinates()
        self.test_template_formats()
        self.analyze_token_ids()
        
        print("\n=== Test Suite Complete ===")
        print("Run with --interactive for interactive testing mode")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Token Testing Tool")
    parser.add_argument("--model-path", default="/Users/fredbliss/Storage/Qwen-Image-Edit",
                       help="Path to the Qwen model directory")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive testing mode")
    
    args = parser.parse_args()
    
    tester = QwenTokenTester(args.model_path)
    
    if args.interactive:
        tester.interactive_test()
    else:
        tester.run_all_tests()


if __name__ == "__main__":
    main()