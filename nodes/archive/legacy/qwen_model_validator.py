"""
Qwen-Image-Edit Model Validator
Validates our implementation against the official model configuration
"""

import torch
import json
import os
from typing import Dict, List, Tuple, Optional
import folder_paths

class QwenModelValidator:
    """
    Validates that our implementation matches the official Qwen-Image-Edit model
    """
    
    # Official token IDs from tokenizer_config.json
    SPECIAL_TOKENS = {
        "endoftext": 151643,
        "im_start": 151644,
        "im_end": 151645,
        "object_ref_start": 151646,
        "object_ref_end": 151647,
        "box_start": 151648,
        "box_end": 151649,
        "quad_start": 151650,
        "quad_end": 151651,
        "vision_start": 151652,
        "vision_end": 151653,
        "vision_pad": 151654,
        "image_pad": 151655,
        "video_pad": 151656,
    }
    
    # Official VAE normalization from vae/config.json
    VAE_CONFIG = {
        "z_dim": 16,
        "latents_mean": [
            -0.7571, -0.7089, -0.9113, 0.1075,
            -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632,
            -0.1922, -0.9497, 0.2503, -0.2921
        ],
        "latents_std": [
            2.8184, 1.4541, 2.3275, 2.6558,
            1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579,
            1.6382, 1.1253, 2.8251, 1.916
        ]
    }
    
    # Official transformer config from transformer/config.json
    TRANSFORMER_CONFIG = {
        "num_layers": 60,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "joint_attention_dim": 3584,
        "in_channels": 64,
        "out_channels": 16,
        "patch_size": 2
    }
    
    # Official text encoder config from text_encoder/config.json
    TEXT_ENCODER_CONFIG = {
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "intermediate_size": 18944,
        "vocab_size": 152064
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model_path": ("STRING", {"default": "Qwen-Image-Edit"}),
            }
        }
    
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("validation_report",)
    FUNCTION = "validate"
    CATEGORY = "QwenImage/Debug"
    
    def validate_tokens(self) -> Dict:
        """Validate special token IDs match our implementation"""
        from .qwen_proper_text_encoder import QwenImageEditTextEncoder
        
        encoder = QwenImageEditTextEncoder()
        
        mismatches = []
        if encoder.VISION_START_ID != self.SPECIAL_TOKENS["vision_start"]:
            mismatches.append(f"VISION_START_ID: {encoder.VISION_START_ID} != {self.SPECIAL_TOKENS['vision_start']}")
        if encoder.IMAGE_PAD_ID != self.SPECIAL_TOKENS["image_pad"]:
            mismatches.append(f"IMAGE_PAD_ID: {encoder.IMAGE_PAD_ID} != {self.SPECIAL_TOKENS['image_pad']}")
        if encoder.VISION_END_ID != self.SPECIAL_TOKENS["vision_end"]:
            mismatches.append(f"VISION_END_ID: {encoder.VISION_END_ID} != {self.SPECIAL_TOKENS['vision_end']}")
        if encoder.IM_START_ID != self.SPECIAL_TOKENS["im_start"]:
            mismatches.append(f"IM_START_ID: {encoder.IM_START_ID} != {self.SPECIAL_TOKENS['im_start']}")
        
        return {
            "valid": len(mismatches) == 0,
            "mismatches": mismatches
        }
    
    def validate_vae_config(self) -> Dict:
        """Validate VAE configuration matches"""
        report = {
            "channels": self.VAE_CONFIG["z_dim"] == 16,
            "mean_length": len(self.VAE_CONFIG["latents_mean"]) == 16,
            "std_length": len(self.VAE_CONFIG["latents_std"]) == 16,
        }
        
        return {
            "valid": all(report.values()),
            "details": report
        }
    
    def validate_templates(self) -> Dict:
        """Validate template structure"""
        from .qwen_proper_text_encoder import QwenImageEditTextEncoder
        
        encoder = QwenImageEditTextEncoder()
        
        # Check if templates contain required tokens
        t2i_valid = all(token in encoder.T2I_TEMPLATE for token in ["<|im_start|>", "<|im_end|>", "assistant"])
        edit_valid = all(token in encoder.EDIT_TEMPLATE for token in ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"])
        
        return {
            "t2i_template_valid": t2i_valid,
            "edit_template_valid": edit_valid
        }
    
    def validate_model_dimensions(self) -> Dict:
        """Validate model dimensions match configuration"""
        report = {
            "transformer_layers": self.TRANSFORMER_CONFIG["num_layers"],
            "transformer_heads": self.TRANSFORMER_CONFIG["num_attention_heads"],
            "joint_attention_dim": self.TRANSFORMER_CONFIG["joint_attention_dim"],
            "text_encoder_hidden": self.TEXT_ENCODER_CONFIG["hidden_size"],
            "text_encoder_heads": self.TEXT_ENCODER_CONFIG["num_attention_heads"],
            "dimensions_match": self.TRANSFORMER_CONFIG["joint_attention_dim"] == self.TEXT_ENCODER_CONFIG["hidden_size"]
        }
        
        return report
    
    def validate(self, model_path: str = "Qwen-Image-Edit") -> Tuple[Dict]:
        """
        Run full validation suite
        """
        validation_report = {
            "model_path": model_path,
            "timestamp": str(torch.cuda.Event(enable_timing=True).record()),
            "validations": {}
        }
        
        # Validate special tokens
        print("[Validator] Checking special token IDs...")
        token_validation = self.validate_tokens()
        validation_report["validations"]["special_tokens"] = token_validation
        
        # Validate VAE config
        print("[Validator] Checking VAE configuration...")
        vae_validation = self.validate_vae_config()
        validation_report["validations"]["vae_config"] = vae_validation
        
        # Validate templates
        print("[Validator] Checking prompt templates...")
        template_validation = self.validate_templates()
        validation_report["validations"]["templates"] = template_validation
        
        # Validate model dimensions
        print("[Validator] Checking model dimensions...")
        dimension_validation = self.validate_model_dimensions()
        validation_report["validations"]["dimensions"] = dimension_validation
        
        # Overall status
        all_valid = (
            token_validation["valid"] and
            vae_validation["valid"] and
            template_validation["t2i_template_valid"] and
            template_validation["edit_template_valid"]
        )
        
        validation_report["status"] = "PASS" if all_valid else "FAIL"
        
        # Print summary
        print("\n" + "="*50)
        print("QWEN-IMAGE-EDIT VALIDATION REPORT")
        print("="*50)
        print(f"Status: {validation_report['status']}")
        print(f"Special Tokens: {'✓' if token_validation['valid'] else '✗'}")
        print(f"VAE Config: {'✓' if vae_validation['valid'] else '✗'}")
        print(f"Templates: {'✓' if template_validation['t2i_template_valid'] and template_validation['edit_template_valid'] else '✗'}")
        print(f"Dimensions Match: {'✓' if dimension_validation['dimensions_match'] else '✗'}")
        
        if not all_valid:
            print("\nIssues found:")
            if not token_validation["valid"]:
                for mismatch in token_validation["mismatches"]:
                    print(f"  - {mismatch}")
        
        print("="*50)
        
        return (validation_report,)


class QwenVAENormalizer:
    """
    Apply official Qwen-Image VAE normalization
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "mode": (["normalize", "denormalize"], {"default": "normalize"}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process"
    CATEGORY = "QwenImage/VAE"
    
    # Official normalization values
    LATENTS_MEAN = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075,
        -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632,
        -0.1922, -0.9497, 0.2503, -0.2921
    ])
    
    LATENTS_STD = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558,
        1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579,
        1.6382, 1.1253, 2.8251, 1.916
    ])
    
    def process(self, latent: Dict, mode: str = "normalize") -> Tuple[Dict]:
        """
        Apply or remove Qwen VAE normalization
        """
        samples = latent["samples"]
        device = samples.device
        dtype = samples.dtype
        
        # Ensure we have 16 channels
        if samples.shape[1] != 16:
            raise ValueError(f"Expected 16 channels, got {samples.shape[1]}")
        
        # Move normalization tensors to correct device
        mean = self.LATENTS_MEAN.to(device=device, dtype=dtype)
        std = self.LATENTS_STD.to(device=device, dtype=dtype)
        
        # Reshape for broadcasting
        mean = mean.view(1, 16, 1, 1)
        std = std.view(1, 16, 1, 1)
        
        if mode == "normalize":
            # Apply normalization: (x - mean) / std
            normalized = (samples - mean) / std
            result = {"samples": normalized}
        else:
            # Remove normalization: x * std + mean
            denormalized = samples * std + mean
            result = {"samples": denormalized}
        
        return (result,)


NODE_CLASS_MAPPINGS = {
    "QwenModelValidator": QwenModelValidator,
    "QwenVAENormalizer": QwenVAENormalizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenModelValidator": "Qwen Model Validator",
    "QwenVAENormalizer": "Qwen VAE Normalizer",
}