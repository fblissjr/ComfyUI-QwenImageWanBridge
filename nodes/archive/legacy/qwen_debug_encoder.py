"""
Debug encoder to analyze conditioning tensor values and statistics
Helps identify why output is burnt/pixelated compared to official node
"""

import torch
import math
import logging
import comfy.utils
import node_helpers

logger = logging.getLogger(__name__)

class QwenDebugTextEncoder:
    """
    Debug text encoder that closely mimics the official TextEncodeQwenImageEdit
    but with extensive debugging to identify value differences
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "debug_level": (["minimal", "basic", "detailed", "extreme"], {"default": "detailed"}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "debug_info")
    FUNCTION = "encode"
    CATEGORY = "QwenImage/Debug"
    DESCRIPTION = "Debug encoder that mimics official node with extensive logging"
    
    def encode(self, clip, prompt, vae=None, image=None, debug_level="detailed"):
        """
        Exact reproduction of TextEncodeQwenImageEdit with debug logging
        """
        debug_info = []
        
        # Step 1: Process image exactly like official node
        ref_latent = None
        if image is None:
            images = []
            debug_info.append("No image provided - text-only mode")
        else:
            debug_info.append(f"Input image shape: {image.shape}")
            debug_info.append(f"Input image dtype: {image.dtype}")
            debug_info.append(f"Input image range: [{image.min():.4f}, {image.max():.4f}]")
            
            # Exact copy of official movedim operation
            samples = image.movedim(-1, 1)
            debug_info.append(f"After movedim shape: {samples.shape}")
            
            # Calculate scaling exactly like official
            total = int(1024 * 1024)
            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)
            
            debug_info.append(f"Scaling: {samples.shape[3]}x{samples.shape[2]} -> {width}x{height}")
            debug_info.append(f"Scale factor: {scale_by:.6f}")
            
            # Use ComfyUI's common_upscale exactly like official
            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            image = s.movedim(1, -1)
            
            debug_info.append(f"After upscale shape: {image.shape}")
            debug_info.append(f"After upscale range: [{image.min():.4f}, {image.max():.4f}]")
            
            # Extract RGB channels exactly like official
            images = [image[:, :, :, :3]]
            debug_info.append(f"RGB image shape: {images[0].shape}")
            
            # VAE encoding if provided
            if vae is not None:
                ref_latent = vae.encode(image[:, :, :, :3])
                debug_info.append(f"Reference latent shape: {ref_latent.shape}")
                debug_info.append(f"Reference latent range: [{ref_latent.min():.4f}, {ref_latent.max():.4f}]")
        
        # Step 2: Tokenize exactly like official
        debug_info.append(f"\nTokenizing with {len(images)} images")
        tokens = clip.tokenize(prompt, images=images)
        
        if debug_level in ["detailed", "extreme"]:
            # Analyze tokens
            if isinstance(tokens, dict):
                for key in tokens:
                    if isinstance(tokens[key], list) and len(tokens[key]) > 0:
                        token_data = tokens[key][0]
                        if isinstance(token_data, list):
                            debug_info.append(f"Token key '{key}': {len(token_data)} tokens")
                            if debug_level == "extreme" and len(token_data) > 0:
                                # Sample first few tokens
                                sample_tokens = token_data[:10]
                                debug_info.append(f"  First tokens: {sample_tokens}")
        
        # Step 3: Encode tokens exactly like official
        debug_info.append("\nEncoding tokens...")
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        # Analyze conditioning tensor
        if isinstance(conditioning, list) and len(conditioning) > 0:
            cond_tensor = conditioning[0][0]
            cond_dict = conditioning[0][1] if len(conditioning[0]) > 1 else {}
            
            debug_info.append(f"\nConditioning tensor analysis:")
            debug_info.append(f"  Shape: {cond_tensor.shape}")
            debug_info.append(f"  Dtype: {cond_tensor.dtype}")
            debug_info.append(f"  Device: {cond_tensor.device}")
            debug_info.append(f"  Range: [{cond_tensor.min():.6f}, {cond_tensor.max():.6f}]")
            debug_info.append(f"  Mean: {cond_tensor.mean():.6f}")
            debug_info.append(f"  Std: {cond_tensor.std():.6f}")
            
            if debug_level in ["detailed", "extreme"]:
                # Analyze per-dimension statistics
                if len(cond_tensor.shape) >= 3:
                    dim_means = cond_tensor.mean(dim=(0, 1))
                    dim_stds = cond_tensor.std(dim=(0, 1))
                    debug_info.append(f"  Embedding dim: {cond_tensor.shape[-1]}")
                    debug_info.append(f"  Dim mean range: [{dim_means.min():.6f}, {dim_means.max():.6f}]")
                    debug_info.append(f"  Dim std range: [{dim_stds.min():.6f}, {dim_stds.max():.6f}]")
                
                # Check for NaN or Inf
                has_nan = torch.isnan(cond_tensor).any()
                has_inf = torch.isinf(cond_tensor).any()
                debug_info.append(f"  Has NaN: {has_nan}")
                debug_info.append(f"  Has Inf: {has_inf}")
                
                if debug_level == "extreme":
                    # Sample some actual values
                    flat = cond_tensor.flatten()
                    sample_size = min(10, flat.shape[0])
                    sample_values = flat[:sample_size].tolist()
                    debug_info.append(f"  Sample values: {[f'{v:.6f}' for v in sample_values]}")
            
            # Check conditioning dictionary
            debug_info.append(f"\nConditioning dict keys: {list(cond_dict.keys())}")
            if "pooled_output" in cond_dict:
                pooled = cond_dict["pooled_output"]
                if torch.is_tensor(pooled):
                    debug_info.append(f"  Pooled shape: {pooled.shape}")
                    debug_info.append(f"  Pooled range: [{pooled.min():.6f}, {pooled.max():.6f}]")
        
        # Step 4: Add reference latents exactly like official
        if ref_latent is not None:
            debug_info.append("\nAdding reference latents...")
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": [ref_latent]}, append=True
            )
            
            # Check if reference latents were added
            if isinstance(conditioning, list) and len(conditioning) > 0:
                cond_dict = conditioning[0][1] if len(conditioning[0]) > 1 else {}
                if "reference_latents" in cond_dict:
                    debug_info.append("  Reference latents successfully added")
                    ref_list = cond_dict["reference_latents"]
                    if isinstance(ref_list, list) and len(ref_list) > 0:
                        ref = ref_list[0]
                        if torch.is_tensor(ref):
                            debug_info.append(f"  Reference shape: {ref.shape}")
                            debug_info.append(f"  Reference range: [{ref.min():.6f}, {ref.max():.6f}]")
        
        # Format debug info as string
        debug_str = "\n".join(debug_info)
        
        # Print to console if detailed or extreme
        if debug_level in ["detailed", "extreme"]:
            print("\n" + "="*60)
            print("QWEN DEBUG ENCODER OUTPUT")
            print("="*60)
            print(debug_str)
            print("="*60 + "\n")
        
        return (conditioning, debug_str)


class QwenCompareEncoders:
    """
    Compare outputs from different text encoders to identify differences
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_a": ("CONDITIONING",),
                "conditioning_b": ("CONDITIONING",),
                "label_a": ("STRING", {"default": "Encoder A"}),
                "label_b": ("STRING", {"default": "Encoder B"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("comparison",)
    FUNCTION = "compare"
    CATEGORY = "QwenImage/Debug"
    DESCRIPTION = "Compare two conditioning tensors to identify differences"
    
    def compare(self, conditioning_a, conditioning_b, label_a="Encoder A", label_b="Encoder B"):
        """Compare two conditioning tensors"""
        comparison = []
        
        comparison.append(f"Comparing {label_a} vs {label_b}")
        comparison.append("="*60)
        
        # Extract tensors and dicts
        tensor_a = conditioning_a[0][0] if conditioning_a else None
        dict_a = conditioning_a[0][1] if len(conditioning_a[0]) > 1 else {}
        
        tensor_b = conditioning_b[0][0] if conditioning_b else None
        dict_b = conditioning_b[0][1] if len(conditioning_b[0]) > 1 else {}
        
        if tensor_a is not None and tensor_b is not None:
            # Compare shapes
            comparison.append(f"\nTensor Shapes:")
            comparison.append(f"  {label_a}: {tensor_a.shape}")
            comparison.append(f"  {label_b}: {tensor_b.shape}")
            
            if tensor_a.shape == tensor_b.shape:
                # Compare values
                comparison.append(f"\nTensor Statistics:")
                comparison.append(f"  {label_a} range: [{tensor_a.min():.6f}, {tensor_a.max():.6f}]")
                comparison.append(f"  {label_b} range: [{tensor_b.min():.6f}, {tensor_b.max():.6f}]")
                comparison.append(f"  {label_a} mean: {tensor_a.mean():.6f}")
                comparison.append(f"  {label_b} mean: {tensor_b.mean():.6f}")
                comparison.append(f"  {label_a} std: {tensor_a.std():.6f}")
                comparison.append(f"  {label_b} std: {tensor_b.std():.6f}")
                
                # Calculate differences
                diff = (tensor_a - tensor_b).abs()
                comparison.append(f"\nDifferences:")
                comparison.append(f"  Max absolute diff: {diff.max():.6f}")
                comparison.append(f"  Mean absolute diff: {diff.mean():.6f}")
                comparison.append(f"  % of values different: {(diff > 1e-6).float().mean() * 100:.2f}%")
                
                # Check if tensors are close
                are_close = torch.allclose(tensor_a, tensor_b, rtol=1e-5, atol=1e-5)
                comparison.append(f"  Tensors are close (rtol=1e-5): {are_close}")
                
                if not are_close:
                    # Find where they differ most
                    flat_diff = diff.flatten()
                    max_diff_idx = flat_diff.argmax()
                    max_diff_val = flat_diff[max_diff_idx]
                    
                    flat_a = tensor_a.flatten()
                    flat_b = tensor_b.flatten()
                    
                    comparison.append(f"\n  Largest difference at index {max_diff_idx}:")
                    comparison.append(f"    {label_a}: {flat_a[max_diff_idx]:.6f}")
                    comparison.append(f"    {label_b}: {flat_b[max_diff_idx]:.6f}")
                    comparison.append(f"    Diff: {max_diff_val:.6f}")
            else:
                comparison.append("\n  SHAPES DO NOT MATCH - Cannot compare values")
        
        # Compare dictionaries
        comparison.append(f"\nConditioning Dictionary Keys:")
        comparison.append(f"  {label_a}: {sorted(dict_a.keys())}")
        comparison.append(f"  {label_b}: {sorted(dict_b.keys())}")
        
        # Check for differences in keys
        keys_a = set(dict_a.keys())
        keys_b = set(dict_b.keys())
        
        only_in_a = keys_a - keys_b
        only_in_b = keys_b - keys_a
        
        if only_in_a:
            comparison.append(f"  Only in {label_a}: {sorted(only_in_a)}")
        if only_in_b:
            comparison.append(f"  Only in {label_b}: {sorted(only_in_b)}")
        
        # Format and print
        result = "\n".join(comparison)
        
        print("\n" + "="*60)
        print("ENCODER COMPARISON")
        print("="*60)
        print(result)
        print("="*60 + "\n")
        
        return (result,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenDebugTextEncoder": QwenDebugTextEncoder,
    "QwenCompareEncoders": QwenCompareEncoders,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenDebugTextEncoder": "Qwen Debug Text Encoder",
    "QwenCompareEncoders": "Qwen Compare Encoders",
}