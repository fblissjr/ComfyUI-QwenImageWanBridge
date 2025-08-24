"""
Latent I/O nodes for saving and loading latents to disk
Useful for debugging and sharing test cases
"""

import torch
import numpy as np
import os
import json
from pathlib import Path

class LoadLatentFromFile:
    """
    Load a latent tensor from disk (.pt, .safetensors, or .npz file)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": "latent.pt"}),
                "key": ("STRING", {"default": "samples"}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "load"
    CATEGORY = "QwenWANBridge/IO"
    
    def load(self, file_path, key):
        info = []
        info.append(f"Loading from: {file_path}")
        
        # Expand user path
        file_path = os.path.expanduser(file_path)
        
        # Check if file exists
        if not os.path.exists(file_path):
            # Try common locations
            comfy_path = os.path.join(os.getcwd(), file_path)
            if os.path.exists(comfy_path):
                file_path = comfy_path
            else:
                input_path = os.path.join(os.getcwd(), "input", file_path)
                if os.path.exists(input_path):
                    file_path = input_path
                else:
                    raise FileNotFoundError(f"File not found: {file_path}")
        
        info.append(f"Found at: {file_path}")
        
        # Load based on file extension
        ext = Path(file_path).suffix.lower()
        
        if ext == ".pt" or ext == ".pth":
            # PyTorch format
            data = torch.load(file_path, map_location="cpu")
            info.append("Format: PyTorch")
            
        elif ext == ".safetensors":
            # SafeTensors format
            try:
                from safetensors.torch import load_file
                data = load_file(file_path)
                info.append("Format: SafeTensors")
            except ImportError:
                raise ImportError("safetensors library not installed")
                
        elif ext == ".npz":
            # NumPy format
            np_data = np.load(file_path)
            data = {}
            for k in np_data.keys():
                data[k] = torch.from_numpy(np_data[k])
            info.append("Format: NumPy")
            
        elif ext == ".npy":
            # Single NumPy array
            np_array = np.load(file_path)
            data = {key: torch.from_numpy(np_array)}
            info.append("Format: NumPy (single array)")
            
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Extract the tensor
        if isinstance(data, torch.Tensor):
            # Direct tensor
            samples = data
            info.append("Loaded direct tensor")
        elif isinstance(data, dict):
            # Dictionary format
            if key in data:
                samples = data[key]
                info.append(f"Extracted key: {key}")
            elif "samples" in data:
                samples = data["samples"]
                info.append("Used default key: samples")
            elif len(data) == 1:
                # Single key, use it
                actual_key = list(data.keys())[0]
                samples = data[actual_key]
                info.append(f"Used only key: {actual_key}")
            else:
                info.append(f"Available keys: {list(data.keys())}")
                raise KeyError(f"Key '{key}' not found. Available: {list(data.keys())}")
        else:
            raise TypeError(f"Unexpected data type: {type(data)}")
        
        # Ensure it's a tensor
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples)
        
        # Info about loaded tensor
        info.append(f"Shape: {samples.shape}")
        info.append(f"Dtype: {samples.dtype}")
        info.append(f"Device: {samples.device}")
        info.append(f"Min: {samples.min().item():.4f}")
        info.append(f"Max: {samples.max().item():.4f}")
        info.append(f"Mean: {samples.mean().item():.4f}")
        info.append(f"Std: {samples.std().item():.4f}")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            samples = samples.cuda()
            info.append("Moved to GPU")
        
        # Create latent dictionary
        latent = {"samples": samples}
        
        return (latent, "\n".join(info))


class SaveLatentToFile:
    """
    Save a latent tensor to disk for later use
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "filename": ("STRING", {"default": "latent"}),
                "format": (["pt", "safetensors", "npz"], {"default": "pt"}),
                "save_info": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "save_path")
    FUNCTION = "save"
    CATEGORY = "QwenWANBridge/IO"
    OUTPUT_NODE = True
    
    def save(self, latent, filename, format, save_info):
        # Prepare output directory
        output_dir = os.path.join(os.getcwd(), "output", "latents")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}.{format}"
        file_path = os.path.join(output_dir, full_filename)
        
        # Extract samples
        samples = latent["samples"]
        
        # Move to CPU for saving
        samples_cpu = samples.cpu()
        
        # Save based on format
        if format == "pt":
            # PyTorch format
            torch.save({"samples": samples_cpu}, file_path)
            
        elif format == "safetensors":
            # SafeTensors format
            try:
                from safetensors.torch import save_file
                save_file({"samples": samples_cpu}, file_path)
            except ImportError:
                # Fallback to PT format
                torch.save({"samples": samples_cpu}, file_path.replace(".safetensors", ".pt"))
                file_path = file_path.replace(".safetensors", ".pt")
                
        elif format == "npz":
            # NumPy format
            np.savez_compressed(file_path, samples=samples_cpu.numpy())
        
        # Save info file if requested
        if save_info:
            info_path = file_path.replace(f".{format}", "_info.json")
            info = {
                "shape": list(samples.shape),
                "dtype": str(samples.dtype),
                "min": float(samples.min().item()),
                "max": float(samples.max().item()),
                "mean": float(samples.mean().item()),
                "std": float(samples.std().item()),
                "timestamp": timestamp,
                "format": format
            }
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
        
        return (latent, file_path)


class CreateTestLatent:
    """
    Create test latents with specific patterns for debugging
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pattern": ([
                    "zeros",
                    "ones", 
                    "random",
                    "gradient",
                    "checkerboard",
                    "center_dot",
                    "noise_levels"
                ], {"default": "random"}),
                "channels": ("INT", {"default": 16, "min": 1, "max": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                "height": ("INT", {"default": 60, "min": 8, "max": 256}),
                "width": ("INT", {"default": 104, "min": 8, "max": 256}),
                "frames": ("INT", {"default": 1, "min": 1, "max": 100}),
                "value_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "create"
    CATEGORY = "QwenWANBridge/IO"
    
    def create(self, pattern, channels, batch_size, height, width, frames, value_scale, seed):
        torch.manual_seed(seed)
        
        info = []
        info.append(f"Creating test latent: {pattern}")
        
        # Create shape
        if frames > 1:
            shape = (batch_size, channels, frames, height, width)
            info.append(f"Video latent: {shape}")
        else:
            shape = (batch_size, channels, height, width)
            info.append(f"Image latent: {shape}")
        
        # Create pattern
        if pattern == "zeros":
            tensor = torch.zeros(shape)
            
        elif pattern == "ones":
            tensor = torch.ones(shape) * value_scale
            
        elif pattern == "random":
            tensor = torch.randn(shape) * value_scale
            
        elif pattern == "gradient":
            # Create gradient from -1 to 1
            if frames > 1:
                tensor = torch.zeros(shape)
                for f in range(frames):
                    for h in range(height):
                        tensor[:, :, f, h, :] = (h / height - 0.5) * 2 * value_scale
            else:
                tensor = torch.zeros(shape)
                for h in range(height):
                    tensor[:, :, h, :] = (h / height - 0.5) * 2 * value_scale
                    
        elif pattern == "checkerboard":
            # Create checkerboard pattern
            tensor = torch.zeros(shape)
            size = 8  # Checker size
            for h in range(0, height, size*2):
                for w in range(0, width, size*2):
                    if frames > 1:
                        tensor[:, :, :, h:h+size, w:w+size] = value_scale
                        tensor[:, :, :, h+size:h+size*2, w+size:w+size*2] = value_scale
                    else:
                        tensor[:, :, h:h+size, w:w+size] = value_scale
                        tensor[:, :, h+size:h+size*2, w+size:w+size*2] = value_scale
                        
        elif pattern == "center_dot":
            # Create center dot
            tensor = torch.zeros(shape)
            center_h, center_w = height // 2, width // 2
            radius = min(height, width) // 4
            
            for h in range(height):
                for w in range(width):
                    if (h - center_h)**2 + (w - center_w)**2 < radius**2:
                        if frames > 1:
                            tensor[:, :, :, h, w] = value_scale
                        else:
                            tensor[:, :, h, w] = value_scale
                            
        elif pattern == "noise_levels":
            # Different noise level per channel
            tensor = torch.zeros(shape)
            for c in range(channels):
                noise_level = (c + 1) / channels * value_scale
                if frames > 1:
                    tensor[:, c] = torch.randn(batch_size, frames, height, width) * noise_level
                else:
                    tensor[:, c] = torch.randn(batch_size, height, width) * noise_level
        
        # Stats
        info.append(f"Min: {tensor.min().item():.4f}")
        info.append(f"Max: {tensor.max().item():.4f}")
        info.append(f"Mean: {tensor.mean().item():.4f}")
        info.append(f"Std: {tensor.std().item():.4f}")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            tensor = tensor.cuda()
            info.append("Created on GPU")
        
        latent = {"samples": tensor}
        
        return (latent, "\n".join(info))