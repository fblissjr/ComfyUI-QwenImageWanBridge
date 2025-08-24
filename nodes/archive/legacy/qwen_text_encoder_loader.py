"""
Simplified Qwen Text Encoder Loader
Works with ComfyUI's existing Load Diffusion Model workflow
"""

import torch
import folder_paths
import comfy.sd
import comfy.model_management as mm

class QwenTextEncoderLoader:
    """
    Load Qwen2.5-VL text encoder separately
    Use with Load Diffusion Model for the DiT model
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": (folder_paths.get_filename_list("text_encoders"),),
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load"
    CATEGORY = "Qwen/Loaders"
    
    def load(self, text_encoder):
        """
        Load Qwen text encoder from text_encoders folder
        """
        # Get full path
        text_encoder_path = folder_paths.get_full_path("text_encoders", text_encoder)
        
        print(f"[QwenTextEncoderLoader] Loading from: {text_encoder_path}")
        
        # Load using ComfyUI's clip loader
        clip = comfy.sd.load_clip(
            ckpt_paths=[text_encoder_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )
        
        # Add vision token IDs to the clip model for reference
        if hasattr(clip, 'cond_stage_model'):
            clip.cond_stage_model.vision_start_token_id = 151652
            clip.cond_stage_model.vision_end_token_id = 151653  
            clip.cond_stage_model.image_pad_token_id = 151655
            print("[QwenTextEncoderLoader] Added vision token IDs to CLIP model")
        
        return (clip,)


class QwenDiffusionModelLoader:
    """
    Load Qwen-Image diffusion model
    Simplified version that works with ComfyUI's standard workflow
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Qwen/Loaders"
    
    def load(self, model_name):
        """
        Load Qwen diffusion model
        """
        model_path = folder_paths.get_full_path("diffusion_models", model_name)
        
        print(f"[QwenDiffusionModelLoader] Loading from: {model_path}")
        
        # Load using ComfyUI's unet loader
        model = comfy.sd.load_diffusion_model(
            unet_path=model_path,
            model_options={}
        )
        
        # Mark as Qwen model for downstream nodes
        if hasattr(model, 'model'):
            model.model.is_qwen_image = True
            model.model.num_channels = 16  # Qwen uses 16-channel latents
            print("[QwenDiffusionModelLoader] Marked as Qwen-Image model with 16 channels")
        
        return (model,)


class QwenVAELoader:
    """
    Load Qwen-Image VAE (16-channel)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
            }
        }
    
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "Qwen/Loaders"
    
    def load(self, vae_name):
        """
        Load Qwen VAE with 16-channel support
        """
        vae_path = folder_paths.get_full_path("vae", vae_name)
        
        print(f"[QwenVAELoader] Loading from: {vae_path}")
        
        # Load VAE using ComfyUI's standard method
        import safetensors.torch
        sd = safetensors.torch.load_file(vae_path)
        
        # Create VAE instance
        vae = comfy.sd.VAE(sd=sd)
        
        # Mark as 16-channel VAE
        vae.num_channels = 16
        vae.is_qwen_vae = True
        
        # Patch the decode function to handle dimension issues
        original_decode = vae.decode
        original_encode = vae.encode
        
        def safe_decode(latent):
            # Handle 64-channel latents from sampler by taking first 16 channels
            if latent.shape[1] == 64:
                print(f"[QwenVAE] Converting 64-channel latent to 16-channel for decoding")
                latent = latent[:, :16, :, :]
            elif latent.shape[1] != 16:
                print(f"[QwenVAE] Warning: Expected 16-channel latent, got {latent.shape[1]} channels")
            
            # Ensure latent has correct dimensions
            if len(latent.shape) == 4:
                # Standard format B, C, H, W - good
                pass
            elif len(latent.shape) == 3:
                # Missing batch dimension
                latent = latent.unsqueeze(0)
            elif len(latent.shape) == 5:
                # Video format B, C, T, H, W - take first frame
                print(f"[QwenVAE] Got video format latent, taking first frame")
                latent = latent[:, :, 0, :, :]
            else:
                print(f"[QwenVAE] Unexpected latent shape for decode: {latent.shape}")
            
            return original_decode(latent)
        
        def safe_encode(image):
            # Ensure image has correct dimensions for encoding
            if len(image.shape) == 3:
                # H, W, C -> B, C, H, W
                image = image.unsqueeze(0).permute(0, 3, 1, 2)
            elif len(image.shape) == 4:
                # Check if B, H, W, C or B, C, H, W
                if image.shape[-1] <= 4:  # Likely B, H, W, C
                    image = image.permute(0, 3, 1, 2)
            
            return original_encode(image)
        
        vae.decode = safe_decode
        vae.encode = safe_encode
        
        # Add Qwen-specific normalization values
        vae.channel_mean = torch.tensor([
            -0.75711936, -0.70888418, -0.38333333, -0.37139618,
            -0.26931345, -0.2332851, -0.15819438, -0.14663641,
            -0.1356761, -0.07720678, -0.06771877, -0.0151537,
            -0.00552954, 0.03209098, 0.04779443, 0.07762699
        ])
        vae.channel_std = torch.tensor([
            2.81843495, 1.45407188, 0.91930789, 0.49169022,
            1.00717043, 0.53106242, 0.79167205, 0.42144197,
            0.80064684, 0.42448482, 0.71785361, 0.38139084,
            0.70482969, 0.37519583, 0.6934309, 0.36967117
        ])
        
        print(f"[QwenVAELoader] Loaded 16-channel Qwen VAE")
        
        return (vae,)


class QwenCheckpointLoaderSimple:
    """
    All-in-one Qwen checkpoint loader
    Loads model, CLIP, and VAE from a single checkpoint
    """
    
    @classmethod  
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"
    CATEGORY = "Qwen/Loaders"
    
    def load(self, ckpt_name):
        """
        Load complete Qwen checkpoint
        """
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        print(f"[QwenCheckpointLoader] Loading from: {ckpt_path}")
        
        # Load using ComfyUI's checkpoint loader
        model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )
        
        # Mark components as Qwen
        if model and hasattr(model, 'model'):
            model.model.is_qwen_image = True
            model.model.num_channels = 16
            
        if clip and hasattr(clip, 'cond_stage_model'):
            clip.cond_stage_model.vision_start_token_id = 151652
            clip.cond_stage_model.vision_end_token_id = 151653
            clip.cond_stage_model.image_pad_token_id = 151655
            
        if vae:
            vae.num_channels = 16
            vae.is_qwen_vae = True
        
        print(f"[QwenCheckpointLoader] Loaded Qwen-Image checkpoint with 16-channel support")
        
        return (model, clip, vae)


NODE_CLASS_MAPPINGS = {
    "QwenTextEncoderLoader": QwenTextEncoderLoader,
    "QwenDiffusionModelLoader": QwenDiffusionModelLoader,
    "QwenVAELoader": QwenVAELoader,
    "QwenCheckpointLoaderSimple": QwenCheckpointLoaderSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTextEncoderLoader": "Load Qwen Text Encoder",
    "QwenDiffusionModelLoader": "Load Qwen Diffusion Model",
    "QwenVAELoader": "Load Qwen VAE",
    "QwenCheckpointLoaderSimple": "Load Qwen Checkpoint",
}