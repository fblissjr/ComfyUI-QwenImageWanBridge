# ComfyUI-QwenImageWanBridge

Direct latent bridge from Qwen-Image to WAN video generation.

**WAN 2.1**: 16 channels (Compatible with Qwen-Image!)
**WAN 2.2**: 48 channels (Causes pixelation - 16ch→48ch mismatch)

Despite 99.98% VAE similarity and semantic alignment (both Alibaba, WAN trained on Qwen2.5-VL), the small latent distribution differences cause quality loss. I2V models are extremely sensitive to exact latent characteristics.

## Other Thoughts

1. Frame 0 should not have noise, BUT frames 0+n needs to absolutely think frame 0 has noise and its just denoising from here... so how?
2. If we aligned the two text embeddings, could we somehow predict the noise that frame 0 needs to have?
3. How do we create frames 0+n if we want frame 0 to be the input image without using I2V?
4. Can V2V play a role? (or in this case, simulated V2V)

## Credits

- Qwen-Image and WAN 2.2 by Alibaba
- ComfyUI-WanVideoWrapper by Kijai
