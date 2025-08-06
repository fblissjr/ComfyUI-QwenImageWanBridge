# ComfyUI-QwenImageWanBridge

Custom nodes for bridging Qwen-Image and WAN video models in ComfyUI.

## Nodes

### Diagnostic Nodes

#### VAE Diagnostic
Analyzes compatibility between Qwen and WAN VAEs
- Compares latent encodings
- Shows statistical differences
- Recommends best practices

#### Simple VAE Test
Quick recommendations for different workflows
- Qwen to WAN
- WAN to Qwen
- Best practices

### 🌉 Bridge Nodes

#### Qwen-WAN Bridge
Bridges latents between models
- Handles dimension differences
- Fixes temporal structure
- Uses WAN VAE (always!)

#### Always Use WAN VAE
Enforces the golden rule - WAN VAE for everything
- Decodes with WAN VAE
- Works with both model types
- No bizarre frames!

#### Latent Mixer
Mix latents from different models
- Linear or spherical interpolation
- Style mixing
- Cross-model blending

## Example Workflows

### Basic Cross-Model Workflow
```
[Qwen Model] → [KSampler] → [Always Use WAN VAE] → [Image]
                                    ↑
                              [Load WAN VAE]
```

### Diagnostic Workflow
```
[Load Image] → [VAE Diagnostic] → [View Report]
                ↑            ↑
          [Qwen VAE]    [WAN VAE]
```

### Style Mixing Workflow
```
[Qwen Generate] → [Latent Mixer] → [Always Use WAN VAE] → [Image]
[WAN Generate] ↗                            ↑
                                       [Load WAN VAE]
```
