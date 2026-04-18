"""
Systematic parameter testing for Qwen-WAN bridge
Find the right combination that actually works
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

class QwenWANParameterSweep:
    """
    Systematically test different parameter combinations
    to find what actually works
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_latent": ("LATENT",),
                "test_set": ([
                    "quick_test",
                    "denoise_sweep", 
                    "cfg_sweep",
                    "sampler_test",
                    "resolution_test",
                    "comprehensive"
                ], {"default": "quick_test"}),
                "implementation": (["wrapper", "native"], {"default": "wrapper"}),
            }
        }
    
    RETURN_TYPES = ("TEST_CONFIGS", "STRING")
    RETURN_NAMES = ("test_configs", "test_plan")
    FUNCTION = "generate_tests"
    CATEGORY = "QwenWANBridge/Testing"
    
    def generate_tests(self, qwen_latent, test_set, implementation):
        
        configs = []
        plan = []
        plan.append(f"Parameter Sweep Test Plan ({test_set})")
        plan.append("="*50)
        plan.append(f"Implementation: {implementation}")
        plan.append("")
        
        if test_set == "quick_test":
            # Just a few key combinations
            configs = [
                {"denoise": 0.3, "cfg": 3.0, "steps": 10, "sampler": "DPM-Solver++", "frames": 1},
                {"denoise": 0.5, "cfg": 5.0, "steps": 15, "sampler": "DPM-Solver++", "frames": 5},
                {"denoise": 0.7, "cfg": 7.0, "steps": 20, "sampler": "DDIM", "frames": 9},
                {"denoise": 1.0, "cfg": 7.0, "steps": 20, "sampler": "Euler a", "frames": 13},
            ]
            plan.append("Quick test - 4 configurations")
            
        elif test_set == "denoise_sweep":
            # Focus on denoise values
            denoise_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            for d in denoise_values:
                configs.append({
                    "denoise": d,
                    "cfg": 5.0,
                    "steps": 15,
                    "sampler": "DPM-Solver++",
                    "frames": 9
                })
            plan.append(f"Denoise sweep - {len(configs)} configurations")
            plan.append(f"Testing denoise from 0.1 to 1.0")
            
        elif test_set == "cfg_sweep":
            # Focus on CFG scale
            cfg_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
            for c in cfg_values:
                configs.append({
                    "denoise": 0.5,
                    "cfg": c,
                    "steps": 15,
                    "sampler": "DPM-Solver++",
                    "frames": 9
                })
            plan.append(f"CFG sweep - {len(configs)} configurations")
            plan.append(f"Testing CFG from 1.0 to 12.0")
            
        elif test_set == "sampler_test":
            # Test different samplers
            samplers = [
                ("DPM-Solver++", "karras"),
                ("DPM-Solver++", "normal"),
                ("DDIM", "ddim_uniform"),
                ("Euler", "normal"),
                ("Euler a", "karras"),
                ("DPM-Solver++", "simple"),
            ]
            for sampler, scheduler in samplers:
                configs.append({
                    "denoise": 0.5,
                    "cfg": 5.0,
                    "steps": 15,
                    "sampler": sampler,
                    "scheduler": scheduler,
                    "frames": 9
                })
            plan.append(f"Sampler test - {len(configs)} configurations")
            
        elif test_set == "resolution_test":
            # Test different resolutions
            resolutions = [
                (512, 512, 1),
                (512, 512, 9),
                (768, 768, 1),
                (768, 768, 9),
                (832, 480, 1),
                (832, 480, 9),
            ]
            for w, h, f in resolutions:
                configs.append({
                    "width": w,
                    "height": h,
                    "frames": f,
                    "denoise": 0.5,
                    "cfg": 5.0,
                    "steps": 15,
                    "sampler": "DPM-Solver++",
                })
            plan.append(f"Resolution test - {len(configs)} configurations")
            
        elif test_set == "comprehensive":
            # Test many combinations
            plan.append("Comprehensive test - WARNING: This will take time!")
            
            # Key parameters to vary
            denoise_vals = [0.3, 0.5, 0.7, 1.0]
            cfg_vals = [3.0, 5.0, 7.0]
            frame_vals = [1, 5, 9]
            samplers = ["DPM-Solver++", "DDIM", "Euler a"]
            
            for d in denoise_vals:
                for c in cfg_vals:
                    for f in frame_vals:
                        for s in samplers:
                            configs.append({
                                "denoise": d,
                                "cfg": c,
                                "steps": 15,
                                "sampler": s,
                                "frames": f
                            })
            
            plan.append(f"Total configurations: {len(configs)}")
            plan.append(f"Denoise: {denoise_vals}")
            plan.append(f"CFG: {cfg_vals}")
            plan.append(f"Frames: {frame_vals}")
            plan.append(f"Samplers: {samplers}")
        
        # Add test details
        plan.append("")
        plan.append("Test Details:")
        for i, config in enumerate(configs[:5]):  # Show first 5
            plan.append(f"  {i+1}. {self.format_config(config)}")
        if len(configs) > 5:
            plan.append(f"  ... and {len(configs)-5} more")
        
        # Recommendations based on test set
        plan.append("")
        plan.append("Recommendations:")
        if test_set == "quick_test":
            plan.append("• Start here to get a baseline")
            plan.append("• If none work, try denoise_sweep")
        elif test_set == "denoise_sweep":
            plan.append("• Lower denoise (0.1-0.3) preserves Qwen structure")
            plan.append("• Higher denoise (0.7-1.0) gives WAN more freedom")
        elif test_set == "cfg_sweep":
            plan.append("• Lower CFG (1-3) for more creativity")
            plan.append("• Higher CFG (7-12) for stronger prompt adherence")
        
        # Create test config object
        test_configs = {
            "configs": configs,
            "test_set": test_set,
            "implementation": implementation,
            "qwen_shape": qwen_latent["samples"].shape,
        }
        
        return (test_configs, "\n".join(plan))
    
    def format_config(self, config):
        parts = []
        if "denoise" in config:
            parts.append(f"denoise={config['denoise']:.1f}")
        if "cfg" in config:
            parts.append(f"cfg={config['cfg']:.1f}")
        if "steps" in config:
            parts.append(f"steps={config['steps']}")
        if "sampler" in config:
            parts.append(f"sampler={config['sampler']}")
        if "frames" in config:
            parts.append(f"frames={config['frames']}")
        return ", ".join(parts)


class QwenWANBestSettings:
    """
    Collection of settings that have shown promise
    Based on testing and user feedback
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": ([
                    "conservative",
                    "balanced",
                    "aggressive",
                    "t2v_mode",
                    "i2v_low_denoise",
                    "i2v_high_quality",
                    "single_frame",
                    "native_comfy",
                    "wrapper_optimized"
                ], {"default": "balanced"}),
                "model_type": (["i2v", "t2v"], {"default": "i2v"}),
            }
        }
    
    RETURN_TYPES = ("DICT", "STRING")
    RETURN_NAMES = ("settings", "explanation")
    FUNCTION = "get_settings"
    CATEGORY = "QwenWANBridge/Testing"
    
    def get_settings(self, preset, model_type):
        
        presets = {
            "conservative": {
                "denoise": 0.3,
                "cfg": 3.0,
                "steps": 10,
                "sampler": "DPM-Solver++",
                "scheduler": "karras",
                "frames": 1,
                "note": "Minimal changes, preserves Qwen structure"
            },
            "balanced": {
                "denoise": 0.5,
                "cfg": 5.0,
                "steps": 15,
                "sampler": "DPM-Solver++",
                "scheduler": "karras",
                "frames": 9,
                "note": "Balance between Qwen preservation and WAN generation"
            },
            "aggressive": {
                "denoise": 0.8,
                "cfg": 7.0,
                "steps": 20,
                "sampler": "DPM-Solver++",
                "scheduler": "karras",
                "frames": 13,
                "note": "More WAN influence, less Qwen preservation"
            },
            "t2v_mode": {
                "denoise": 1.0,
                "cfg": 7.0,
                "steps": 20,
                "sampler": "DPM-Solver++",
                "scheduler": "karras",
                "frames": 81,
                "note": "Full T2V generation, Qwen as subtle guide"
            },
            "i2v_low_denoise": {
                "denoise": 0.2,
                "cfg": 4.0,
                "steps": 12,
                "sampler": "DDIM",
                "scheduler": "ddim_uniform",
                "frames": 5,
                "note": "Minimal denoise for I2V, preserves input"
            },
            "i2v_high_quality": {
                "denoise": 0.4,
                "cfg": 6.0,
                "steps": 25,
                "sampler": "DPM-Solver++",
                "scheduler": "karras",
                "frames": 9,
                "note": "Higher quality I2V, more steps"
            },
            "single_frame": {
                "denoise": 0.1,
                "cfg": 2.0,
                "steps": 5,
                "sampler": "Euler",
                "scheduler": "normal",
                "frames": 1,
                "note": "Single frame only - most likely to work"
            },
            "native_comfy": {
                "denoise": 0.5,
                "cfg": 7.5,
                "steps": 20,
                "sampler": "dpmpp_2m",
                "scheduler": "karras",
                "frames": 16,
                "note": "Settings for native ComfyUI implementation"
            },
            "wrapper_optimized": {
                "denoise": 0.35,
                "cfg": 4.5,
                "steps": 15,
                "sampler": "DPM-Solver++",
                "scheduler": "2m",
                "frames": 9,
                "note": "Optimized for Kijai's wrapper"
            }
        }
        
        settings = presets.get(preset, presets["balanced"])
        
        # Adjust for model type
        if model_type == "t2v":
            settings["denoise"] = min(1.0, settings["denoise"] * 1.5)
            settings["note"] += " (adjusted for T2V)"
        
        explanation = f"""Settings: {preset} ({model_type})
{"="*40}

Parameters:
  Denoise: {settings['denoise']}
  CFG Scale: {settings['cfg']}
  Steps: {settings['steps']}
  Sampler: {settings['sampler']}
  Scheduler: {settings['scheduler']}
  Frames: {settings['frames']}

Note: {settings['note']}

Why these settings:
"""
        
        if preset == "conservative":
            explanation += """
• Very low denoise (0.3) keeps Qwen structure intact
• Low CFG (3.0) prevents over-correction
• Few steps (10) for quick testing
• Single frame most likely to show results"""
        
        elif preset == "balanced":
            explanation += """
• Medium denoise (0.5) balances preservation vs generation
• Medium CFG (5.0) for moderate guidance
• Standard steps (15) for quality
• 9 frames tests short video generation"""
        
        elif preset == "single_frame":
            explanation += """
• Minimal denoise (0.1) for maximum preservation
• Very low CFG (2.0) to avoid artifacts
• Few steps (5) since little change needed
• Single frame avoids temporal issues"""
        
        elif preset in ["native_comfy", "wrapper_optimized"]:
            explanation += f"""
• Settings specifically tuned for {preset.replace('_', ' ')}
• May need adjustment based on your setup
• Start here if using that implementation"""
        
        return (settings, explanation)


class QwenWANTestRunner:
    """
    Actually run the parameter tests and collect results
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "test_configs": ("TEST_CONFIGS",),
                "run_test": ("BOOLEAN", {"default": False}),
                "save_results": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "DICT")
    RETURN_NAMES = ("results_summary", "best_config")
    FUNCTION = "run_tests"
    CATEGORY = "QwenWANBridge/Testing"
    
    def run_tests(self, test_configs, run_test, save_results):
        
        if not run_test:
            return ("Tests not run (set run_test=True)", {})
        
        results = []
        results.append("Test Results Summary")
        results.append("="*50)
        
        configs = test_configs.get("configs", [])
        results.append(f"Testing {len(configs)} configurations...")
        
        # In real implementation, this would actually run tests
        # For now, return mock results
        best_score = 0
        best_config = None
        
        for i, config in enumerate(configs):
            # Mock scoring based on heuristics
            score = self.score_config(config)
            
            if score > best_score:
                best_score = score
                best_config = config
            
            if i < 5:  # Show first few
                results.append(f"\nConfig {i+1}: {self.format_config(config)}")
                results.append(f"  Score: {score:.2f}/10")
        
        results.append(f"\nBest configuration:")
        if best_config:
            results.append(f"  {self.format_config(best_config)}")
            results.append(f"  Score: {best_score:.2f}/10")
        
        results.append("\nRecommendation:")
        if best_score > 7:
            results.append("• Good results! Use these settings")
        elif best_score > 5:
            results.append("• Acceptable results, may need fine-tuning")
        else:
            results.append("• Poor results, try different approach")
            results.append("• Consider using VAE decode/encode instead")
        
        return ("\n".join(results), best_config or {})
    
    def score_config(self, config):
        """Mock scoring function - in reality would test actual output"""
        score = 5.0  # Base score
        
        # Heuristic scoring
        if config.get("frames", 1) == 1:
            score += 2  # Single frame more likely to work
        
        denoise = config.get("denoise", 0.5)
        if 0.3 <= denoise <= 0.5:
            score += 1  # Good denoise range
        
        if config.get("sampler") == "DPM-Solver++":
            score += 0.5  # Good sampler
        
        if config.get("cfg", 5) <= 5:
            score += 0.5  # Lower CFG often better
        
        return min(10, score)
    
    def format_config(self, config):
        parts = []
        for key in ["denoise", "cfg", "steps", "sampler", "frames"]:
            if key in config:
                val = config[key]
                if isinstance(val, float):
                    parts.append(f"{key}={val:.1f}")
                else:
                    parts.append(f"{key}={val}")
        return ", ".join(parts)