"""
Qwen Processor Node for Wrapper System
Processes text and images using the loaded processor from QwenVLTextEncoderLoaderWrapper
"""

import torch
import logging
from typing import Optional, List, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Set up verbose logging
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class QwenProcessorWrapper:
    """Process text and images using the Qwen2.5-VL processor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processor": ("QWEN_PROCESSOR",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape"
                }),
            },
            "optional": {
                "images": ("IMAGE",),
                "mode": (["text_to_image", "image_edit"], {
                    "default": "text_to_image"
                }),
                "add_picture_labels": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add 'Picture X:' labels for multiple images"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "System prompt for template"
                }),
                "return_tensors": (["pt", "np"], {
                    "default": "pt"
                }),
            }
        }

    RETURN_TYPES = ("QWEN_PROCESSED", "STRING")
    RETURN_NAMES = ("processed", "formatted_text")
    FUNCTION = "process"
    CATEGORY = "QwenImage/Wrapper"
    TITLE = "Qwen Processor (Wrapper)"

    def process(self, processor, text: str, images=None, mode: str = "text_to_image",
                add_picture_labels: bool = True, system_prompt: str = "",
                return_tensors: str = "pt"):
        """
        Process text and images using the Qwen processor.

        This handles:
        - Text tokenization
        - Image preprocessing (resize, normalize)
        - Vision token insertion
        - "Picture X:" labeling
        - Template formatting
        """

        logger.info("="*60)
        logger.info("QWEN PROCESSOR - Starting")
        logger.info(f"Mode: {mode}")
        logger.info(f"Text length: {len(text)} chars")
        logger.info(f"Images provided: {images is not None}")
        if images is not None:
            logger.info(f"Image shape: {images.shape if hasattr(images, 'shape') else 'unknown'}")
        logger.info(f"Add picture labels: {add_picture_labels}")
        logger.info(f"System prompt: {bool(system_prompt)}")
        logger.info("="*60)

        if processor is None:
            logger.error("No processor provided!")
            raise ValueError("No processor provided. Connect a QwenVLTextEncoderLoaderWrapper.")

        formatted_text = text
        vision_tokens = ""

        # Handle image processing
        if images is not None and mode == "image_edit":
            # Convert ComfyUI images (B, H, W, C) to PIL images
            pil_images = []

            if len(images.shape) == 4:  # Batch of images
                for i in range(images.shape[0]):
                    img_array = (images[i].cpu().numpy() * 255).astype(np.uint8)
                    pil_images.append(Image.fromarray(img_array))
            else:  # Single image
                img_array = (images.cpu().numpy() * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_array))

            # Add Picture labels if requested
            if add_picture_labels and len(pil_images) > 1:
                picture_labels = []
                for i in range(len(pil_images)):
                    picture_labels.append(f"Picture {i+1}:")
                    vision_tokens += f"Picture {i+1}: <|vision_start|><|image_pad|><|vision_end|>"

                # Update formatted text with labels
                formatted_text = vision_tokens + text
            else:
                vision_tokens = "<|vision_start|><|image_pad|><|vision_end|>" * len(pil_images)
                formatted_text = vision_tokens + text

        # Format with template if system prompt provided
        if system_prompt:
            formatted_text = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{formatted_text}<|im_end|>
<|im_start|>assistant
"""

        # Process with the actual processor
        try:
            if images is not None and mode == "image_edit":
                # Process with images
                processed = processor(
                    text=formatted_text,
                    images=pil_images,
                    return_tensors=return_tensors
                )
            else:
                # Text only
                processed = processor(
                    text=formatted_text,
                    return_tensors=return_tensors
                )

            logger.info(f"Processed text with {len(pil_images) if images is not None else 0} images")

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            # Fallback to simple dict
            processed = {
                "input_ids": torch.zeros((1, 100), dtype=torch.long),
                "attention_mask": torch.ones((1, 100), dtype=torch.long),
            }

        return (processed, formatted_text)


class QwenProcessedToEmbedding:
    """Convert processed output to embeddings using text encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("QWEN_TEXT_ENCODER",),
                "processed": ("QWEN_PROCESSED",),
            },
            "optional": {
                "drop_tokens": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 128,
                    "tooltip": "Number of tokens to drop (34 for T2I, 64 for I2E)"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "QwenImage/Wrapper"
    TITLE = "Qwen Processed to Embedding (Wrapper)"

    def encode(self, text_encoder, processed, drop_tokens: int = 0):
        """
        Convert processed tokens to embeddings using the text encoder.

        This replaces the standard text encoding with processor-based encoding.
        """

        if text_encoder is None:
            raise ValueError("No text encoder provided.")

        if processed is None:
            raise ValueError("No processed input provided.")

        # Handle different input types
        if not isinstance(processed, dict):
            # If it's a BatchEncoding or similar, convert to dict
            if hasattr(processed, 'data'):
                processed = processed.data
            elif hasattr(processed, '__dict__'):
                processed = vars(processed)
            else:
                logger.error(f"Processed input type: {type(processed)}")
                logger.error(f"Processed input: {processed}")
                raise ValueError(f"Invalid processed input type: {type(processed)}")

        # Get input IDs and attention mask
        input_ids = processed.get("input_ids")
        attention_mask = processed.get("attention_mask", None)
        pixel_values = processed.get("pixel_values", None)
        image_grid_thw = processed.get("image_grid_thw", None)

        if input_ids is None:
            logger.error(f"Processed keys: {list(processed.keys())}")
            raise ValueError("No input_ids in processed output.")

        # Move to correct device
        device = 'cpu'
        if hasattr(text_encoder, 'parameters'):
            try:
                device = next(iter(text_encoder.parameters())).device
            except (StopIteration, AttributeError):
                # No parameters or empty, try to get device another way
                if hasattr(text_encoder, 'device'):
                    device = text_encoder.device
                else:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'

        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

        # Encode with text encoder
        try:
            # Build kwargs for encoder
            encode_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            # Add pixel values if available (for vision processing)
            if pixel_values is not None:
                encode_kwargs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                encode_kwargs["image_grid_thw"] = image_grid_thw

            # DiffSynth-style encoding
            if hasattr(text_encoder, 'encode_from_ids'):
                embeddings = text_encoder.encode_from_ids(**encode_kwargs)
            elif hasattr(text_encoder, 'model'):
                # Qwen2VLForConditionalGeneration - need to use the model's encoder
                with torch.no_grad():
                    try:
                        outputs = text_encoder.model(
                            **encode_kwargs,
                            output_hidden_states=True,
                            return_dict=True
                        )
                        embeddings = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
                    except Exception as model_err:
                        logger.error(f"Model forward pass failed: {model_err}")
                        logger.error(f"Available model attributes: {dir(text_encoder)}")
                        # Try getting the encoder embeddings directly
                        if hasattr(text_encoder, 'get_input_embeddings'):
                            embeddings = text_encoder.get_input_embeddings()(input_ids)
                        else:
                            raise
            else:
                # Standard transformer encoding
                outputs = text_encoder(
                    **encode_kwargs,
                    output_hidden_states=True
                )
                embeddings = outputs.last_hidden_state

            # Apply token dropping if specified
            if drop_tokens > 0 and embeddings.shape[1] > drop_tokens:
                embeddings = embeddings[:, drop_tokens:, :]
                logger.info(f"Dropped first {drop_tokens} tokens")

            # Convert to ComfyUI conditioning format
            # Include attention mask in conditioning for proper handling
            cond_dict = {}
            if attention_mask is not None:
                cond_dict["attention_mask"] = attention_mask
                if drop_tokens > 0 and attention_mask.shape[1] > drop_tokens:
                    cond_dict["attention_mask"] = attention_mask[:, drop_tokens:]

            conditioning = [[embeddings, cond_dict]]

            return (conditioning,)

        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            # Return empty conditioning with proper dimensions
            # Get hidden size from text encoder if possible
            hidden_size = 768  # Default
            if hasattr(text_encoder, 'config'):
                hidden_size = getattr(text_encoder.config, 'hidden_size', 768)
            return ([[torch.zeros((1, 77, hidden_size)), {}]],)


# Node registration
NODE_CLASS_MAPPINGS = {
    "QwenProcessorWrapper": QwenProcessorWrapper,
    "QwenProcessedToEmbedding": QwenProcessedToEmbedding,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenProcessorWrapper": "Qwen Processor (Wrapper)",
    "QwenProcessedToEmbedding": "Qwen Processed to Embedding (Wrapper)",
}