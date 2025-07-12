from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import sys
import os

# Add project root to sys.path to allow importing utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import logger, APP_CONFIG

# Determine device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
logger.info(f"Vision Model: Using device {DEVICE}")

PROCESSOR = None
MODEL = None

# Load configuration for vision model
vision_config = APP_CONFIG.get("vision_model", {})
MODEL_ID = vision_config.get("model_id", "Salesforce/blip-image-captioning-base") # Default if not in config

def load_model():
    """Loads the BLIP image captioning model and processor."""
    global PROCESSOR, MODEL
    if PROCESSOR is None or MODEL is None:
        logger.info(f"Loading vision model {MODEL_ID} to {DEVICE}...")
        try:
            PROCESSOR = BlipProcessor.from_pretrained(MODEL_ID)
            MODEL = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
            logger.info(f"Vision model {MODEL_ID} loaded successfully to {DEVICE}.")
        except Exception as e:
            logger.error(f"Error loading vision model {MODEL_ID}: {e}", exc_info=True)
            PROCESSOR = None
            MODEL = None
    return PROCESSOR is not None and MODEL is not None

def generate_caption(image_path_or_pil_image):
    """
    Generates a caption for a given image using parameters from config.

    Args:
        image_path_or_pil_image (str or PIL.Image): Path to the image file or a PIL Image object.

    Returns:
        str: The generated caption, or an error message if captioning fails.
    """
    if PROCESSOR is None or MODEL is None:
        if not load_model(): # load_model already logs errors
            return "Error: Vision model not loaded. Please check logs."

    try:
        if isinstance(image_path_or_pil_image, str):
            raw_image = Image.open(image_path_or_pil_image).convert('RGB')
        elif isinstance(image_path_or_pil_image, Image.Image):
            raw_image = image_path_or_pil_image.convert('RGB')
        else:
            logger.warning("Invalid image input type for captioning.")
            return "Error: Invalid image input. Must be a file path or PIL Image."

        # Get generation parameters from config
        max_len = vision_config.get("max_length", 50)
        num_bms = vision_config.get("num_beams", 4)

        inputs = PROCESSOR(raw_image, return_tensors="pt").to(DEVICE)

        logger.debug(f"Generating caption with max_length={max_len}, num_beams={num_bms}")
        out = MODEL.generate(**inputs, max_length=max_len, num_beams=num_bms)
        caption = PROCESSOR.decode(out[0], skip_special_tokens=True)
        logger.info(f"Generated caption: {caption}")
        return caption
    except Exception as e:
        logger.error(f"Error generating caption: {e}", exc_info=True)
        return f"Error generating caption: {e}"

if __name__ == '__main__':
    # Example usage:
    # This will use the logger and config defined in utils.py

    # Ensure models are loaded for the test
    if not load_model():
        logger.error("Cannot run example: Failed to load vision model.")
        sys.exit(1)

    example_image_path = None
    if len(sys.argv) > 1:
        example_image_path = sys.argv[1]
        if not os.path.exists(example_image_path):
            logger.warning(f"Provided image path {example_image_path} does not exist. Using dummy image.")
            example_image_path = None # Fallback to dummy

    if example_image_path:
         logger.info(f"Attempting to caption image: {example_image_path}")
         test_image_input = example_image_path
    else:
        logger.info("No image path provided or path invalid. Creating a dummy image for testing.")
        try:
            dummy_image = Image.new('RGB', (256, 256), color = 'white')
            test_image_input = dummy_image # Pass the PIL image directly
        except Exception as e:
            logger.error(f"Failed to create dummy image: {e}", exc_info=True)
            sys.exit(1)

    caption_result = generate_caption(test_image_input)
    logger.info(f"Example Generated Caption: {caption_result}")
