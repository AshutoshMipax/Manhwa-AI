from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import sys
import os

# This check allows the module to be imported by app.py in the root
# and also allows running the module directly for testing.
try:
    from utils import logger, APP_CONFIG
except ImportError:
    # If running directly, the path to the root needs to be added.
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
MODEL_ID = vision_config.get("model_id", "Salesforce/blip-image-captioning-base")

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
    """
    if PROCESSOR is None or MODEL is None:
        if not load_model():
            return "Error: Vision model not loaded. Please check logs."

    try:
        if isinstance(image_path_or_pil_image, str):
            raw_image = Image.open(image_path_or_pil_image).convert('RGB')
        elif isinstance(image_path_or_pil_image, Image.Image):
            raw_image = image_path_or_pil_image.convert('RGB')
        else:
            logger.warning("Invalid image input type for captioning.")
            return "Error: Invalid image input. Must be a file path or PIL Image."

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
    logger.info("Running captioner.py directly for testing...")
    if not load_model():
        logger.error("Cannot run example: Failed to load vision model.")
        sys.exit(1)

    logger.info("Creating a dummy image for testing.")
    try:
        dummy_image = Image.new('RGB', (256, 256), color = 'white')
        test_image_input = dummy_image
    except Exception as e:
        logger.error(f"Failed to create dummy image: {e}", exc_info=True)
        sys.exit(1)

    caption_result = generate_caption(test_image_input)
    logger.info(f"Example Generated Caption: {caption_result}")
