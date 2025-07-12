import logging
import json
import os

DEFAULT_CONFIG = {
  "vision_model": {
    "model_id": "Salesforce/blip-image-captioning-base",
    "max_length": 50,
    "num_beams": 4
  },
  "llm": {
    "model_id": "distilgpt2",
    "max_length": 250,
    "min_length": 50,
    "num_beams": 5,
    "temperature": 0.7,
    "no_repeat_ngram_size": 2
  },
  "tts": {
    "lang": "en"
  },
  "video_maker": {
    "output_fps": 24,
    "default_image_duration": 3
  },
  "logging": {
    "log_file": "app.log",
    "log_level": "INFO"
  }
}

def load_config():
    """Loads configuration from config.json, falling back to defaults."""
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                config_from_file = json.load(f)

            # A simple merge for nested dictionaries.
            merged_config = DEFAULT_CONFIG.copy()
            for key, value in config_from_file.items():
                if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
            return merged_config
        except (json.JSONDecodeError, IOError) as e:
            # Using print here because logger might not be set up yet or might be the source of the issue.
            print(f"Warning: Could not load or parse config.json: {e}. Using default config.")
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG

APP_CONFIG = load_config()

def setup_logger():
    """Sets up a logger that writes to console and a file."""
    logger = logging.getLogger("ManhwaAI")

    # Prevent adding multiple handlers if this function is called again.
    if logger.hasHandlers():
        return logger

    log_config = APP_CONFIG.get("logging", {})
    log_level_str = log_config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logger.setLevel(log_level)

    log_file = log_config.get("log_file", "app.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    try:
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        logger.error(f"Failed to set up file handler for logging to {log_file}: {e}")

    return logger

logger = setup_logger()
