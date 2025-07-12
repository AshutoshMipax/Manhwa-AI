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

            # Simple merge: file values override defaults, no deep merge for nested dicts
            # For more complex configs, a deep merge would be better.
            merged_config = DEFAULT_CONFIG.copy()
            for key, value in config_from_file.items():
                if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
            return merged_config
        except json.JSONDecodeError as e:
            print(f"Error decoding config.json: {e}. Using default config.")
            return DEFAULT_CONFIG
        except Exception as e:
            print(f"Error loading config.json: {e}. Using default config.")
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG

APP_CONFIG = load_config()

def setup_logger():
    """Sets up a logger that writes to console and a file."""
    logger = logging.getLogger("ManhwaAI")

    # Prevent multiple handlers if function is called again
    if logger.hasHandlers():
        logger.handlers.clear()

    log_level_str = APP_CONFIG.get("logging", {}).get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    log_file = APP_CONFIG.get("logging", {}).get("log_file", "app.log")

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    try:
        fh = logging.FileHandler(log_file, mode='a') # Append mode
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        logger.error(f"Failed to set up file handler for logging: {e}")
        # Continue with console logging

    return logger

logger = setup_logger()

if __name__ == "__main__":
    # Example usage of the logger and config
    logger.info("Logger setup complete.")
    logger.debug("This is a debug message.") # Won't show if level is INFO
    logger.warning("This is a warning.")
    logger.error("This is an error.")

    vision_model_config = APP_CONFIG.get("vision_model", {})
    logger.info(f"Vision model ID from config: {vision_model_config.get('model_id')}")

    # Test loading a non-existent key
    non_existent_config = APP_CONFIG.get("non_existent_key", {"default_value": "test"})
    logger.info(f"Non-existent config test: {non_existent_config}")
