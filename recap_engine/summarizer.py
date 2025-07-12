from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import os

try:
    from utils import logger, APP_CONFIG
except ImportError:
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
logger.info(f"LLM: Using device {DEVICE}")

TOKENIZER = None
MODEL = None

llm_config = APP_CONFIG.get("llm", {})
MODEL_ID = llm_config.get("model_id", "distilgpt2")

def load_model():
    """Loads the LLM model and tokenizer."""
    global TOKENIZER, MODEL
    if TOKENIZER is None or MODEL is None:
        logger.info(f"Loading LLM model {MODEL_ID} to {DEVICE}...")
        try:
            TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
            MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
            if MODEL.config.pad_token_id is None:
                MODEL.config.pad_token_id = MODEL.config.eos_token_id
            logger.info(f"LLM model {MODEL_ID} loaded successfully to {DEVICE}.")
        except Exception as e:
            logger.error(f"Error loading LLM model {MODEL_ID}: {e}", exc_info=True)
            TOKENIZER = None
            MODEL = None
    return TOKENIZER is not None and MODEL is not None

def generate_recap_script(captions_text):
    """
    Generates a recap script from captions using parameters from config.
    """
    if TOKENIZER is None or MODEL is None:
        if not load_model():
            return "Error: LLM model not loaded. Please check logs."

    try:
        max_len = llm_config.get("max_length", 200)
        min_len = llm_config.get("min_length", 50)
        num_bms = llm_config.get("num_beams", 5)
        temp = llm_config.get("temperature", 0.7)
        no_repeat_ngram = llm_config.get("no_repeat_ngram_size", 2)

        prompt = f"Based on the following sequence of events described by image captions, write an engaging recap or summary. \n\nCaptions:\n{captions_text}\n\nRecap:\n"

        max_prompt_tokens = 1024 - max_len
        inputs = TOKENIZER.encode(prompt, return_tensors='pt', truncation=True, max_length=max_prompt_tokens).to(DEVICE)

        if inputs.numel() == 0:
            logger.warning("LLM input encoding resulted in empty tensor.")
            return "Error: Could not encode prompt for LLM."

        attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=DEVICE)

        logger.debug(
            f"Generating recap with max_length={max_len}, min_length={min_len}, "
            f"num_beams={num_bms}, temperature={temp}, no_repeat_ngram_size={no_repeat_ngram}"
        )
        output_sequences = MODEL.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_length=inputs.shape[1] + max_len,
            min_length=inputs.shape[1] + min_len,
            temperature=temp,
            num_beams=num_bms,
            early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram,
            pad_token_id=MODEL.config.eos_token_id
        )

        if output_sequences.numel() == 0:
            logger.warning("LLM generated an empty sequence.")
            return "Error: LLM generated an empty sequence."

        # Re-encode the truncated prompt to ensure accurate slicing of the output
        truncated_prompt = TOKENIZER.decode(inputs[0], skip_special_tokens=True)
        recap_part = TOKENIZER.decode(output_sequences[0], skip_special_tokens=True)

        if recap_part.startswith(truncated_prompt):
            recap_part = recap_part[len(truncated_prompt):]

        recap_part = recap_part.strip()

        if not recap_part or len(recap_part) < 10:
            logger.warning(f"Recap extraction resulted in short/empty string. Fallback to decoding only generated tokens.")
            generated_tokens_only = output_sequences[0][inputs.shape[-1]:]
            recap_part = TOKENIZER.decode(generated_tokens_only, skip_special_tokens=True).strip()
            if not recap_part:
                 logger.error("Could not extract recap from LLM output even with fallback.")
                 return "Error: Could not extract recap from LLM output."

        logger.info(f"Generated recap script (first 100 chars): {recap_part[:100]}...")
        return recap_part

    except Exception as e:
        logger.error(f"Error generating recap script: {e}", exc_info=True)
        return f"Error generating recap script: {e}"

if __name__ == '__main__':
    logger.info("Running summarizer.py directly for testing...")
    if not load_model():
        logger.error("Cannot run example: Failed to load LLM model.")
        sys.exit(1)

    example_captions = """Image 1: a man with black hair is looking at a screen
Image 2: a man in a black suit is holding a sword
Image 3: a man in a black suit is holding a microphone"""

    logger.info(f"Generating recap for captions:\n{example_captions}\n")
    script = generate_recap_script(example_captions)
    logger.info(f"Generated Recap Script:\n{script}")
