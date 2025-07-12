from gtts import gTTS
import tempfile
import os
import sys

try:
    from utils import logger, APP_CONFIG
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from utils import logger, APP_CONFIG

tts_config = APP_CONFIG.get("tts", {})

def text_to_speech(text):
    """
    Converts text to an audio file using gTTS, with language from config.
    """
    if not text or not text.strip():
        logger.warning("TTS Error: Input text is empty.")
        return None

    lang = tts_config.get("lang", "en")

    try:
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()

        logger.debug(f"Generating TTS audio (lang='{lang}') for text: '{text[:70]}...'")
        tts = gTTS(text=text, lang=lang)
        tts.save(temp_audio_path)

        logger.info(f"TTS audio saved to temporary file: {temp_audio_path}")
        return temp_audio_path

    except Exception as e:
        logger.error(f"An error occurred during TTS generation: {e}", exc_info=True)
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception as remove_e:
                logger.error(f"Failed to remove temp audio file {temp_audio_path}: {remove_e}")
        return None

if __name__ == '__main__':
    logger.info("Running speaker.py directly for testing...")
    example_text = "Hello, this is a test of the text-to-speech engine."
    logger.info(f"Converting text to speech: '{example_text}'")

    audio_file = text_to_speech(example_text)

    if audio_file:
        logger.info(f"Successfully generated audio file: {audio_file}")
        try:
            logger.info("Cleaning up temporary audio file from example.")
            os.remove(audio_file)
        except OSError as e:
            logger.error(f"Error removing temporary file in example: {e}")
    else:
        logger.error("Failed to generate audio file in example.")
