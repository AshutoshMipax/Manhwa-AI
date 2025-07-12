import gradio as gr
from PIL import Image
import os
import tempfile
import zipfile
import torch
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip

# --- 1. Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
        # Optionally add a FileHandler here if you want to log to a file
        # logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("ManhwaAI")

# --- 2. Model and App Configuration ---
# Using a simple dictionary for configuration to keep it self-contained.
APP_CONFIG = {
  "vision_model": { "model_id": "Salesforce/blip-image-captioning-base", "max_length": 50, "num_beams": 4 },
  "llm": { "model_id": "distilgpt2", "max_length": 250, "min_length": 50, "num_beams": 5, "temperature": 0.7, "no_repeat_ngram_size": 2 },
  "tts": { "lang": "en" },
  "video_maker": { "output_fps": 24, "default_image_duration": 3 }
}

# --- 3. Global Variables & Model Loading ---
# Determine device for PyTorch
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
logger.info(f"Using device: {DEVICE}")

# Vision Model Globals
vision_processor = None
vision_model = None
vision_model_loaded = False

# LLM Globals
llm_tokenizer = None
llm_model = None
llm_model_loaded = False

def load_vision_model():
    global vision_processor, vision_model, vision_model_loaded
    if vision_model_loaded: return
    try:
        model_id = APP_CONFIG["vision_model"]["model_id"]
        logger.info(f"Loading vision model: {model_id} to {DEVICE}...")
        vision_processor = BlipProcessor.from_pretrained(model_id)
        vision_model = BlipForConditionalGeneration.from_pretrained(model_id).to(DEVICE)
        vision_model_loaded = True
        logger.info("Vision model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading vision model: {e}", exc_info=True)

def load_llm_model():
    global llm_tokenizer, llm_model, llm_model_loaded
    if llm_model_loaded: return
    try:
        model_id = APP_CONFIG["llm"]["model_id"]
        logger.info(f"Loading LLM: {model_id} to {DEVICE}...")
        llm_tokenizer = AutoTokenizer.from_pretrained(model_id)
        llm_model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
        if llm_model.config.pad_token_id is None:
            llm_model.config.pad_token_id = llm_model.config.eos_token_id
        llm_model_loaded = True
        logger.info("LLM loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading LLM: {e}", exc_info=True)

def load_all_models():
    """Wrapper to load all models on app startup."""
    load_vision_model()
    load_llm_model()

# --- 4. Core Application Logic ---

def generate_caption(pil_image):
    if not vision_model_loaded: return "Error: Vision model not loaded."
    try:
        config = APP_CONFIG["vision_model"]
        inputs = vision_processor(pil_image.convert('RGB'), return_tensors="pt").to(DEVICE)
        out = vision_model.generate(**inputs, max_length=config["max_length"], num_beams=config["num_beams"])
        return vision_processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Caption generation failed: {e}", exc_info=True)
        return "Error during caption generation."

def generate_recap_script(captions_text):
    if not llm_model_loaded: return "Error: LLM not loaded."
    try:
        config = APP_CONFIG["llm"]
        prompt = f"Based on these events, write an engaging summary:\n\n{captions_text}\n\nSummary:\n"
        inputs = llm_tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=1024 - config["max_length"]).to(DEVICE)

        output_sequences = llm_model.generate(
            input_ids=inputs,
            max_length=inputs.shape[1] + config["max_length"],
            min_length=inputs.shape[1] + config["min_length"],
            temperature=config["temperature"],
            num_beams=config["num_beams"],
            early_stopping=True,
            no_repeat_ngram_size=config["no_repeat_ngram_size"],
            pad_token_id=llm_model.config.eos_token_id
        )

        full_text = llm_tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        # Simple slicing to get only the generated part
        summary = full_text[len(prompt):].strip()
        return summary
    except Exception as e:
        logger.error(f"Recap script generation failed: {e}", exc_info=True)
        return "Error during script generation."

def text_to_speech(text):
    try:
        lang = APP_CONFIG["tts"]["lang"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            tts = gTTS(text=text, lang=lang)
            tts.save(temp_audio_file.name)
            return temp_audio_file.name
    except Exception as e:
        logger.error(f"TTS generation failed: {e}", exc_info=True)
        return None

def create_video(image_paths, audio_path):
    if not image_paths or not audio_path: return None
    try:
        config = APP_CONFIG["video_maker"]
        logger.info(f"Creating video with {len(image_paths)} images.")
        audio_clip = AudioFileClip(audio_path)
        duration_per_image = audio_clip.duration / len(image_paths)

        # Create video clips from images
        clips = [ImageClip(path, duration=duration_per_image) for path in image_paths]

        # Set video size based on the first image
        video_size = clips[0].size
        # Resize all clips to the same size
        clips = [c.resize(height=video_size[1], width=video_size[0]) for c in clips]

        final_video = concatenate_videoclips(clips, method="compose").set_audio(audio_clip)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            final_video.write_videofile(
                temp_video_file.name,
                fps=config["output_fps"],
                codec="libx264",
                audio_codec="aac",
                logger=None # Suppress moviepy's verbose logging
            )
            return temp_video_file.name
    except Exception as e:
        logger.error(f"Video creation failed: {e}", exc_info=True)
        return None
    finally:
        # Clean up moviepy resources if they exist
        if 'audio_clip' in locals() and audio_clip: audio_clip.close()
        if 'final_video' in locals() and final_video: final_video.close()
        if 'clips' in locals():
            for clip in clips:
                clip.close()

# --- 5. Gradio Interface and Backend Functions ---
def get_initial_state():
    return {"image_paths": [], "audio_path": None}

def process_files_interface(files, current_state):
    if not vision_model_loaded:
        gr.Warning("Vision model isn't loaded. Please wait or restart.")
        return current_state, "", [], "0 images for video"
    if not files:
        return current_state, "No files uploaded.", [], "0 images for video"

    captions_log = ""
    image_previews = []
    image_paths_for_video = []

    for file in files:
        if zipfile.is_zipfile(file.name):
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(file.name, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                for f_name in sorted(os.listdir(temp_dir)):
                    if f_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(temp_dir, f_name)
                        pil_image = Image.open(img_path)
                        image_previews.append(pil_image)
                        captions_log += f"{f_name}: {generate_caption(pil_image)}\n"
        elif file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            pil_image = Image.open(file.name)
            image_previews.append(pil_image)
            image_paths_for_video.append(file.name)
            captions_log += f"{os.path.basename(file.name)}: {generate_caption(pil_image)}\n"

    current_state["image_paths"] = image_paths_for_video
    status_text = f"{len(image_paths_for_video)} images stored for video."
    return current_state, captions_log, image_previews, status_text

def tts_interface(script, current_state):
    if not script:
        gr.Warning("Script is empty. Cannot generate audio.")
        return None, current_state
    audio_path = text_to_speech(script)
    if audio_path:
        current_state["audio_path"] = audio_path
    return audio_path, current_state

def video_interface(current_state):
    image_paths = current_state.get("image_paths", [])
    audio_path = current_state.get("audio_path")
    if not image_paths or not audio_path:
        gr.Warning("Images for video or audio are missing. Please complete all steps.")
        return None
    gr.Info("Video generation started. This may take a moment...")
    video_path = create_video(image_paths, audio_path)
    if video_path:
        gr.Info("Video generation complete!")
    else:
        gr.Error("Video generation failed.")
    return video_path

with gr.Blocks() as iface:
    app_state = gr.State(value=get_initial_state())
    gr.Markdown("# Manhwa-to-Recap Video Generator")
    with gr.Tabs():
        with gr.TabItem("Step 1: Get Captions"):
            image_input = gr.File(label="Upload Images or ZIP", file_count="multiple")
            caption_button = gr.Button("Generate Captions", variant="primary")
            caption_output = gr.Textbox(label="Captions Log", lines=15)
            gallery_output = gr.Gallery(label="Image Previews", columns=5, height="auto")
            video_status = gr.Textbox(label="Status", value="0 images stored for video", interactive=False)
        with gr.TabItem("Step 2: Generate Script & Voice"):
            with gr.Row():
                with gr.Column(scale=2):
                    script_button = gr.Button("Generate Script from Captions", variant="primary")
                    script_output = gr.Textbox(label="Recap Script", lines=10, interactive=True)
                with gr.Column(scale=1):
                    tts_button = gr.Button("Generate Voiceover", variant="primary")
                    audio_output = gr.Audio(label="Voiceover Audio", type="filepath")
        with gr.TabItem("Step 3: Create Video"):
            video_button = gr.Button("Generate Final Video", variant="primary")
            video_output = gr.Video(label="Final Recap Video")

    # Wiring
    caption_button.click(process_files_interface, [image_input, app_state], [app_state, caption_output, gallery_output, video_status])
    script_button.click(generate_recap_script, [caption_output], [script_output])
    tts_button.click(tts_interface, [script_output, app_state], [audio_output, app_state])
    video_button.click(video_interface, [app_state], [video_output])

    iface.load(load_all_models)

if __name__ == "__main__":
    iface.launch()
