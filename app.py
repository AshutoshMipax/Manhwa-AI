import gradio as gr
from PIL import Image
import os
import tempfile
import zipfile

from utils import logger, APP_CONFIG
from vision_models.captioner import load_model as load_vision_model, generate_caption
from recap_engine.summarizer import load_model as load_llm_model, generate_recap_script
from tts_engine.speaker import text_to_speech
from video_maker.compiler import create_video_from_images_and_audio

# --- App State Management ---
def get_initial_state():
    return {
        "image_paths": [],
        "audio_path": None,
        "image_previews": [],
    }

# --- Model Loading ---
vision_model_loaded = False
llm_model_loaded = False

def load_all_models():
    """Loads all necessary models for the application."""
    global vision_model_loaded, llm_model_loaded
    if not vision_model_loaded:
        logger.info("Attempting to load vision model...")
        if load_vision_model():
            vision_model_loaded = True
        else:
            logger.error("Vision model loading failed.")

    if not llm_model_loaded:
        logger.info("Attempting to load LLM...")
        if load_llm_model():
            llm_model_loaded = True
        else:
            logger.error("LLM loading failed.")

# --- Backend Functions ---
def process_uploaded_files(files, current_state):
    if not vision_model_loaded:
        gr.Warning("Vision model is not loaded. Please try restarting.")
        return current_state, "", [], "0 images stored for video."
    if files is None:
        return current_state, "No files uploaded.", [], "0 images stored for video."

    captions_log, image_paths_for_video = "", []
    current_state["image_previews"] = []
    captions_log, image_paths_for_video = "", []
    current_state["image_previews"] = []
    allowed_extensions = ('.png', '.jpg', '.jpeg', '.webp')

    for file_obj in files:
        file_path = file_obj.name
        if zipfile.is_zipfile(file_path):
            captions_log += f"Processing images from ZIP: {os.path.basename(file_path)}...\n"
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                zip_image_paths = sorted([os.path.join(root, f) for root, _, fs in os.walk(temp_dir) for f in fs if f.lower().endswith(allowed_extensions)])
                for img_path in zip_image_paths:
                    try:
                        pil_image = Image.open(img_path).convert("RGB")
                        current_state["image_previews"].append(pil_image)
                        captions_log += f"  - {os.path.basename(img_path)}: {generate_caption(pil_image)}\n"
                    except Exception as e:
                        logger.error(f"Error in ZIP image {os.path.basename(img_path)}: {e}", exc_info=True)
            captions_log += "\nINFO: For video generation, please upload key images individually.\n\n"
        elif file_path.lower().endswith(allowed_extensions):
            image_paths_for_video.append(file_path)
            try:
                pil_image = Image.open(file_path).convert("RGB")
                current_state["image_previews"].append(pil_image)
                captions_log += f"Image '{os.path.basename(file_path)}':\n{generate_caption(pil_image)}\n\n"
            except Exception as e:
                logger.error(f"Error processing image {os.path.basename(file_path)}: {e}", exc_info=True)
    current_state["image_paths"] = image_paths_for_video
    max_previews = 15
    final_previews = current_state["image_previews"]
    if len(final_previews) > max_previews:
        final_previews = final_previews[:max_previews]
    status_text = f"{len(image_paths_for_video)} images stored for video."
    gr.Info(f"Captioning complete. {status_text}")
    return current_state, captions_log, final_previews, gr.update(value=status_text)

def call_generate_recap(captions):
    if not llm_model_loaded:
        gr.Warning("LLM model is not loaded. Please restart.")
        return "Error: LLM model not loaded."
    if not captions or "error" in captions.lower() or not captions.strip():
        gr.Warning("Cannot generate script from empty/error captions.")
        return "Error: No valid captions provided."
    logger.info("Calling LLM to generate recap script...")
    script = generate_recap_script(captions)
    if not script.startswith("Error:"): gr.Info("Recap script generated.")
    else: gr.Error(script)
    return script

def call_text_to_speech_interface(script_text, current_state):
    if not script_text or script_text.startswith("Error:"):
        gr.Warning("Cannot generate audio from empty/error script.")
        current_state["audio_path"] = None
        return None, current_state
    logger.info("Calling TTS engine to generate audio...")
    audio_file_path = text_to_speech(script_text)
    if audio_file_path:
        gr.Info("Voiceover generated successfully.")
        current_state["audio_path"] = audio_file_path
    else:
        gr.Error("TTS generation failed. Check console for errors.")
        current_state["audio_path"] = None
    return audio_file_path, current_state

def call_create_video_interface(current_state):
    image_paths = current_state.get("image_paths", [])
    audio_path = current_state.get("audio_path")
    if not image_paths or not audio_path:
        gr.Warning("Missing images or audio for video generation. Please complete previous steps.")
        return None
    gr.Info("Video generation started... This may take a while.")
    video_path = create_video_from_images_and_audio(image_paths, audio_path)
    if video_path:
        gr.Info("Video generation complete!")
    else:
        gr.Error("Video generation failed. Check console for errors.")
    return video_path

# --- Gradio UI ---
with gr.Blocks() as iface:
    logger.info("Creating Gradio UI...")
    app_state = gr.State(value=get_initial_state())

    gr.Markdown("# Manhwa-to-Recap Video Generator")
    
    with gr.Tabs():
        with gr.TabItem("Step 1: Get Captions"):
            gr.Markdown("Upload manhwa panels as individual images or a single ZIP file. Captions will be generated for each image.")
            image_input = gr.File(label="Upload Manhwa Images or ZIP", file_count="multiple", type="file")
            caption_button = gr.Button("Generate Captions", variant="primary")
            caption_output_text = gr.Textbox(label="Captions Log", lines=15, interactive=False)
            image_output_gallery = gr.Gallery(label="Image Previews", show_label=False, elem_id="gallery", columns=5, object_fit="contain", height="auto")
            video_image_status = gr.Textbox(label="Status for Video Generator", value="0 images stored for video.", interactive=False)

        with gr.TabItem("Step 2: Generate Script & Voice"):
            gr.Markdown("Generate a recap script from captions, then convert it into a voiceover.")
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Generate Recap Script")
                    generate_script_button = gr.Button("Generate Script from Captions", variant="primary")
                    recap_script_output = gr.Textbox(label="Recap Script", lines=10, interactive=True)
                with gr.Column(scale=1):
                    gr.Markdown("### Generate Voiceover")
                    generate_audio_button = gr.Button("Generate Voiceover from Script", variant="primary")
                    audio_output = gr.Audio(label="Voiceover Output", type="filepath")

        with gr.TabItem("Step 3: Create & Download Video"):
            gr.Markdown("Combine the uploaded images (from Step 1) and the generated voiceover (from Step 2) into a final video file.")
            generate_video_button = gr.Button("Generate Final Video", variant="primary")
            video_output = gr.Video(label="Generated Video Output")

    # --- Component Wiring ---
    caption_button.click(
        fn=process_uploaded_files,
        inputs=[image_input, app_state],
        outputs=[app_state, caption_output_text, image_output_gallery, video_image_status]
    )
    generate_script_button.click(
        fn=call_generate_recap,
        inputs=[caption_output_text],
        outputs=[recap_script_output]
    )
    generate_audio_button.click(
        fn=call_text_to_speech_interface,
        inputs=[recap_script_output, app_state],
        outputs=[audio_output, app_state]
    )
    generate_video_button.click(
        fn=call_create_video_interface,
        inputs=[app_state],
        outputs=[video_output]
    )

    iface.load(fn=load_all_models)

if __name__ == "__main__":
    logger.info("Starting Manhwa-AI Gradio application...")
    iface.launch()
