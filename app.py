import gradio as gr
from PIL import Image
import os
import tempfile
import zipfile

# Assuming captioner.py is in vision_models/ and load_model/generate_caption are importable
from vision_models.captioner import load_model as load_vision_model, generate_caption

from utils import logger, APP_CONFIG # Import logger and config

# --- State Management ---
# Using gr.State to hold file paths between stages for a cleaner approach than global variables
# A dict to hold all necessary paths
app_state = gr.State(value={"image_paths": [], "audio_path": None})

# --- Functions ---

def process_uploaded_images(image_files_or_zip, current_state):
    """
    Processes uploaded files, generates captions, and updates app state with image paths.
    NOTE: For video generation from ZIPs, this implementation has a known limitation.
    Files from a ZIP are extracted to a temporary directory that is deleted after this
    function runs. To make them available for video, they would need to be copied to a
    more persistent temporary location. This implementation currently only stores paths
    from direct, non-ZIP image uploads for the video stage.
    """
    if not vision_model_loaded:
        load_all_models() # Ensure models are loaded
        if not vision_model_loaded:
            return "Error: Vision model could not be loaded.", None, None, current_state

    if image_files_or_zip is None:
        return "No image or ZIP file uploaded.", None, None, current_state

    captions_output = ""
    image_previews = []
    image_paths_for_video = []

    files_to_process = image_files_or_zip if isinstance(image_files_or_zip, list) else [image_files_or_zip]

    for file_obj in files_to_process:
        file_path = file_obj.name
        
        if zipfile.is_zipfile(file_path):
            captions_output += f"Processing images from ZIP: {os.path.basename(file_path)}...\n"
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                # Note: These extracted paths are temporary and will NOT be available for video.
                zip_image_paths = [os.path.join(root, f) for root, _, files in os.walk(temp_dir) for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                for i, img_path in enumerate(zip_image_paths):
                    try:
                        pil_image = Image.open(img_path).convert("RGB")
                        image_previews.append(pil_image)
                        caption = generate_caption(pil_image)
                        captions_output += f"  - {os.path.basename(img_path)}: {caption}\n"
                    except Exception as e:
                        captions_output += f"  - Error processing {os.path.basename(img_path)}: {e}\n"
            if zip_image_paths: # Only add warning if images were actually found in ZIP
                captions_output += (
                    f"\nINFO: Processed {len(zip_image_paths)} images from {os.path.basename(file_path)}. "
                    "For video generation, please upload these images individually if desired.\n\n"
                )
            else:
                captions_output += f"INFO: No images found or processed within {os.path.basename(file_path)}.\n\n"
        
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_paths_for_video.append(file_path) # Store path for video
            try:
                pil_image = Image.open(file_path).convert("RGB")
                image_previews.append(pil_image)
                caption = generate_caption(pil_image)
                captions_output += f"Image '{os.path.basename(file_path)}':\n{caption}\n\n"
            except Exception as e:
                captions_output += f"Error processing image {os.path.basename(file_path)}: {e}\n\n"
                image_previews.append(None)
        else:
            captions_output += f"Skipping non-image/non-ZIP file: {os.path.basename(file_path)}\n"

    current_state["image_paths"] = image_paths_for_video
    
    max_previews = 10
    final_previews = [p for p in image_previews if p is not None]
    if len(final_previews) > max_previews:
        captions_output += f"\n(Showing first {max_previews} image previews out of {len(final_previews)})"
        final_previews = final_previews[:max_previews]
        
    return captions_output, final_previews, gr.update(value=f"{len(image_paths_for_video)} images stored for video."), current_state

# This function was missing, re-adding it.
def call_generate_recap(captions, custom_prompt=""):
    if not llm_model_loaded:
        logger.error("LLM model not loaded, cannot generate recap.")
        gr.Warning("Error: LLM model not loaded. Please ensure it loaded correctly on startup.")
        return "Error: LLM model not loaded."
    if not captions or captions.strip() == "No image or ZIP file uploaded." or captions.strip().startswith("Error:"):
        logger.warning("Cannot generate script without valid captions.")
        gr.Warning("Cannot generate script without valid captions. Please upload images and generate captions first.")
        return "Cannot generate script without valid captions."

    full_prompt_for_llm = captions
    if custom_prompt and custom_prompt.strip():
        logger.info(f"Using custom prompt for LLM: {custom_prompt}")
        full_prompt_for_llm = f"{custom_prompt}\n\nBased on these captions:\n{captions}"
    
    logger.info("Calling LLM to generate recap script...")
    script = generate_recap_script(full_prompt_for_llm)
    if script.startswith("Error:"):
        gr.Error(script)
    else:
        gr.Info("Recap script generated.")
    return script

def call_text_to_speech(script_text, current_state):
    if not script_text or not script_text.strip():
        return None, current_state

    audio_file_path = text_to_speech(script_text)

    if audio_file_path:
        current_state["audio_path"] = audio_file_path
        return audio_file_path, current_state
    else:
        gr.Warning("TTS generation failed. Check console for errors.")
        return None, current_state

def call_create_video(current_state):
    image_paths = current_state.get("image_paths", [])
    audio_path = current_state.get("audio_path")

    status_message = ""
    if not image_paths:
        status_message += "No images available for video. Please upload individual image files and generate captions first. "
    if not audio_path:
        status_message += "No audio available for video. Please generate the voiceover first."

    if status_message:
        gr.Warning(status_message)
        return None

    gr.Info("Video generation started... This may take some time.")
    video_path = create_video_from_images_and_audio(image_paths, audio_path)

    if video_path:
        gr.Info("Video generation complete!")
        return video_path
    else:
        gr.Error("Video generation failed. Check console for errors.")
        return None

# --- Gradio Interface Definition ---
with gr.Blocks() as iface:
    gr.Markdown("# Manhwa-to-Recap Video Generator")
    
    # --- Model Loading and State ---
    from recap_engine.summarizer import load_model as load_llm_model, generate_recap_script
    from tts_engine.speaker import text_to_speech
    from video_maker.compiler import create_video_from_images_and_audio

    vision_model_loaded = False
    llm_model_loaded = False

    def load_all_models():
        # ... (load_all_models function remains the same)
        global vision_model_loaded, llm_model_loaded
        if not vision_model_loaded:
            logger.info("Attempting to load vision model via app.py...")
            if load_vision_model(): # This function now uses logger internally
                vision_model_loaded = True
                # logger.info("Vision model has been successfully loaded (triggered by app.py).") # Already logged in captioner.py
            else:
                logger.error("Failed to load vision model (triggered by app.py).") # Already logged in captioner.py
        if not llm_model_loaded:
            logger.info("Attempting to load LLM model via app.py...")
            if load_llm_model(): # This function now uses logger internally
                llm_model_loaded = True
                # logger.info("LLM model has been successfully loaded (triggered by app.py).") # Already logged in summarizer.py
            else:
                logger.error("Failed to load LLM model (triggered by app.py).") # Already logged in summarizer.py
        
        # The individual load_model functions in captioner.py and summarizer.py now do their own detailed logging.
        # So, load_all_models in app.py mainly orchestrates and can rely on those logs.
        # We can add a summary log here.
        if vision_model_loaded and llm_model_loaded:
            logger.info("All models checked/loaded by app.py.")
        elif vision_model_loaded:
            logger.warning("Vision model loaded, but LLM failed (checked by app.py).")
        elif llm_model_loaded:
            logger.warning("LLM model loaded, but Vision failed (checked by app.py).")
        else:
            logger.error("Critical models (Vision and/or LLM) failed to load (checked by app.py).")

    iface.load = load_all_models

    with gr.Tabs():
        with gr.TabItem("Step 1: Get Captions"):
            gr.Markdown("Upload your manhwa panels as individual images or a single ZIP file. Captions will be generated for each image.")
            with gr.Row():
                image_input = gr.File(
                    label="Upload Manhwa Images or ZIP",
                    file_count="multiple",
                    type="file"
                )
            caption_button = gr.Button("Generate Captions", variant="primary")
            gr.Markdown("### Generated Captions")
            caption_output_text = gr.Textbox(label="Captions Log", lines=15, interactive=False)
            gr.Markdown("### Image Previews")
            image_output_gallery = gr.Gallery(label="Image Previews", show_label=False, elem_id="gallery", columns=[5], object_fit="contain", height="auto")

            video_image_status = gr.Textbox(label="Status for Video Generator", value="0 images stored for video.", interactive=False)

        with gr.TabItem("Step 2: Generate Script & Voice"):
            gr.Markdown("Generate a recap script from the captions, then convert that script into a voiceover.")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Generate Recap Script")
                    recap_prompt_input = gr.Textbox(label="Optional: Add custom instructions for the recap style", placeholder="e.g., 'Make it dramatic'", lines=2)
                    generate_script_button = gr.Button("Generate Script", variant="primary")
                    recap_script_output = gr.Textbox(label="Recap Script", lines=10, interactive=True)
                with gr.Column():
                    gr.Markdown("### Generate Voiceover")
                    generate_audio_button = gr.Button("Generate Voiceover", variant="primary")
                    audio_output = gr.Audio(label="Voiceover Output", type="filepath")

        with gr.TabItem("Step 3: Create & Download Video"):
            gr.Markdown("Combine the uploaded images and the generated voiceover into a final video file.")
            generate_video_button = gr.Button("Generate Final Video", variant="primary")
            video_output = gr.Video(label="Generated Video Output")

    # --- Wire up components ---
    caption_button.click(
        fn=process_uploaded_images,
        inputs=[image_input, app_state],
        outputs=[caption_output_text, image_output_gallery, video_image_status, app_state]
    )

    generate_script_button.click(
        fn=call_generate_recap,
        inputs=[caption_output_text, recap_prompt_input],
        outputs=[recap_script_output]
    )

    generate_audio_button.click(
        fn=call_text_to_speech,
        inputs=[recap_script_output, app_state],
        outputs=[audio_output, app_state]
    )

    generate_video_button.click(
        fn=call_create_video,
        inputs=[app_state],
        outputs=[video_output]
    )

if __name__ == "__main__":
    logger.info("Starting Manhwa-AI Gradio application...")
    iface.launch()
