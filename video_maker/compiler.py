from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
import os
import tempfile
from PIL import Image
import sys

try:
    from utils import logger, APP_CONFIG
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from utils import logger, APP_CONFIG

video_config = APP_CONFIG.get("video_maker", {})

def create_video_from_images_and_audio(image_paths, audio_path):
    """
    Creates a video from images and audio using parameters from config.
    """
    output_fps = video_config.get("output_fps", 24)
    default_image_duration = video_config.get("default_image_duration", 3)

    if not image_paths:
        logger.error("Video Error: No image paths provided.")
        return None
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"Video Error: Audio file not found or path is invalid ('{audio_path}').")
        return None

    video_clips = []
    valid_image_paths = []

    video_size = None
    # Determine video size from the first valid image
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                video_size = img.size
                valid_image_paths.append(img_path)
        except Exception as e:
            logger.warning(f"Skipping image {img_path} due to error: {e}")
            continue

    # We need at least one valid image to proceed
    if not video_size or not valid_image_paths:
        logger.error("Video Error: No valid images found to determine video size.")
        return None

    # Use the first valid image's size for all frames
    first_valid_image_path = valid_image_paths[0]
    with Image.open(first_valid_image_path) as img:
        video_size = img.size

    audio_clip = None
    final_video_clip = None
    try:
        audio_clip = AudioFileClip(audio_path)
        total_audio_duration = audio_clip.duration

        num_images = len(valid_image_paths)
        duration_per_image = total_audio_duration / num_images if num_images > 0 and total_audio_duration > 0 else default_image_duration

        if duration_per_image <= 0.01:
            logger.warning(f"Calculated duration per image is too small ({duration_per_image}). Using default: {default_image_duration}s.")
            duration_per_image = default_image_duration

        logger.info(f"Video params: {num_images} images, {total_audio_duration:.2f}s audio, {duration_per_image:.2f}s/image, {output_fps} FPS.")

        for img_path in valid_image_paths:
            try:
                img_clip = ImageClip(img_path, duration=duration_per_image).set_fps(output_fps)
                if img_clip.size != video_size:
                    logger.info(f"Resizing image {img_path} from {img_clip.size} to {video_size}")
                    img_clip = img_clip.resize(width=video_size[0], height=video_size[1])
                video_clips.append(img_clip)
            except Exception as e:
                logger.error(f"Error processing image {img_path} for video: {e}. Skipping.", exc_info=True)
                continue

        if not video_clips:
            logger.error("Video Error: No video clips were created from images.")
            return None

        final_video_clip = concatenate_videoclips(video_clips, method="compose")
        final_video_clip = final_video_clip.set_audio(audio_clip)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            output_video_path = temp_video_file.name

        logger.info(f"Writing final video to: {output_video_path}")
        final_video_clip.write_videofile(
            output_video_path,
            fps=output_fps,
            codec="libx264",
            audio_codec="aac",
            logger=None,
            ffmpeg_params=["-pix_fmt", "yuv420p"]
        )
        logger.info("Video written successfully.")
        return output_video_path

    except Exception as e:
        logger.error(f"An error occurred during video compilation: {e}", exc_info=True)
        return None
    finally:
        # Gracefully close all opened clips
        for clip in video_clips:
            if clip: clip.close()
        if audio_clip: audio_clip.close()
        if final_video_clip: final_video_clip.close()

if __name__ == '__main__':
    logger.info("Running compiler.py directly for testing...")
    # This example is complex to run standalone as it needs images and audio.
    # The main test will be through the Gradio app.
    logger.info("Standalone test for compiler.py is best performed via the main app.")
