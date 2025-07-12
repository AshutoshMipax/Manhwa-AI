from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
import os
import tempfile
from PIL import Image
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import logger, APP_CONFIG

# Load configuration for video maker
video_config = APP_CONFIG.get("video_maker", {})

def create_video_from_images_and_audio(image_paths, audio_path):
    """
    Creates a video from a sequence of images and an audio file using parameters from config.

    Args:
        image_paths (list): A list of paths to the image files.
        audio_path (str): Path to the audio file (e.g., MP3 from TTS).

    Returns:
        str: Path to the generated MP4 video file, or None if an error occurs.
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
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                video_size = img.size
                valid_image_paths.append(img_path)
                break
        except Exception as e:
            logger.warning(f"Skipping image {img_path} due to error: {e}")
            continue

    if not video_size:
        logger.error("Video Error: No valid images found to determine video size.")
        return None

    try:
        audio_clip = AudioFileClip(audio_path)
        total_audio_duration = audio_clip.duration
    except Exception as e:
        logger.error(f"Video Error: Could not load audio clip '{audio_path}': {e}", exc_info=True)
        return None

    if not valid_image_paths:
        logger.error("Video Error: No valid image paths remained after initial check.")
        if audio_clip: audio_clip.close()
        return None

    num_images = len(valid_image_paths)
    if num_images == 0:
        logger.error("Video Error: No images to process (num_images is 0).")
        if audio_clip: audio_clip.close()
        return None

    duration_per_image = total_audio_duration / num_images if num_images > 0 and total_audio_duration > 0 else default_image_duration

    if duration_per_image <= 0.01: # Check for very small or zero duration
        logger.warning(f"Calculated duration per image is very small or zero ({duration_per_image}). Using default: {default_image_duration}s.")
        duration_per_image = default_image_duration

    logger.info(f"Total audio duration: {total_audio_duration}s, Num images: {num_images}, Duration per image: {duration_per_image}s, FPS: {output_fps}")

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
        if audio_clip: audio_clip.close()
        return None

    output_video_path = None # Define to ensure it's in scope for finally
    final_video_clip = None
    try:
        final_video_clip = concatenate_videoclips(video_clips, method="compose")
        final_video_clip = final_video_clip.set_audio(audio_clip)

        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_video_path = temp_video_file.name
        temp_video_file.close()

        logger.info(f"Writing final video to: {output_video_path}")
        final_video_clip.write_videofile(
            output_video_path,
            fps=output_fps,
            codec="libx264",
            audio_codec="aac",
            logger=None, # MoviePy can be very verbose, use our logger instead.
            ffmpeg_params=["-pix_fmt", "yuv420p"]
        )
        logger.info("Video written successfully.")
        return output_video_path

    except Exception as e:
        logger.error(f"An error occurred during video compilation: {e}", exc_info=True)
        if output_video_path and os.path.exists(output_video_path):
            try:
                os.remove(output_video_path)
            except Exception as remove_e:
                 logger.error(f"Failed to remove temporary video file {output_video_path} after error: {remove_e}")
        return None
    finally:
        for clip in video_clips:
            if hasattr(clip, 'close') and callable(clip.close): clip.close()
        if audio_clip and hasattr(audio_clip, 'close') and callable(audio_clip.close): audio_clip.close()
        if final_video_clip and hasattr(final_video_clip, 'close') and callable(final_video_clip.close): final_video_clip.close()


if __name__ == '__main__':
    logger.info("Video Maker Example (using config and logger)")

    dummy_image_paths = []
    temp_dir_main = tempfile.mkdtemp()
    dummy_audio_path_main = None

    try:
        for i in range(3):
            try:
                img = Image.new('RGB', (320, 240), color = ('blue' if i % 2 == 0 else 'green')) # Smaller size
                img_path = os.path.join(temp_dir_main, f"dummy_image_{i+1}.png")
                img.save(img_path)
                dummy_image_paths.append(img_path)
            except Exception as e_img:
                logger.error(f"Failed to create dummy image {i+1}: {e_img}")

        if not dummy_image_paths:
            raise Exception("No dummy images created for example.")

        try:
            from gtts import gTTS # Ensure gTTS is importable if this example is run directly
            tts_lang = APP_CONFIG.get("tts", {}).get("lang", "en")
            tts = gTTS(text="Test audio for video.", lang=tts_lang)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=temp_dir_main) as temp_f:
                dummy_audio_path_main = temp_f.name
            tts.save(dummy_audio_path_main)

            if not os.path.exists(dummy_audio_path_main) or os.path.getsize(dummy_audio_path_main) == 0:
                 raise Exception("Failed to save dummy audio or file is empty.")
            logger.info(f"Dummy audio created for example: {dummy_audio_path_main}")
        except ImportError:
            logger.error("gTTS library not found, cannot create dummy audio for example. Skipping audio part of example.")
        except Exception as e_audio:
            logger.error(f"Could not create dummy audio file using gTTS: {e_audio}")


        if dummy_image_paths and dummy_audio_path_main:
            logger.info(f"Using images for example: {dummy_image_paths}")
            logger.info(f"Using audio for example: {dummy_audio_path_main}")
            # FPS from config will be used by the function
            video_file = create_video_from_images_and_audio(dummy_image_paths, dummy_audio_path_main)
            if video_file:
                logger.info(f"Successfully generated example video: {video_file}")
                # To prevent auto-deletion by OS, you might copy it elsewhere if you want to inspect it
                # For this test, we'll just log its creation and then it will be cleaned up (or not, if NamedTemporaryFile(delete=False))
                # os.remove(video_file) # If we wanted to clean it up immediately
            else:
                logger.error("Failed to generate example video.")
        elif dummy_image_paths:
            logger.warning("Dummy images created, but no dummy audio. Video example will be incomplete.")
        else:
            logger.error("Not enough dummy files to create an example video.")

    finally:
        logger.info("Cleaning up dummy files and directory from Video Maker example...")
        for p in dummy_image_paths:
            if os.path.exists(p): os.remove(p)
        if dummy_audio_path_main and os.path.exists(dummy_audio_path_main):
            if temp_dir_main in dummy_audio_path_main :
                os.remove(dummy_audio_path_main)
        if os.path.exists(temp_dir_main): os.rmdir(temp_dir_main)
        logger.info("Cleanup complete for Video Maker example.")
