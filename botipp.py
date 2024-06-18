import os
import ffmpeg
import cv2
import librosa
import logging
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, TextClip, CompositeVideoClip
import openai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI API
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Function to get editing rules from LLM
def get_editing_rules(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to extract frames from video
def extract_frames(video_path, output_folder):
    logger.info(f"Extracting frames from {video_path} to {output_folder}")
    try:
        ffmpeg.input(video_path).output(f'{output_folder}/frame_%04d.png').run()
    except ffmpeg.Error as e:
        logger.error(f"Error extracting frames: {e}")
        raise

# Function to detect scenes in a video
def detect_scenes(video_path):
    logger.info(f"Detecting scenes in {video_path}")
    try:
        from scenedetect import SceneManager, VideoManager
        from scenedetect.detectors import ContentDetector

        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        video = VideoManager([video_path])
        video.set_downscale_factor()
        video.start()
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        return scene_list
    except Exception as e:
        logger.error(f"Error detecting scenes: {e}")
        raise

# Function to analyze audio
def analyze_audio(audio_path):
    logger.info(f"Analyzing audio at {audio_path}")
    try:
        y, sr = librosa.load(audio_path)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        return tempo, beat_frames
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        raise

# Function to enhance audio for ASMR
def enhance_audio(input_audio_path, output_audio_path):
    y, sr = librosa.load(input_audio_path, sr=None)
    y_filtered = librosa.effects.preemphasis(y)
    y_amplified = y_filtered * 1.5
    librosa.output.write_wav(output_audio_path, y_amplified, sr)

# Function to apply editing rules
def apply_editing_rules(video_path, rules, output_path):
    logger.info("Applying editing rules to video")
    try:
        clip = VideoFileClip(video_path)

        # Apply rules (example rules are included, modify as per LLM output)
        for rule in rules.split('\n'):
            if "slow motion" in rule:
                clip = clip.fx(vfx.speedx, 0.5)
            elif "text" in rule:
                txt_clip = TextClip(rule.replace("text: ", ""), fontsize=70, color='white')
                txt_clip = txt_clip.set_pos('center').set_duration(clip.duration)
                clip = CompositeVideoClip([clip, txt_clip])
            elif "brightness" in rule:
                clip = clip.fx(vfx.colorx, 1.2)
            elif "fade" in rule:
                clip = clip.fadein(1).fadeout(1)

        # Save the processed video
        clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    except Exception as e:
        logger.error(f"Error applying editing rules: {e}")
        raise

# Function to process video files in a directory
def process_videos(input_dir, output_dir, prompt):
    for filename in os.listdir(input_dir):
        if filename.endswith(".webm") or filename.endswith(".mp4"):
            video_path = os.path.join(input_dir, filename)
            output_video_path = os.path.join(output_dir, f"processed_{filename}")
            output_audio_path = os.path.join(output_dir, f"enhanced_audio_{filename}.wav")

            try:
                os.makedirs(output_dir, exist_ok=True)

                # Extract frames
                logger.info(f"Processing video: {filename}")
                extract_frames(video_path, output_dir)

                # Detect scenes
                scenes = detect_scenes(video_path)
                logger.info(f"Scenes detected: {scenes}")

                # Analyze audio
                audio_analysis = analyze_audio(video_path)
                logger.info(f"Audio analysis: {audio_analysis}")

                # Enhance audio for ASMR
                enhance_audio(video_path, output_audio_path)
                logger.info(f"Audio enhancement completed for {filename}")

                # Get editing rules from LLM
                logger.info("Getting editing rules from LLM...")
                editing_rules = get_editing_rules(prompt)
                logger.info(f"Editing rules received: {editing_rules}")

                # Apply editing rules
                apply_editing_rules(video_path, editing_rules, output_video_path)
                logger.info(f"Video processing completed for {filename}")

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")

# Define the input and output directories
input_directory = '/path/to/input_videos'
output_directory = '/path/to/output_videos'
llm_prompt = "Create video editing rules for a Food ASMR YouTube Shorts video"

# Process videos
process_videos(input_directory, output_directory, llm_prompt)
