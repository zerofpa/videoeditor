import os
import ffmpeg
import cv2
import librosa
import logging
import openai
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, TextClip, CompositeVideoClip
import argparse
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI API
openai.api_key = ''
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
        video_stream = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        scene_manager.detect_scenes(video_stream)
        scene_list = scene_manager.get_scene_list()
        return scene_list
    except Exception as e:
        logger.error(f"Error detecting scenes: {e}")
        raise

# Function to analyze audio
def analyze_audio(audio_path):
    logger.info(f"Analyzing audio at {audio_path}")
    try:
        # Use a smaller chunk duration to reduce memory usage
        y, sr = librosa.load(audio_path, sr=None, duration=60)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        return tempo, beat_frames
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        raise

# Function to enhance audio for ASMR
def enhance_audio(input_audio_path, output_audio_path):
    logger.info(f"Enhancing audio for ASMR in {input_audio_path}")
    try:
        y, sr = librosa.load(input_audio_path, sr=None)
        y_filtered = librosa.effects.preemphasis(y)
        y_amplified = y_filtered * 1.5
        librosa.output.write_wav(output_audio_path, y_amplified, sr)
    except Exception as e:
        logger.error(f"Error enhancing audio: {e}")
        raise

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

# Function to process video files in batches
def process_videos(input_dir, output_dir, prompt, batch_size=1):
    files = [f for f in os.listdir(input_dir) if f.endswith(".webm") or f.endswith(".mp4")]
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        for filename in batch:
            video_path = os.path.join(input_dir, filename)
            output_video_path = os.path.join(output_dir, f"processed_{filename}")
            output_audio_path = os.path.join(output_dir, f"enhanced_audio_{filename}.wav")
            rules_output_path = os.path.join(output_dir, f"rules_{filename}.txt")

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

                # Save the generated rules for external use
                with open(rules_output_path, 'w') as f:
                    f.write(editing_rules)

                # Apply editing rules
                apply_editing_rules(video_path, editing_rules, output_video_path)
                logger.info(f"Video processing completed for {filename}")

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")

# Define the main function
def main():
    parser = argparse.ArgumentParser(description='Process videos with LLM-generated editing rules.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing videos.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory for processed videos.')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to generate editing rules.')

    args = parser.parse_args()

    # Process videos in batches
    process_videos(args.input_dir, args.output_dir, args.prompt, batch_size=1)

if __name__ == '__main__':
    main()
