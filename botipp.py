import os
import ffmpeg
import cv2
import librosa
import logging
from flask import Flask, request, jsonify
import scenedetect
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, TextClip, CompositeVideoClip

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Increase file size limit to 500MB
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

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
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        video = scenedetect.VideoManager([video_path])
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
def apply_editing_rules(video_path, audio_analysis, output_path):
    logger.info("Applying editing rules to video")
    try:
        clip = VideoFileClip(video_path)

        # Enhance colors and brightness
        clip = clip.fx(vfx.colorx, 1.2)

        # Add title text
        txt_clip = TextClip("Delicious Food ASMR", fontsize=70, color='white')
        txt_clip = txt_clip.set_pos('center').set_duration(clip.duration)

        # Combine text with video
        video = CompositeVideoClip([clip, txt_clip])

        # Adjust speed: slow motion for first 5 seconds
        slow_motion = clip.subclip(0, 5).fx(vfx.speedx, 0.5)
        rest_of_clip = clip.subclip(5, clip.duration)
        final_clip = concatenate_videoclips([slow_motion, rest_of_clip])

        # Fade in and out
        final_clip = final_clip.fadein(1).fadeout(1)

        # Save the processed video
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    except Exception as e:
        logger.error(f"Error applying editing rules: {e}")
        raise

# Flask route to handle video uploads
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        logger.error("No video file part in the request")
        return jsonify({'status': 'fail', 'message': 'No video file part'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        logger.error("No selected file")
        return jsonify({'status': 'fail', 'message': 'No selected file'}), 400

    video_path = os.path.join('/uploads', video_file.filename)
    try:
        video_file.save(video_path)
        logger.info(f"Video file saved at {video_path}")

        # Process the video
        output_folder = os.path.join('/uploads', 'output')
        os.makedirs(output_folder, exist_ok=True)
        output_video_path = os.path.join(output_folder, 'processed_video.mp4')
        output_audio_path = os.path.join(output_folder, 'enhanced_audio.wav')

        # Extract frames
        logger.info("Starting frame extraction...")
        extract_frames(video_path, output_folder)
        logger.info("Frame extraction completed.")

        # Detect scenes
        logger.info("Starting scene detection...")
        scenes = detect_scenes(video_path)
        logger.info(f"Scenes detected: {scenes}")

        # Analyze audio
        logger.info("Starting audio analysis...")
        audio_analysis = analyze_audio(video_path)
        logger.info(f"Audio analysis: {audio_analysis}")

        # Enhance audio for ASMR
        logger.info("Enhancing audio for ASMR...")
        enhance_audio(video_path, output_audio_path)
        logger.info("Audio enhancement completed.")

        # Apply editing rules
        logger.info("Starting frame editing...")
        apply_editing_rules(video_path, audio_analysis, output_video_path)
        logger.info(f"Edited frames saved at {output_video_path}")

    except Exception as e:
        logger.error(f"Error processing video file: {e}")
        return jsonify({'status': 'fail', 'message': 'Error processing video file'}), 500

    return jsonify({'status': 'success', 'video_path': video_path, 'processed_video_path': output_video_path})

# GET route to check server status
@app.route('/', methods=['GET'])
def home():
    return "Server is running"

if __name__ == '__main__':
    # Ensure the uploads directory exists
    os.makedirs('/uploads', exist_ok=True)
    logger.info("Starting Flask app")
    app.run(debug=True)
