import os
import ffmpeg
import cv2
import librosa
import logging
from flask import Flask, request, jsonify
import scenedetect

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

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
        scene_list = scenedetect.detect(video_path)
        return scene_list
    except Exception as e:
        logger.error(f"Error detecting scenes: {e}")
        raise

# Function to detect faces in a frame
def detect_faces(frame):
    logger.info("Detecting faces in the frame")
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
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

# Function to apply editing rules
def apply_editing_rules(frames, audio_analysis):
    logger.info("Applying editing rules to frames")
    edited_frames = []
    try:
        for frame in frames:
            # Apply editing rules based on audio analysis and other criteria
            edited_frames.append(frame)
    except Exception as e:
        logger.error(f"Error applying editing rules: {e}")
        raise
    return edited_frames

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

        # Extract frames
        extract_frames(video_path, output_folder)

        # Detect scenes
        scenes = detect_scenes(video_path)
        logger.info(f"Scenes detected: {scenes}")

        # Analyze audio
        audio_analysis = analyze_audio(video_path)
        logger.info(f"Audio analysis: {audio_analysis}")

        # Apply editing rules
        frames = []  # Load frames as needed
        edited_frames = apply_editing_rules(frames, audio_analysis)
        logger.info(f"Edited frames: {len(edited_frames)}")

    except Exception as e:
        logger.error(f"Error processing video file: {e}")
        return jsonify({'status': 'fail', 'message': 'Error processing video file'}), 500

    return jsonify({'status': 'success', 'video_path': video_path})

# GET route to check server status
@app.route('/', methods=['GET'])
def home():
    return "Server is running"

if __name__ == '__main__':
    # Ensure the uploads directory exists
    os.makedirs('/uploads', exist_ok=True)
    logger.info("Starting Flask app")
    app.run(host='0.0.0.0', port=5000, debug=True)
