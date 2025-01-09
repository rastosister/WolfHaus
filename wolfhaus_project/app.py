import os
import uuid
import time
from flask import Flask, render_template, request, jsonify
import whisper

# Initialize Flask app and Whisper model
app = Flask(__name__)
model = whisper.load_model("base")

# Directory to save uploaded files and transcriptions
UPLOAD_FOLDER = 'uploads'
TRANSCRIPTION_FOLDER = 'transcriptions'
MAX_FOLDER_SIZE_MB = 100  # Maximum folder size in MB
VALID_EXTENSIONS = {'wav', 'mp3', 'm4a'}  # Allowed audio file extensions

os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder for audio files
os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)  # Create folder for transcriptions
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TRANSCRIPTION_FOLDER'] = TRANSCRIPTION_FOLDER


# Function to validate file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VALID_EXTENSIONS


# Function to calculate folder size
def calculate_folder_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert bytes to MB


# Function to clean up old files if folder size exceeds the limit
def cleanup_folder(folder, max_size_mb):
    while calculate_folder_size(folder) > max_size_mb:
        files = sorted(
            (os.path.join(folder, f) for f in os.listdir(folder)),
            key=os.path.getmtime
        )
        if files:
            os.remove(files[0])  # Delete the oldest file


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Validate file extension
    if not allowed_file(audio_file.filename):
        return jsonify({'error': 'Invalid file type. Only .wav, .mp3, .m4a are allowed.'}), 400

    # Generate a unique filename with timestamp and UUID
    unique_filename = f"{int(time.time())}_{uuid.uuid4().hex}.{audio_file.filename.rsplit('.', 1)[1].lower()}"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    # Save the audio file permanently
    audio_file.save(audio_path)

    # Clean up folder if it exceeds the size limit
    cleanup_folder(app.config['UPLOAD_FOLDER'], MAX_FOLDER_SIZE_MB)

    # Use Whisper to process the saved audio file
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # Transcribe the audio
    result = model.transcribe(audio)
    transcription = result['text']

    # Save the transcription to a file
    transcription_filename = os.path.splitext(unique_filename)[0] + ".txt"
    transcription_path = os.path.join(app.config['TRANSCRIPTION_FOLDER'], transcription_filename)
    with open(transcription_path, 'w') as transcription_file:
        transcription_file.write(transcription)

    # Clean up transcription folder if it exceeds the size limit
    cleanup_folder(app.config['TRANSCRIPTION_FOLDER'], MAX_FOLDER_SIZE_MB)

    return jsonify({
        'transcription': transcription,
        'audio_path': audio_path,
        'transcription_path': transcription_path
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
