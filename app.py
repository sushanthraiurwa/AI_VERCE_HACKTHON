import os
import librosa
import numpy as np
import requests
from flask import Flask, request, jsonify
import torch
from model import EmotionClassifier
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Define Hugging Face URLs
MODEL_URL = "https://huggingface.co/sushanthrai/AI_VERCE_HACKTHON/resolve/main/emotion_model.pth"
LABELS_URL = "https://huggingface.co/sushanthrai/AI_VERCE_HACKTHON/resolve/main/label_classes.npy"

# Define file paths
MODEL_PATH = "emotion_model.pth"
LABELS_PATH = "label_classes.npy"

def download_file(url, output_path):
    """Downloads a file from Hugging Face if it doesn't exist."""
    if not os.path.exists(output_path):  
        print(f"Downloading {output_path} from Hugging Face...")
        response = requests.get(url, stream=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(1024):
                if chunk:
                    f.write(chunk)

# Download necessary files
download_file(MODEL_URL, MODEL_PATH)
download_file(LABELS_URL, LABELS_PATH)

# Model parameters (must match model.py configuration)
input_size = 8000  # 40 MFCCs * 200 frames
hidden_size = 64
num_classes = 7  # Number of emotion classes

# Initialize and load the model
model = EmotionClassifier(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Load label encoder classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(LABELS_PATH, allow_pickle=True)

# Helper function to check allowed file extensions
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract MFCC features from an audio file
def extract_features(audio_bytes, max_pad_len=200):
    try:
        audio, sample_rate = librosa.load(BytesIO(audio_bytes), sr=None, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Pad or truncate MFCCs to a fixed length
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs.flatten()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Function to predict emotion from an audio file
def predict_emotion(audio_bytes):
    features = extract_features(audio_bytes)
    if features is None:
        return "Error: Could not extract features."

    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(features_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    emotion = label_encoder.inverse_transform([predicted_class])[0]
    return emotion

# Flask route for file upload and emotion prediction
@app.route("/upload", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        audio_bytes = file.read()
        emotion = predict_emotion(audio_bytes)
        return jsonify({"message": "File processed successfully", "emotion": emotion})

    return jsonify({"error": "Invalid file type"}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
