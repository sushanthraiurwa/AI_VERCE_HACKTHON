import os
import librosa
import numpy as np
from flask import Flask, request, jsonify
import torch
from model import EmotionClassifier
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
import os
import gdown

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and label encoder


# Define file paths
MODEL_PATH = "emotion_model.pth"
LABELS_PATH = "label_classes.npy"  # Change name if it's a different file

# Google Drive file IDs
MODEL_DRIVE_ID = "17-v6WN3m3GwUOsS7aNDJTN7IJCJ2Iy6W"
LABELS_DRIVE_ID = "1hpiqnT2e3LfNI8zqnE-nrcSJB1o4VOLf"

def download_from_drive(file_id, output_path):
    """Downloads a file from Google Drive if it doesn't exist."""
    if not os.path.exists(output_path):  
        print(f"Downloading {output_path} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# Download necessary files
download_from_drive(MODEL_DRIVE_ID, MODEL_PATH)
download_from_drive(LABELS_DRIVE_ID, LABELS_PATH)


# Model parameters (must match the model.py configuration)
input_size = 8000  # 40 MFCCs * 200 frames
hidden_size = 64
num_classes = 7  # Number of emotion classes

# Initialize and load the model
model = EmotionClassifier(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Load label encoder classes
# Load label encoder classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load(LABELS_PATH, allow_pickle=True)  #

# Helper function to check allowed file extensions
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract MFCC features from an audio file
def extract_features(audio_bytes, max_pad_len=200):
    try:
        # Load audio from bytes
        audio, sample_rate = librosa.load(BytesIO(audio_bytes), sr=None, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)  # Extract 40 MFCCs

        # Pad or truncate MFCCs to a fixed length
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs.flatten()  # Flatten to (8000,)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Function to predict emotion from an audio file
def predict_emotion(audio_bytes):
    features = extract_features(audio_bytes)
    if features is None:
        return "Error: Could not extract features."

    # Convert features to a PyTorch tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Predict emotion
    with torch.no_grad():
        output = model(features_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # Map predicted class to emotion label
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
        # Read the file into memory
        audio_bytes = file.read()

        # Predict emotion
        emotion = predict_emotion(audio_bytes)

        return jsonify({"message": "File processed successfully", "emotion": emotion})

    return jsonify({"error": "Invalid file type"}), 400

# Run the Flask app
if __name__ == "__main__":
    # For production, bind to all available IPs (0.0.0.0) and specify the port
    app.run(host="0.0.0.0", port=5000, debug=True)  # Adjust port as necessary
