import torch
import numpy as np
import librosa
from model import EmotionClassifier
from sklearn.preprocessing import LabelEncoder

# Load the trained model
MODEL_PATH = "emotion_model.pth"
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("label_classes.npy", allow_pickle=True)

# Model Parameters (match with model.py)
# Model Parameters (match with model.py)
input_size = 8000  # Set this to 40 to match MFCC shape
hidden_size = 64
num_classes = len(label_encoder.classes_)

# Initialize and load model
model = EmotionClassifier(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()


# Initialize and load model
model = EmotionClassifier(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Function to extract features from a new audio file
def extract_features(file_path, max_pad_len=200):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]

    return mfccs.flatten()  # Convert 40x200 -> (8000,)

# Predict emotion from an audio file
def predict_emotion(audio_path):
    features = extract_features(audio_path)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(features)
        predicted_class = torch.argmax(output, dim=1).item()
    
    emotion = label_encoder.inverse_transform([predicted_class])[0]
    return emotion

# Test the model with a new audio file
test_audio = "D:/thanmya/finalsushanthaudio/uploads/OAF_back_angry.wav" 
# D:\thanmya\finalsushanthaudio\uploads\song7.wav
 # Change to your test file
predicted_emotion = predict_emotion(test_audio)
print(f"🎤 Predicted Emotion: {predicted_emotion}")
