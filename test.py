import torch
import numpy as np
import librosa
from model import EmotionClassifier
from sklearn.preprocessing import LabelEncoder

# Load label classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("label_classes.npy", allow_pickle=True)

# ✅ Ensure the input size matches training (8000 if using 40 MFCCs x 200 frames)
input_size = 8000  
hidden_size = 64
num_classes = len(label_encoder.classes_)

# Initialize model
model = EmotionClassifier(input_size, hidden_size, num_classes)

# ✅ Load the trained weights
model.load_state_dict(torch.load("emotion_model.pth"))
model.eval()

print("✅ Model Loaded Successfully!")
