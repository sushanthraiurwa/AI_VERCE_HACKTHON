import os
import librosa
import numpy as np

# Define dataset path
DATASET_PATH = "D:/AI_VERCE/finalsushanthaudio/dataset"

# Emotions mapping (based on folder names)
EMOTIONS = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "pleasant_surprise": "surprise",
    "sad": "sad"
}

# Function to extract MFCC features
def extract_features(file_path, max_pad_len=200):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)  # 40 MFCCs
        
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# Prepare dataset
features_list = []
labels_list = []

for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                label = next((emo for key, emo in EMOTIONS.items() if key in folder.lower()), None)
                if label:
                    features = extract_features(file_path)
                    if features is not None:
                        features_list.append(features)
                        labels_list.append(label)

# Convert to NumPy array
features_array = np.array(features_list, dtype=object)  # Use dtype=object to avoid shape mismatch
labels_array = np.array(labels_list)

# Save data as dictionary
data_dict = {"features": features_array, "label": labels_array}
np.save("preprocessed_data.npy", data_dict)

print(f"✅ Preprocessing Complete! Processed {len(features_list)} audio files.")
