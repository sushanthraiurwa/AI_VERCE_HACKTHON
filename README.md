# Emotion Detection from Voice Signals

## 📌 Project Overview
This project is an **AI-powered forensic solution** that detects and categorizes emotions (happiness, sadness, anger, neutrality) from voice inputs. The system supports  **uploaded audio files** while displaying detected emotions as text output.

### 🎯 Key Features
- 📂 **Supports both live audio & uploaded files**.
- 📊 **Uses MFCC feature extraction** for audio processing.
- 🤖 **Deep learning-based classification** with PyTorch.
- 📈 **Scalable & optimized for accuracy** over speed.
- 🌐 **Deployed using Flask for API integration.**

## 🛠️ Tech Stack
- **Programming Language**: Python 🐍
- **AI/ML Framework**: PyTorch ⚡
- **Audio Processing**: Librosa 🎵
- **Web Framework**: Flask 🌎

## 🚀 Installation Guide
### 🔹 1. Clone the Repository
```sh
git clone https://github.com/your-repo/emotion-detection.git
cd emotion-detection
```

### 🔹 2. Install Dependencies
Ensure Python 3.8+ is installed, then run:
```sh
pip install -r requirements.txt
```
Add these line in the end of App.py

```if __name__ == "__main__":
port = int(os.environ.get("PORT", 5000))  # Get port from Render, default to 5000
app.run(host="0.0.0.0", port=port, debug=True)
```

### 🔹 3. Download Pretrained Model
Hugging Fac elink  : https://huggingface.co/sushanthrai/AI_VERCE_HACKTHON/tree/main
Download the `emotion_model.pth` file from hugging face and place them in the project root.
Download the `lable_classes.npy` file from hugging face  and place them in the project root.
And Change the App.py for load model from local


## 🎤 How It Works
### 1️⃣ **Feature Extraction (MFCCs)**
- Extracts **32 MFCCs** over **32 frames** from input audio.
- Uses `librosa.feature.mfcc()` for feature extraction.
- The extracted features are **normalized using a scaler**.

### 2️⃣ **Deep Learning Model (PyTorch)**
- A **CNN-based model** processes the MFCC features.
- The **Softmax activation** determines the emotion class.
- Outputs **predicted emotion & confidence score**.

### 3️⃣ **Flask API for Deployment**
- Provides endpoints for **real-time detection**:
  - `/predict` → Accepts an audio file & returns emotion.
  - `/live` → Captures microphone input for detection.

## 📌 Usage Guide
### 🔹 1. Run the Flask Server
```sh
python app.py
```
### 🔹 2. Test API with an Audio File
```sh
curl -X POST -F "file=@sample.wav" http://127.0.0.1:5000/upload
```
And upload the audio file

Response:
```json
{
  "emotion":"angry",
"message":"File processed successfully"
}
```


## 📌 Future Enhancements
- 🎙️ Real-time emotion detection from voice input.
- 🧠 **Improve model accuracy** using transformer models.
- 🎭 **Expand emotion classes** beyond 4 categories.
- 🖥️ **Integrate with GUI for better UX**.
- ☁️ **Deploy as a cloud-based API** for scalability.

## 💡 Contributing
Pull requests are welcome! Feel free to improve the model, API, or add new features.

## 📝 License
MIT License. See `LICENSE` for details.

---
🚀 **Developed by Sushanth Rai & Team** 🎯

