# ISL (Indian Sign Language) Prediction Model

A real-time sign language recognition system. This application is trained on hand landmarks using MediaPipe and machine learning classifiers. It achieves high accuracy through data augmentation and advanced feature extraction. Recently updated to include interactive Text-To-Speech (TTS) predictions and Voice-To-Sign features!

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Libraries & Technologies Used](#libraries--technologies-used)
- [Usage (Real-Time Interface)](#usage-real-time-interface)
- [How It Works](#how-it-works)
- [Dataset Information](#dataset-information)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Real-Time Hand Detection**: Uses MediaPipe for fast, highly accurate 21-point 3D hand landmark detection.
- **Machine Learning Classifiers**: Employs trained scikit-learn models (Random Forest / Gradient Boosting) to classify signs.
- **Data Augmentation**: Automatically generates 6x more training samples from existing images (rotations, brightness adjustments, flips) to ensure robust training.
- **Multi-Hand Support**: Capable of detecting and processing up to two hands simultaneously.
- **Real-Time Feedback & Smoothing**: Uses a rolling 7-frame prediction buffer to smooth out jitter and present confident predictions.
- **🔊 Text-to-Speech (TTS)**: Automatically speaks the predicted sign out loud using the macOS native `say` command via background threading (so it doesn't interrupt the video feed).
- **🎤 Voice-To-Sign (Speech Recognition)**: Press 'v' to speak into your microphone! The app will transcribe your speech into text sequentially, look up the corresponding ISL sign from the dataset, and vividly display the sign image right on the video feed.

## 📁 Project Structure

```
ISL_pred/
├── 01_extract_landmarks.py         # Extract & augment hand landmarks from raw images
├── 02_train_model.py               # Train classifier model
├── 03_real_time_prediction.py      # Real-time webcam prediction & voice functionalities
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── landmarks_data/                 # Extracted features (auto-generated)
│   ├── X_landmarks.npy             # Feature vectors
│   ├── y_labels.npy                # Sign labels
│   └── sign_labels.pkl             # Sign name mapping
└── models/                         # Trained models (auto-generated)
    ├── isl_model.pkl               # Best classifier
    └── sign_labels.pkl             # Sign labels
```

## 🚀 Installation

### Prerequisites
- Python 3.9+
- Webcam
- macOS (recommended for TTS Native support, though Linux/Windows can work with minor modifications to the TTS command)
- Microphone (for Voice-to-Sign feature)

### Setup

1. **Install dependencies:**
```bash
pip3 install -r requirements.txt
```

2. **Verify installation:**
```bash
python3 -c "import cv2, mediapipe, sklearn, speech_recognition, sounddevice, soundfile; print('All packages installed!')"
```

## ⚡ Quick Start

Run the complete pipeline in 3 steps:

```bash
# 1. Extract and augment landmarks from dataset
python3 01_extract_landmarks.py

# 2. Train the model
python3 02_train_model.py

# 3. Start real-time prediction and interaction
python3 03_real_time_prediction.py
```

## 🛠 Libraries & Technologies Used

- **OpenCV (`cv2`)**: Used for all image and video frame processing, rendering the webcam stream, drawing bounding boxes, panels, text, and integrating sign images onto the screen.
- **MediaPipe**: Built by Google, providing state-of-the-art ML solutions for high-fidelity hand tracking.
- **scikit-learn (`sklearn`)**: Used for the machine learning algorithms. Specifically, `RandomForestClassifier` and `GradientBoostingClassifier` are used alongside data splitting and cross-validation utilities to robustly identify signs.
- **NumPy**: The core tool for processing arrays and matrices; used heavily for calculating hand center coordinates, distances, formatting features, and averaging smoothing predictions.
- **Joblib / Pickle**: Used to reliably serialize and save the trained ML models to disk, then instantly load them for real-time predictions.
- **SpeechRecognition (`speech_recognition`)**: Interfaces with the Google Web Speech API to provide highly accurate, cloud-based transcription when the user speaks into the microphone.
- **sounddevice & soundfile**: Captures raw audio natively from the microphone to an array, and converts it cleanly to a `.wav` file entirely bypassing the notorious dependency issues associated with PyAudio/PortAudio wrappers.
- **Subprocess**: Safely delegates Text-to-Speech instructions to the macOS operating system’s internal `say` command, ensuring a lightweight and completely stable offline TTS pipeline without requiring unmaintained third-party audio packages.
- **Threading (`threading`)**: Essential for both TTS and Speech Recognition to occur in the background, ensuring the `cv2` video frame execution loop isn't blocked.

## 📖 Usage (Real-Time Interface)

Start the main application:
```bash
python3 03_real_time_prediction.py
```

### Controls:
- **`q`**: Quit the application.
- **`s`**: Take a screenshot (saved to directory as `screenshot_X.jpg`).
- **`v`**: Enter **Voice Input mode**.

### When in Voice Input Mode:
1. Hit `v` and wait for the screen to say **"Recording... Speak now! (5 seconds)"**.
2. Speak a phrase clearly into the microphone.
3. The app processes the audio.
4. Words will pop up asynchronously on-screen at 1.5-second intervals.
5. If a word matches an available ISL sign, a preview image of that exact sign will be cleanly extracted from your dataset folder and projected directly into the webcam feed underneath the word.

## 🧠 How It Works

### 1. Landmark Extraction
Images are routed through MediaPipe, translating physical geometry into numerical data:
- Extracts 21 precise 3D (x, y, z) landmarks across fingers and palm.
- Derives complex hand features like specific finger distances from the wrist, and absolute center coordinates. Totaling **134 unique features per image**.

### 2. Model Training
Uses the augmented 134-length vectors to train two primary models:
- **Random Forest**: (200 trees, max_depth=15, balanced class weights). Outstanding at generalized classification.
- **Gradient Boosting**: An alternative sequential learning classifier.
Validates stability using 5-fold cross-validation.

### 3. Voice-to Sign Image Pipeline
- Captures raw audio via `sounddevice`, converts it to a standard `int16` `.wav` temporary file.
- Handled efficiently by `speech_recognition` pushing data over Google’s transcription network in the background thread.
- Placed in a UI queue which intercepts the words every 1.5 seconds.
- Uses `cv2` and a caching class (`SignImageLoader`) to lookup the mapped ISL image and overlay it dynamically on the screen matrix boundaries.

## 📂 Dataset Information

Expected directory structure:
```
/Downloads/ISL Dataset/
├── Bus/
├── Hello/
├── Pray/
├── Stop/
├── Telephone/
├── Water/
└── Yes/
```

*Note: Ensure images are clear and front-facing for highest accuracy. The voice feature automatically parses your folder names (`Bus`, `Hello`) to link spoken words to images!*


## 🐛 Troubleshooting

**Voice Input fails or doesn't hear me?**
Check System Settings > Privacy & Security > Microphone, and ensure your Terminal/IDE holds permission to record your microphone.

**Text-to-Speech Error / Nothing happens?**
The TTS engine utilizes the macOS `say` command. To ensure volume levels are appropriate or to test functionality, open terminal and type `say "hello"`. 

**"Could not open webcam" Error?**
Your machine might list external logic cameras or continuity cameras before your main lens. Inside `03_real_time_prediction.py`, modify `cap = cv2.VideoCapture(0)` to `cap = cv2.VideoCapture(1)` or `2`.

---
**Status:** Production Ready ✓
