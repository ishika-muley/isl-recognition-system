"""
Real-time ISL sign prediction from webcam with improved preprocessing
Features: TTS for predicted signs, Voice-to-text word display
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import pickle
import os
import time
import threading
import queue
import subprocess
import tempfile

# Speech Recognition setup
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf


import pyttsx3

import platform

import platform
import os

class TTSEngine:
    """Synchronous TTS Engine to guarantee audio pipeline completion"""
    def __init__(self):
        self.is_mac = platform.system() == "Darwin"
    
    def speak(self, text):
        """Play TTS sound reliably"""
        safe_text = str(text).replace("'", "").replace('"', "")
        print(f"\n[ AUDIO ] 🔊 Triggering Voice For: ---> {safe_text.upper()} <---")
        
        try:
            if self.is_mac:
                # Runs the 'say' command in the foreground (blocks for ~0.5s to guarantee it finishes)
                os.system(f"say '{safe_text}'")
            else:
                # Dedicated python process fallback for Windows/Linux
                cmd = f'python3 -c "import pyttsx3; engine = pyttsx3.init(); engine.say(\'{safe_text}\'); engine.runAndWait()"'
                os.system(cmd)
        except Exception as e:
            print(f"[ AUDIO ERROR ] Voice failed to trigger: {e}")
            
    def stop(self):
        pass


class SignImageLoader:
    """Preloads and caches sign images from the ISL dataset for display"""
    
    DATASET_PATH = "/Users/nayankumbhare/Downloads/ISL Dataset"
    
    def __init__(self, display_size=(200, 200)):
        self.display_size = display_size
        self._cache = {}  # sign_name (lowercase) -> resized cv2 image
        self._load_all()
    
    def _load_all(self):
        """Preload the first image from each sign folder"""
        if not os.path.isdir(self.DATASET_PATH):
            print(f"Warning: Dataset not found at {self.DATASET_PATH}")
            return
        
        for sign_folder in os.listdir(self.DATASET_PATH):
            folder_path = os.path.join(self.DATASET_PATH, sign_folder)
            if not os.path.isdir(folder_path):
                continue
            
            # Get first image file from the folder
            images = sorted([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp'))
            ])
            
            if images:
                img_path = os.path.join(folder_path, images[0])
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, self.display_size)
                    self._cache[sign_folder.lower()] = img
                    print(f"  Loaded sign image: {sign_folder} ({images[0]})")
        
        print(f"Sign images loaded: {len(self._cache)} signs")
    
    def get_image(self, word):
        """Get the sign image for a word (case-insensitive match).
        Returns the cv2 image or None if no match."""
        key = word.strip().lower()
        return self._cache.get(key)
    
    def get_available_signs(self):
        """Return list of sign names that have images loaded"""
        return list(self._cache.keys())


class VoiceInputHandler:
    """Captures voice input using sounddevice and produces words to display one-by-one"""
    def __init__(self, sign_image_loader=None):
        self.recognizer = sr.Recognizer()
        self.words = []              # words to display
        self.current_word = None     # currently displayed word
        self.current_image = None    # cv2 image for current word (or None)
        self.word_display_time = 0   # when current word started displaying
        self.is_listening = False    # whether actively listening
        self.status_message = ""     # status to show on screen
        self.display_interval = 1.5  # seconds per word
        self.sample_rate = 16000     # recording sample rate
        self.record_duration = 5     # max seconds to record
        self.sign_image_loader = sign_image_loader
    
    def start_listening(self):
        """Start listening in a background thread"""
        if self.is_listening:
            return
        self.is_listening = True
        self.current_word = None
        self.current_image = None
        self.words = []
        self.status_message = "Listening... Speak now!"
        thread = threading.Thread(target=self._listen, daemon=True)
        thread.start()
    
    def _listen(self):
        """Capture audio using sounddevice and recognize speech"""
        try:
            # Record audio using sounddevice (no PyAudio needed)
            self.status_message = "Recording... Speak now! (5 seconds)"
            audio_data = sd.rec(
                int(self.record_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16'
            )
            sd.wait()  # Wait until recording is done
            
            # Save to temporary WAV file
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp_file.name
            tmp_file.close()
            sf.write(tmp_path, audio_data, self.sample_rate)
            
            # Use speech_recognition to transcribe the WAV file
            self.status_message = "Processing speech..."
            with sr.AudioFile(tmp_path) as source:
                audio = self.recognizer.record(source)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            text = self.recognizer.recognize_google(audio)
            self.words = text.split()
            self.current_word = None
            self.current_image = None
            self.word_display_time = 0
            self.status_message = f"Recognized: {text}"
            print(f"Voice recognized: {text}")
            
        except sr.UnknownValueError:
            self.status_message = "Could not understand. Try again (press 'v')."
            print("Voice: Could not understand audio")
        except sr.RequestError as e:
            self.status_message = f"Recognition error: {e}"
            print(f"Voice recognition error: {e}")
        except Exception as e:
            self.status_message = f"Error: {str(e)[:40]}"
            print(f"Voice input error: {e}")
        finally:
            self.is_listening = False
    
    def update(self):
        """Update word display — call each frame. Returns (current_word, current_image)."""
        now = time.time()
        
        if not self.words:
            return self.current_word, self.current_image
        
        # Time to show next word?
        if self.current_word is None or (now - self.word_display_time) >= self.display_interval:
            self.current_word = self.words.pop(0)
            self.word_display_time = now
            # Look up corresponding sign image
            if self.sign_image_loader:
                self.current_image = self.sign_image_loader.get_image(self.current_word)
            else:
                self.current_image = None
        
        return self.current_word, self.current_image
    
    def has_active_display(self):
        """Check if there are words being displayed or pending"""
        return bool(self.words) or (
            self.current_word is not None and 
            (time.time() - self.word_display_time) < self.display_interval
        )


# Load model components
MODEL_PATH = "./models"
model = joblib.load(os.path.join(MODEL_PATH, "isl_model.pkl"))

with open(os.path.join(MODEL_PATH, "sign_labels.pkl"), "rb") as f:
    SIGNS = pickle.load(f)

print(f"Loaded model with signs: {SIGNS}")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuration
CONFIDENCE_THRESHOLD = 0.4  # Lowered threshold for real-time
SMOOTHING_WINDOW = 7

class PredictionBuffer:
    """Buffer for smoothing predictions"""
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.predictions = []
        self.confidences = []
    
    def add(self, pred, confidence):
        self.predictions.append(pred)
        self.confidences.append(confidence)
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.confidences.pop(0)
    
    def get_smoothed(self):
        if not self.predictions:
            return None, 0
        
        # Weighted voting - higher confidence predictions weighted more
        if None in self.predictions:
            predictions_only = [p for p in self.predictions if p is not None]
            if not predictions_only:
                return None, 0
            pred = max(set(predictions_only), key=predictions_only.count)
        else:
            pred = max(set(self.predictions), key=self.predictions.count)
        
        # Get average confidence for this prediction
        mask = np.array(self.predictions) == pred
        avg_conf = np.array(self.confidences)[mask].mean()
        
        return pred, avg_conf

def extract_landmarks_with_features(hand_landmarks_list):
    """Extract landmarks and hand-specific features"""
    landmarks = []
    
    if hand_landmarks_list:
        for hand_landmarks in hand_landmarks_list:
            # Raw landmarks
            points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            
            for x, y, z in points:
                landmarks.extend([x, y, z])
            
            # Hand features: distances from wrist to finger tips
            wrist = np.array(points[0])
            for tip_idx in [4, 8, 12, 16, 20]:
                tip = np.array(points[tip_idx])
                dist = np.linalg.norm(tip[:2] - wrist[:2])
                landmarks.append(dist)
            
            # Hand center
            all_points = np.array(points)
            center = all_points.mean(axis=0)
            landmarks.extend(center)
    
    # Pad to fixed size
    max_features = 126 + 5 + 3
    if len(landmarks) < max_features:
        landmarks = np.pad(landmarks, (0, max_features - len(landmarks)), mode='constant')
    else:
        landmarks = landmarks[:max_features]
    
    return np.array(landmarks)

def predict_sign(landmarks):
    """Predict sign from landmarks"""
    if landmarks is None or np.all(landmarks == 0):
        return None, 0
    
    # Predict
    probabilities = model.predict_proba([landmarks])[0]
    prediction = model.predict([landmarks])[0]
    confidence = probabilities[prediction]
    
    if confidence >= CONFIDENCE_THRESHOLD:
        return SIGNS[prediction], confidence
    
    return None, confidence

def draw_info(frame, sign, confidence, fps, hand_count, voice_handler=None):
    """Draw prediction info on frame"""
    height, width, _ = frame.shape
    
    # Background panel
    cv2.rectangle(frame, (10, 10), (width - 10, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (width - 10, 120), (0, 255, 0), 2)
    
    # Text
    y_offset = 35
    
    if sign:
        text = f"Sign: {sign}"
        confidence_text = f"Confidence: {confidence:.2%}"
        color = (0, 255, 0)
    else:
        text = "Detecting..."
        confidence_text = f"Max confidence: {confidence:.2%}"
        color = (0, 165, 255)
    
    cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, confidence_text, (20, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Hands: {hand_count} | FPS: {fps:.1f}", (20, y_offset + 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Voice input display panel
    if voice_handler:
        panel_y = 130
        img_size = 200  # sign image display size
        panel_height = 100  # text panel height
        
        # Get current word and image
        displayed_word, sign_image = voice_handler.update()
        has_image = sign_image is not None and voice_handler.has_active_display()
        
        # If we have a sign image, make the panel taller to fit it
        total_panel_h = panel_height + (img_size + 20 if has_image else 0)
        
        cv2.rectangle(frame, (10, panel_y), (width - 10, panel_y + total_panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, panel_y), (width - 10, panel_y + total_panel_h), (255, 165, 0), 2)
        
        cv2.putText(frame, "Voice Input:", (20, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Show status message
        if voice_handler.status_message:
            cv2.putText(frame, voice_handler.status_message, (20, panel_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show current word and sign image
        if displayed_word and voice_handler.has_active_display():
            cv2.putText(frame, displayed_word.upper(), (20, panel_y + 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # Draw the sign image if available
            if has_image:
                img_y = panel_y + panel_height + 10
                img_x = (width - img_size) // 2  # center the image
                
                # Ensure image fits within frame bounds
                if img_y + img_size <= height and img_x + img_size <= width and img_x >= 0:
                    # Add a white border around the image
                    cv2.rectangle(frame, (img_x - 3, img_y - 3), 
                                  (img_x + img_size + 3, img_y + img_size + 3), 
                                  (255, 255, 255), 2)
                    frame[img_y:img_y + img_size, img_x:img_x + img_size] = sign_image
                    
                    # Label below image
                    label = f"ISL Sign: {displayed_word.title()}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    label_x = img_x + (img_size - label_size[0]) // 2
                    cv2.putText(frame, label, (label_x, img_y + img_size + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif displayed_word:
                # Word doesn't match any sign — show a hint
                cv2.putText(frame, f"(no ISL sign for '{displayed_word}')", 
                            (20, panel_y + panel_height - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1)
    
    # Instructions
    cv2.putText(frame, "'q' quit | 's' screenshot | 'v' voice input", (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def main():
    """Main real-time prediction loop"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    pred_buffer = PredictionBuffer(SMOOTHING_WINDOW)
    frame_count = 0
    start_time = time.time()
    screenshot_count = 0
    
    # Initialize TTS engine (speaks predicted signs)
    tts = TTSEngine()
    last_spoken_sign = None  # track to avoid repeating
    last_spoken_time = 0     # timestamp to enforce a cooldown
    
    # --- Performance Tracking Setup ---
    prediction_stats = {sign: {"count": 0, "conf_sum": 0.0} for sign in SIGNS}
    frame_latencies = []
    total_voice_inputs = 0
    
    # Preload sign images from dataset
    print("\nLoading sign images from dataset...")
    sign_image_loader = SignImageLoader(display_size=(200, 200))
    
    # Initialize Voice Input handler (with image loader)
    voice_handler = VoiceInputHandler(sign_image_loader=sign_image_loader)
    
    print("\n" + "="*60)
    print("Starting real-time sign prediction...")
    print("="*60)
    print(f"Available signs: {SIGNS}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Take screenshot")
    print("  'v' - Voice input (speak a sentence, words appear one-by-one)")
    print("="*60 + "\n")
    
    while True:
        loop_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for selfie view
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        
        # Process with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        hand_count = 0
        current_sign = None
        current_conf = 0
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Extract and predict
            landmarks = extract_landmarks_with_features(results.multi_hand_landmarks)
            sign, confidence = predict_sign(landmarks)
            pred_buffer.add(sign, confidence)
            
            current_sign, current_conf = pred_buffer.get_smoothed()
        else:
            pred_buffer.add(None, 0)
        
        # Track statistics
        if current_sign:
            prediction_stats[current_sign]["count"] += 1
            prediction_stats[current_sign]["conf_sum"] += current_conf
        
        # --- Feature 1: Speak the predicted sign ---
        now = time.time()
        if current_sign and current_sign != last_spoken_sign:
            tts.speak(current_sign)
            last_spoken_sign = current_sign
            last_spoken_time = now
        elif current_sign is None:
            # Enforce a 2.5 second cooldown before resetting allowing it to speak again. 
            # This prevents the camera from losing your hand for 1 frame and immediately rapid-firing TTS again.
            if (now - last_spoken_time) > 2.5:
                last_spoken_sign = None
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Draw info (with voice handler for word display)
        draw_info(frame, current_sign, current_conf, fps, hand_count, voice_handler)
        
        # Display
        cv2.imshow("ISL Sign Prediction", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_count += 1
            filename = f"screenshot_{screenshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('v'):
            # Feature 2: Voice input mode
            voice_handler.start_listening()
            total_voice_inputs += 1
            
        # Track latency
        frame_latencies.append((time.time() - loop_start) * 1000)
    
    # Cleanup
    tts.stop()
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    # --- Performance Matrix Printout ---
    print("\n" + "="*50)
    print("📊 SESSION PERFORMANCE MATRIX")
    print("="*50)
    print(f"Total Session Time : {elapsed:.2f} seconds")
    print(f"Frames Processed   : {frame_count} frames")
    print(f"Overall FPS        : {fps:.1f} fps")
    
    if frame_latencies:
        avg_latency = sum(frame_latencies) / len(frame_latencies)
        print(f"Average Latency    : {avg_latency:.1f} ms/frame")
        
    print(f"Voice Inputs Used  : {total_voice_inputs}")
    
    print("\n[ Predicted Signs Breakdown ]")
    print(f"{'Sign':<15} | {'Frames Detected':<16} | {'Avg Confidence':<15}")
    print("-" * 55)
    
    signs_detected = False
    for sign, stats in prediction_stats.items():
        count = stats["count"]
        if count > 0:
            avg_conf = stats["conf_sum"] / count
            print(f"{sign:<15} | {count:<16} | {avg_conf:.1%}")
            signs_detected = True
            
    if not signs_detected:
        print("No ISL signs detected during this session.")
        
    print("="*50)
    
    # --- ML Baseline Metrics ---
    metrics_path = "./models/metrics.json"
    if os.path.exists(metrics_path):
        import json
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            print("\n[ Model Evaluation Metrics ]")
            print(f"Test Accuracy    : {metrics.get('accuracy', 0):.1%}")
            
            macro_avg = metrics.get('macro avg', {})
            print(f"Macro Precision  : {macro_avg.get('precision', 0):.1%}")
            print(f"Macro Recall     : {macro_avg.get('recall', 0):.1%}")
            print(f"F1-Score         : {macro_avg.get('f1-score', 0):.1%}")
            print("="*50 + "\n")
        except Exception:
            print("\n")
    else:
        print("\n")

if __name__ == "__main__":
    main()
