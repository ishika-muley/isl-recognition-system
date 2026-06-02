"""
Extract and augment hand landmarks from ISL images
Creates multiple augmented samples from each image for better training data
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import pickle

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

DATASET_PATH = "/Users/nayankumbhare/Downloads/ISL Dataset"
OUTPUT_PATH = "./landmarks_data"
SIGNS = ["Bus", "Hello", "Pray", "Stop", "Telephone", "Water", "Yes"]

os.makedirs(OUTPUT_PATH, exist_ok=True)

def augment_image(image):
    """Apply augmentation to image"""
    augmented = []
    
    # Original
    augmented.append(image.copy())
    
    # Horizontal flip
    augmented.append(cv2.flip(image, 1))
    
    # Slight rotation (+5 degrees)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 5, 1.0)
    augmented.append(cv2.warpAffine(image, M, (w, h)))
    
    # Slight rotation (-5 degrees)
    M = cv2.getRotationMatrix2D((w/2, h/2), -5, 1.0)
    augmented.append(cv2.warpAffine(image, M, (w, h)))
    
    # Brightness increase
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * 1.1
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    augmented.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    
    # Brightness decrease
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * 0.9
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    augmented.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    
    return augmented

def extract_landmarks_with_features(image_path):
    """
    Extract landmarks and additional hand features
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        landmarks = []
        
        # Extract landmarks from all detected hands
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get landmark coordinates
            points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            
            # Add raw landmarks
            for x, y, z in points:
                landmarks.extend([x, y, z])
            
            # Add hand features: distances, angles
            # Distance from wrist to each finger tip
            wrist = np.array(points[0])
            for tip_idx in [4, 8, 12, 16, 20]:  # Thumb, index, middle, ring, pinky tips
                tip = np.array(points[tip_idx])
                dist = np.linalg.norm(tip[:2] - wrist[:2])
                landmarks.append(dist)
            
            # Hand center of mass
            all_points = np.array(points)
            center = all_points.mean(axis=0)
            landmarks.extend(center)
        
        # Pad to fixed size
        max_features = 126 + 5 + 3  # landmarks + distances + center
        if len(landmarks) < max_features:
            landmarks = np.pad(landmarks, (0, max_features - len(landmarks)), mode='constant')
        else:
            landmarks = landmarks[:max_features]
        
        return np.array(landmarks)
    except Exception as e:
        return None

def process_dataset():
    """Process all images with augmentation"""
    all_landmarks = []
    all_labels = []
    
    print("Extracting and augmenting landmarks...\n")
    
    for sign_idx, sign in enumerate(SIGNS):
        sign_path = os.path.join(DATASET_PATH, sign)
        total_generated = 0
        
        if not os.path.exists(sign_path):
            print(f"Warning: {sign_path} not found")
            continue
        
        image_files = [f for f in os.listdir(sign_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_file in image_files:
            image_path = os.path.join(sign_path, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                continue
            
            # Generate augmented versions
            augmented_images = augment_image(image)
            
            for aug_image in augmented_images:
                landmarks = extract_landmarks_with_features(image_path)
                
                if landmarks is not None:
                    all_landmarks.append(landmarks)
                    all_labels.append(sign_idx)
                    total_generated += 1
        
        print(f"{sign}: {total_generated} samples generated")
    
    X = np.array(all_landmarks)
    y = np.array(all_labels)
    
    print(f"\nTotal augmented samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    np.save(os.path.join(OUTPUT_PATH, "X_landmarks.npy"), X)
    np.save(os.path.join(OUTPUT_PATH, "y_labels.npy"), y)
    
    with open(os.path.join(OUTPUT_PATH, "sign_labels.pkl"), "wb") as f:
        pickle.dump(SIGNS, f)
    
    print(f"\nData saved to {OUTPUT_PATH}/")

if __name__ == "__main__":
    process_dataset()
    hands.close()
