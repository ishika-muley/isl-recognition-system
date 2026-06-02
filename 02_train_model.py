"""
Train improved model with better configuration for small datasets
Uses cross-validation and better hyperparameters
"""

import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

DATA_PATH = "./landmarks_data"
MODEL_PATH = "./models"

os.makedirs(MODEL_PATH, exist_ok=True)

def load_data():
    """Load preprocessed landmarks"""
    X = np.load(os.path.join(DATA_PATH, "X_landmarks.npy"))
    y = np.load(os.path.join(DATA_PATH, "y_labels.npy"))
    
    with open(os.path.join(DATA_PATH, "sign_labels.pkl"), "rb") as f:
        signs = pickle.load(f)
    
    return X, y, signs

def train_model():
    """Train the classification model"""
    print("Loading augmented data...")
    X, y, signs = load_data()
    
    print(f"Total samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Number of signs: {len(signs)}")
    print(f"Signs: {signs}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}\n")
    
    # Create pipeline with standardization and model
    print("Training Random Forest with better hyperparameters...")
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        ))
    ])
    
    # Train
    model.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=signs, zero_division=0))
    
    # Also train with Gradient Boosting for comparison
    print("\n" + "="*50)
    print("Training Gradient Boosting model...")
    gb_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ))
    ])
    
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    
    print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}\n")
    
    import json
    
    # Save the best model and export metrics
    if gb_accuracy > accuracy:
        print("Gradient Boosting performs better. Saving GB model...")
        joblib.dump(gb_model, os.path.join(MODEL_PATH, "isl_model.pkl"))
        best_model = "Gradient Boosting"
        report_dict = classification_report(y_test, gb_pred, target_names=signs, zero_division=0, output_dict=True)
    else:
        print("Random Forest performs better. Saving RF model...")
        joblib.dump(model, os.path.join(MODEL_PATH, "isl_model.pkl"))
        best_model = "Random Forest"
        report_dict = classification_report(y_test, y_pred, target_names=signs, zero_division=0, output_dict=True)
    
    # Save the evaluation metrics to json
    with open(os.path.join(MODEL_PATH, "metrics.json"), "w") as f:
        json.dump(report_dict, f, indent=4)
    
    with open(os.path.join(MODEL_PATH, "sign_labels.pkl"), "wb") as f:
        pickle.dump(signs, f)
    
    print(f"\n✓ Model saved to {MODEL_PATH}/")
    print(f"✓ Metrics exported to {MODEL_PATH}/metrics.json")
    print(f"Best model: {best_model}")

if __name__ == "__main__":
    train_model()
