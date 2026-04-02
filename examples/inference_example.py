"""
GTSRB Traffic Sign Classification - Inference Example
Author: Yasharth Dwivedi
GitHub: https://github.com/YASHARTH-DWIVEDI

This script demonstrates how to use the trained model for inference
on single images and batches of images.
"""

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# Load pre-trained model
print("Loading model...")
MODEL_PATH = 'gtsrb_best_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✓ Model loaded from {MODEL_PATH}")

# Class names
CLASS_NAMES = [f"Class {i}" for i in range(43)]

# Image preprocessing function
def preprocess_image(image_path, target_size=(48, 48)):
    """
    Load and preprocess an image for model inference.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (default: 48x48)
    
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image = image.astype('float32') / 255.0
    
    return image

# Single image prediction
def predict_single(image_path):
    """
    Predict traffic sign class for a single image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Dictionary with prediction results
    """
    # Preprocess
    image = preprocess_image(image_path)
    image_batch = np.expand_dims(image, axis=0)
    
    # Predict
    predictions = model.predict(image_batch, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    return {
        'image_path': image_path,
        'class_id': int(pred_class),
        'class_name': CLASS_NAMES[pred_class],
        'confidence': float(confidence),
        'all_scores': predictions[0].tolist()
    }

# Batch prediction
def predict_batch(image_paths):
    """
    Predict traffic sign classes for a batch of images.
    
    Args:
        image_paths: List of image file paths
    
    Returns:
        List of prediction dictionaries
    """
    images = []
    valid_paths = []
    
    # Load and preprocess all images
    for img_path in image_paths:
        try:
            image = preprocess_image(img_path)
            images.append(image)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
    
    if not images:
        raise ValueError("No valid images to process")
    
    # Stack images
    image_batch = np.array(images)
    
    # Batch predict
    predictions = model.predict(image_batch, verbose=0)
    
    # Format results
    results = []
    for i, img_path in enumerate(valid_paths):
        pred_class = np.argmax(predictions[i])
        confidence = np.max(predictions[i]) * 100
        results.append({
            'image_path': img_path,
            'class_id': int(pred_class),
            'class_name': CLASS_NAMES[pred_class],
            'confidence': float(confidence)
        })
    
    return results

# Visualize prediction
def visualize_prediction(image_path):
    """
    Display image with prediction result.
    
    Args:
        image_path: Path to the image file
    """
    # Load image for display
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get prediction
    result = predict_single(image_path)
    
    # Display
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"{result['class_name']}\nConfidence: {result['confidence']:.2f}%", 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Confidence distribution
def analyze_predictions(image_paths):
    """
    Analyze prediction confidence distribution.
    
    Args:
        image_paths: List of image file paths
    """
    results = predict_batch(image_paths)
    
    confidences = [r['confidence'] for r in results]
    
    print("\n" + "="*50)
    print("PREDICTION ANALYSIS")
    print("="*50)
    print(f"Total images: {len(results)}")
    print(f"Average confidence: {np.mean(confidences):.2f}%")
    print(f"Min confidence: {np.min(confidences):.2f}%")
    print(f"Max confidence: {np.max(confidences):.2f}%")
    print(f"Std confidence: {np.std(confidences):.2f}%")
    
    # Confidence distribution
    high_conf = sum(1 for c in confidences if c > 95)
    med_conf = sum(1 for c in confidences if 85 < c <= 95)
    low_conf = sum(1 for c in confidences if c <= 85)
    
    print(f"\nConfidence distribution:")
    print(f"  >95%: {high_conf} ({100*high_conf/len(results):.1f}%)")
    print(f"  85-95%: {med_conf} ({100*med_conf/len(results):.1f}%)")
    print(f"  <85%: {low_conf} ({100*low_conf/len(results):.1f}%)")

# Example usage
if __name__ == "__main__":
    print("\n" + "="*50)
    print("GTSRB TRAFFIC SIGN CLASSIFIER - INFERENCE")
    print("="*50)
    
    # Example 1: Single image prediction
    print("\n[Example 1] Single Image Prediction")
    print("-" * 50)
    
    # Replace with actual image path
    sample_image = "sample_traffic_sign.jpg"
    
    try:
        result = predict_single(sample_image)
        print(f"Image: {result['image_path']}")
        print(f"Predicted Class: {result['class_name']} (ID: {result['class_id']})")
        print(f"Confidence: {result['confidence']:.2f}%")
    except FileNotFoundError:
        print(f"Note: Sample image '{sample_image}' not found")
        print("To use this example, provide a real traffic sign image")
    
    # Example 2: Batch prediction
    print("\n[Example 2] Batch Prediction")
    print("-" * 50)
    
    # Find all jpg files in current directory
    image_files = list(Path('.').glob('*.jpg')) + list(Path('.').glob('*.png'))
    
    if image_files:
        print(f"Found {len(image_files)} images")
        results = predict_batch([str(f) for f in image_files[:5]])  # First 5
        
        for result in results:
            print(f"  {Path(result['image_path']).name}: "
                  f"{result['class_name']} ({result['confidence']:.2f}%)")
        
        if len(image_files) > 5:
            analyze_predictions([str(f) for f in image_files])
    else:
        print("No images found in current directory")
    
    # Example 3: Top predictions
    print("\n[Example 3] Top-5 Predictions")
    print("-" * 50)
    
    try:
        result = predict_single(sample_image)
        scores = result['all_scores']
        
        # Get top 5
        top_5_idx = np.argsort(scores)[-5:][::-1]
        
        for rank, idx in enumerate(top_5_idx, 1):
            print(f"{rank}. {CLASS_NAMES[idx]}: {scores[idx]*100:.2f}%")
    except:
        print("Sample image not available")
    
    print("\n" + "="*50)
    print("Inference demonstration complete!")
    print("="*50 + "\n")
