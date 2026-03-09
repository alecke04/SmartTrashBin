#!/usr/bin/env python3
"""
Test manual images with both Keras and TFLite models
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from models.preprocessing import MobileNetV3PreprocessingLayer
from metrics.custom_metrics import TrashRecall

# Configuration
TARGET_SIZE = (224, 224)
RUNS_DIR = Path(__file__).parent.parent / 'runs'
TFLITE_DIR = RUNS_DIR / 'tflite'
CONFIG_DIR = Path(__file__).parent.parent.parent

# ============================================================================
# HARDCODED IMAGE PATHS - Edit these with your actual image paths
# ============================================================================
MANUAL_IMAGE_PATHS = [
    # Example format: r"C:\path\to\image.jpg"
    # Add your image paths here:
    
    # Batteries:
    # r"C:\Users\aleck\Pictures\battery1.jpg",
    # r"C:\Users\aleck\Pictures\battery2.jpg",
    
    # Recyclables:
    # r"C:\Users\aleck\Pictures\recyclable1.jpg",
    
    # Trash:
    # r"C:\Users\aleck\Pictures\trash1.jpg",
]

# ============================================================================

def load_keras_models():
    """Load TensorFlow Keras models"""
    print("[LOADING KERAS MODELS]")
    
    # Stage 1
    stage1_models = sorted(RUNS_DIR.glob('stage1_battery_detector_honest_*.keras'))
    if not stage1_models:
        raise FileNotFoundError("No Stage 1 Keras model found")
    stage1_path = stage1_models[-1]
    stage1_model = keras.models.load_model(
        str(stage1_path),
        custom_objects={'MobileNetV3PreprocessingLayer': MobileNetV3PreprocessingLayer}
    )
    print(f"  Stage 1: {stage1_path.name}")
    
    # Stage 2
    stage2_models = sorted(RUNS_DIR.glob('stage2_waste_classifier_honest_*.keras'))
    if not stage2_models:
        raise FileNotFoundError("No Stage 2 Keras model found")
    stage2_path = stage2_models[-1]
    stage2_model = keras.models.load_model(
        str(stage2_path),
        custom_objects={'MobileNetV3PreprocessingLayer': MobileNetV3PreprocessingLayer, 'TrashRecall': TrashRecall}
    )
    print(f"  Stage 2: {stage2_path.name}")
    
    return stage1_model, stage1_path, stage2_model, stage2_path

def load_tflite_models():
    """Load TFLite models"""
    print("[LOADING TFLITE MODELS]")
    
    # Stage 1
    stage1_models = sorted(TFLITE_DIR.glob('stage1_battery_detector_honest_*_quantized.tflite'))
    if not stage1_models:
        print("  ⚠️  TFLite Stage 1 model not found")
        return None, None, None, None
    stage1_path = stage1_models[-1]
    stage1_interpreter = tf.lite.Interpreter(model_path=str(stage1_path))
    stage1_interpreter.allocate_tensors()
    print(f"  Stage 1: {stage1_path.name}")
    
    # Stage 2
    stage2_models = sorted(TFLITE_DIR.glob('stage2_waste_classifier_honest_*_quantized.tflite'))
    if not stage2_models:
        print("  ⚠️  TFLite Stage 2 model not found")
        return stage1_interpreter, stage1_path, None, None
    stage2_path = stage2_models[-1]
    stage2_interpreter = tf.lite.Interpreter(model_path=str(stage2_path))
    stage2_interpreter.allocate_tensors()
    print(f"  Stage 2: {stage2_path.name}")
    
    return stage1_interpreter, stage1_path, stage2_interpreter, stage2_path

def load_thresholds():
    """Load locked thresholds"""
    stage1_threshold = 0.15
    stage2_threshold = 0.07
    return stage1_threshold, stage2_threshold

def load_image(image_path):
    """Load and preprocess image"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(TARGET_SIZE)
        img_array = np.array(img, dtype=np.uint8)
        return img_array, img
    except Exception as e:
        print(f"  Error loading image: {e}")
        return None, None

def predict_keras_stage1(model, image, threshold):
    """Keras Stage 1 prediction"""
    batch = np.expand_dims(image, axis=0)
    start = time.time()
    prob = model.predict(batch, verbose=0)[0, 0]
    elapsed = time.time() - start
    
    is_battery = prob >= threshold
    return is_battery, prob, elapsed

def predict_tflite_stage1(interpreter, image, threshold):
    """TFLite Stage 1 prediction"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    batch = np.expand_dims(image, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], batch)
    
    start = time.time()
    interpreter.invoke()
    elapsed = time.time() - start
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prob = float(output_data[0][0])
    is_battery = prob >= threshold
    return is_battery, prob, elapsed

def predict_keras_stage2(model, image, threshold):
    """Keras Stage 2 prediction"""
    batch = np.expand_dims(image, axis=0)
    start = time.time()
    prob = model.predict(batch, verbose=0)[0, 0]
    elapsed = time.time() - start
    
    is_recyclable = prob >= threshold
    return is_recyclable, prob, elapsed

def predict_tflite_stage2(interpreter, image, threshold):
    """TFLite Stage 2 prediction"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    batch = np.expand_dims(image, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], batch)
    
    start = time.time()
    interpreter.invoke()
    elapsed = time.time() - start
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prob = float(output_data[0][0])
    is_recyclable = prob >= threshold
    return is_recyclable, prob, elapsed

def main():
    if not MANUAL_IMAGE_PATHS:
        print("ERROR: No image paths defined in MANUAL_IMAGE_PATHS")
        print("Please edit this script and add your image paths, e.g.:")
        print('  MANUAL_IMAGE_PATHS = [')
        print('    r"C:\\Users\\aleck\\Pictures\\battery1.jpg",')
        print('    r"C:\\Users\\aleck\\Pictures\\recyclable1.jpg",')
        print('  ]')
        return
    
    print("="*70)
    print("MANUAL IMAGE TEST: TENSORFLOW vs TFLITE")
    print("="*70)
    
    # Load models
    print()
    keras_s1, keras_s1_path, keras_s2, keras_s2_path = load_keras_models()
    
    print()
    tflite_s1, tflite_s1_path, tflite_s2, tflite_s2_path = load_tflite_models()
    has_tflite = tflite_s1 is not None
    
    # Load thresholds
    stage1_threshold, stage2_threshold = load_thresholds()
    print(f"\n  Stage 1 threshold: {stage1_threshold}")
    print(f"  Stage 2 threshold: {stage2_threshold}")
    
    # Test images
    print(f"\n[TESTING {len(MANUAL_IMAGE_PATHS)} IMAGES]")
    print("="*70)
    
    for idx, image_path in enumerate(MANUAL_IMAGE_PATHS, 1):
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"\n[{idx}/{len(MANUAL_IMAGE_PATHS)}] {image_path.name}")
            print(f"  ERROR: File not found")
            continue
        
        print(f"\n[{idx}/{len(MANUAL_IMAGE_PATHS)}] {image_path.name}")
        
        # Load image
        img_array, img = load_image(str(image_path))
        if img_array is None:
            continue
        
        # Stage 1: Keras
        keras_s1_battery, keras_s1_prob, keras_s1_time = predict_keras_stage1(keras_s1, img_array, stage1_threshold)
        keras_s1_result = "🔋 BATTERY" if keras_s1_battery else "♻️  NOT BATTERY"
        print(f"  S1 Keras:  {keras_s1_result} ({keras_s1_prob:.4f}) - {keras_s1_time*1000:.1f}ms")
        
        # Stage 1: TFLite
        if has_tflite:
            tflite_s1_battery, tflite_s1_prob, tflite_s1_time = predict_tflite_stage1(tflite_s1, img_array, stage1_threshold)
            tflite_s1_result = "🔋 BATTERY" if tflite_s1_battery else "♻️  NOT BATTERY"
            agree = "✓" if keras_s1_battery == tflite_s1_battery else "✗ DISAGREE"
            print(f"  S1 TFLite: {tflite_s1_result} ({tflite_s1_prob:.4f}) - {tflite_s1_time*1000:.1f}ms {agree}")
        
        # Stage 2: Only if not battery
        if not keras_s1_battery and (not has_tflite or not tflite_s1_battery):
            keras_s2_recycle, keras_s2_prob, keras_s2_time = predict_keras_stage2(keras_s2, img_array, stage2_threshold)
            keras_s2_result = "♻️  RECYCLABLE" if keras_s2_recycle else "🗑️  TRASH"
            print(f"  S2 Keras:  {keras_s2_result} ({keras_s2_prob:.4f}) - {keras_s2_time*1000:.1f}ms")
            
            if has_tflite:
                tflite_s2_recycle, tflite_s2_prob, tflite_s2_time = predict_tflite_stage2(tflite_s2, img_array, stage2_threshold)
                tflite_s2_result = "♻️  RECYCLABLE" if tflite_s2_recycle else "🗑️  TRASH"
                agree = "✓" if keras_s2_recycle == tflite_s2_recycle else "✗ DISAGREE"
                print(f"  S2 TFLite: {tflite_s2_result} ({tflite_s2_prob:.4f}) - {tflite_s2_time*1000:.1f}ms {agree}")
    
    print("\n" + "="*70)
    print("✓ Manual image test complete!")

if __name__ == '__main__':
    main()
