"""
Full Pipeline Test: Compare TensorFlow vs TFLite Models

Tests both model formats on same images and compares:
- Accuracy (same predictions?)
- Speed (inference time)
- Predictions (do they match?)
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import json
from datetime import datetime
import matplotlib.pyplot as plt
import random
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

# Serializable preprocessing layer
class MobileNetV3PreprocessingLayer(keras.layers.Layer):
    """Serializable preprocessing layer for MobileNetV3"""
    def call(self, x):
        """Apply MobileNetV3 preprocessing"""
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        return preprocess_input(tf.cast(x, tf.float32))
    
    def get_config(self):
        return super().get_config()

class TrashRecall(keras.metrics.Metric):
    """Compute recall for negative class (trash)."""
    def __init__(self, name='trash_recall', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_binary = tf.cast(y_pred >= 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        tn = tf.reduce_sum(tf.cast((y_pred_binary == 0) & (y_true == 0), tf.float32))
        fp = tf.reduce_sum(tf.cast((y_pred_binary == 1) & (y_true == 0), tf.float32))
        self.tn.assign_add(tn)
        self.fp.assign_add(fp)
    
    def result(self):
        return self.tn / (self.tn + self.fp + 1e-7)
    
    def reset_states(self):
        self.tn.assign(0)
        self.fp.assign(0)

# Constants
TARGET_SIZE = (224, 224)
RUNS_DIR = Path(__file__).parent.parent.parent / 'runs'
TFLITE_DIR = RUNS_DIR / 'tflite'
CONFIG_DIR = Path(__file__).parent.parent.parent.parent
DATA_SPLIT_PATH = Path(__file__).parent.parent.parent / 'data' / 'proper_splits'

def load_keras_models():
    """Load TensorFlow Keras models"""
    # Stage 1
    stage1_models = sorted(RUNS_DIR.glob('stage1_battery_detector_honest_*.keras'))
    if not stage1_models:
        raise FileNotFoundError("No Stage 1 Keras model found")
    stage1_path = stage1_models[-1]
    stage1_model = keras.models.load_model(
        str(stage1_path),
        custom_objects={'MobileNetV3PreprocessingLayer': MobileNetV3PreprocessingLayer}
    )
    
    # Stage 2
    stage2_models = sorted(RUNS_DIR.glob('stage2_waste_classifier_honest_*.keras'))
    if not stage2_models:
        raise FileNotFoundError("No Stage 2 Keras model found")
    stage2_path = stage2_models[-1]
    stage2_model = keras.models.load_model(
        str(stage2_path),
        custom_objects={
            'MobileNetV3PreprocessingLayer': MobileNetV3PreprocessingLayer,
            'TrashRecall': TrashRecall
        }
    )
    
    return stage1_model, stage1_path, stage2_model, stage2_path

def load_tflite_models():
    """Load TFLite models"""
    # Stage 1
    stage1_tflite_models = sorted(TFLITE_DIR.glob('stage1_battery_detector_honest_*_quantized.tflite'))
    if not stage1_tflite_models:
        print("  No Stage 1 TFLite model found - skipping TFLite comparison")
        return None, None, None, None
    
    stage1_path = stage1_tflite_models[-1]
    
    # Stage 2
    stage2_tflite_models = sorted(TFLITE_DIR.glob('stage2_waste_classifier_honest_*_quantized.tflite'))
    if not stage2_tflite_models:
        print("  No Stage 2 TFLite models found - skipping TFLite comparison")
        return None, None, None, None
    
    stage2_path = stage2_tflite_models[-1]
    
    # Load interpreters
    stage1_interpreter = tf.lite.Interpreter(str(stage1_path))
    stage1_interpreter.allocate_tensors()
    
    stage2_interpreter = tf.lite.Interpreter(str(stage2_path))
    stage2_interpreter.allocate_tensors()
    
    return stage1_interpreter, stage1_path, stage2_interpreter, stage2_path

def predict_keras_stage1(model, image, threshold):
    """TensorFlow prediction"""
    batch = np.expand_dims(image, axis=0)
    prob = model.predict(batch, verbose=0)[0, 0]
    is_battery = prob >= threshold
    confidence = prob if is_battery else (1.0 - prob)
    return is_battery, prob, confidence

def predict_tflite_stage1(interpreter, image, threshold):
    """TFLite prediction"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    batch = np.expand_dims(image, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], batch)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prob = float(output_data[0][0])
    is_battery = prob >= threshold
    confidence = prob if is_battery else (1.0 - prob)
    return is_battery, prob, confidence

def predict_keras_stage2(model, image, threshold):
    """TensorFlow Stage 2 prediction"""
    batch = np.expand_dims(image, axis=0)
    prob = model.predict(batch, verbose=0)[0, 0]
    is_recyclable = prob >= threshold
    confidence = prob if is_recyclable else (1.0 - prob)
    return is_recyclable, prob, confidence

def predict_tflite_stage2(interpreter, image, threshold):
    """TFLite Stage 2 prediction"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    batch = np.expand_dims(image, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], batch)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prob = float(output_data[0][0])
    is_recyclable = prob >= threshold
    confidence = prob if is_recyclable else (1.0 - prob)
    return is_recyclable, prob, confidence

def load_thresholds():
    """Load locked thresholds"""
    stage1_config = CONFIG_DIR / 'src' / 'production' / 'stage1_config.json'
    stage2_config = CONFIG_DIR / 'stage2_config.json'
    
    stage1_threshold = 0.15
    stage2_threshold = 0.07
    
    if stage1_config.exists():
        with open(stage1_config) as f:
            stage1_threshold = json.load(f).get('threshold', 0.15)
    
    if stage2_config.exists():
        with open(stage2_config) as f:
            stage2_threshold = json.load(f).get('threshold', 0.07)
    
    return stage1_threshold, stage2_threshold

def collect_test_images(battery_count=3, trash_count=3, recyclable_count=4):
    """Randomly collect test images with specific distribution"""
    images_data = {
        'battery': [],
        'waste_trash': [],
        'waste_recyclable': []
    }
    
    class_configs = [
        ('battery_recybat', 'battery'),
        ('battery_singapore', 'battery'),
        ('battery_original', 'battery'),
        ('recyclable_glass', 'waste_recyclable'),
        ('recyclable_metal', 'waste_recyclable'),
        ('recyclable_paper', 'waste_recyclable'),
        ('recyclable_plastic', 'waste_recyclable'),
        ('recyclable_cardboard', 'waste_recyclable'),
        ('trash_biological', 'waste_trash'),
        ('trash_shoes', 'waste_trash'),
        ('trash_clothes', 'waste_trash'),
        ('trash_trash', 'waste_trash'),
    ]
    
    for class_name, category in class_configs:
        class_dir = DATA_SPLIT_PATH / 'test' / class_name
        if not class_dir.exists():
            continue
        
        for img_path in sorted(class_dir.glob('*')):
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(TARGET_SIZE)
                img_array = np.array(img, dtype=np.uint8)
                images_data[category].append({
                    'path': str(img_path),
                    'class': class_name,
                    'category': category,
                    'image': img_array
                })
            except:
                pass
    
    # Select specified counts from each category
    selected = []
    if images_data['battery']:
        selected.extend(random.sample(images_data['battery'], min(battery_count, len(images_data['battery']))))
    if images_data['waste_trash']:
        selected.extend(random.sample(images_data['waste_trash'], min(trash_count, len(images_data['waste_trash']))))
    if images_data['waste_recyclable']:
        selected.extend(random.sample(images_data['waste_recyclable'], min(recyclable_count, len(images_data['waste_recyclable']))))
    
    random.shuffle(selected)
    return selected

def main():
    print("="*70)
    print("MODEL COMPARISON: TENSORFLOW vs TFLITE")
    print("="*70)
    
    # Load Keras models
    print("\n[LOADING KERAS MODELS]")
    keras_s1, keras_s1_path, keras_s2, keras_s2_path = load_keras_models()
    print(f"  Stage 1: {keras_s1_path.name}")
    print(f"  Stage 2: {keras_s2_path.name}")
    
    # Load TFLite models
    print("\n[LOADING TFLITE MODELS]")
    tflite_s1, tflite_s1_path, tflite_s2, tflite_s2_path = load_tflite_models()
    
    if tflite_s1 is not None:
        print(f"  Stage 1: {tflite_s1_path.name}")
        print(f"  Stage 2: {tflite_s2_path.name}")
        has_tflite = True
    else:
        print("  ⚠️  TFLite models not found yet. Convert them first:")
        print("     python src/convert_to_tflite.py")
        has_tflite = False
    
    # Load thresholds
    stage1_threshold, stage2_threshold = load_thresholds()
    print(f"\n  Stage 1 threshold: {stage1_threshold}")
    print(f"  Stage 2 threshold: {stage2_threshold}")
    
    # Collect test images
    print("\n[COLLECTING TEST IMAGES]")
    test_images = collect_test_images(battery_count=3, trash_count=3, recyclable_count=4)
    print(f"  Total: {len(test_images)} images (3 battery, 3 trash, 4 recyclable)")
    
    # Run comparison
    print("\n" + "="*70)
    print("PREDICTIONS")
    print("="*70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'has_tflite': has_tflite,
        'predictions': []
    }
    
    keras_correct_s1 = 0
    tflite_correct_s1 = 0
    keras_correct_s2 = 0
    tflite_correct_s2 = 0
    
    keras_times_s1 = []
    tflite_times_s1 = []
    keras_times_s2 = []
    tflite_times_s2 = []
    
    agreements = 0
    
    for i, img_data in enumerate(test_images):
        img_path = img_data['path']
        true_class = img_data['class']
        true_category = img_data['category']
        image = img_data['image']
        
        print(f"\n[{i+1}/{len(test_images)}] {Path(img_path).name} ({true_class})")
        
        # Stage 1: Keras
        start = time.time()
        keras_s1_battery, keras_s1_prob, _ = predict_keras_stage1(keras_s1, image, stage1_threshold)
        keras_s1_time = time.time() - start
        keras_times_s1.append(keras_s1_time * 1000)  # ms
        
        expected_battery = (true_category == 'battery')
        if keras_s1_battery == expected_battery:
            keras_correct_s1 += 1
            keras_s1_result = "✓"
        else:
            keras_s1_result = "✗"
        
        print(f"  S1 Keras: {keras_s1_result} ({keras_s1_prob:.4f}) - {keras_s1_time*1000:.1f}ms")
        
        # Stage 1: TFLite
        if has_tflite:
            start = time.time()
            tflite_s1_battery, tflite_s1_prob, _ = predict_tflite_stage1(tflite_s1, image, stage1_threshold)
            tflite_s1_time = time.time() - start
            tflite_times_s1.append(tflite_s1_time * 1000)  # ms
            
            if tflite_s1_battery == expected_battery:
                tflite_correct_s1 += 1
                tflite_s1_result = "✓"
            else:
                tflite_s1_result = "✗"
            
            # Check agreement
            if keras_s1_battery == tflite_s1_battery:
                agreements += 1
                agreement_mark = "✓"
            else:
                agreement_mark = "✗ DISAGREE"
            
            print(f"  S1 TFLite: {tflite_s1_result} ({tflite_s1_prob:.4f}) - {tflite_s1_time*1000:.1f}ms {agreement_mark}")
        
        # Stage 2: Only if not battery
        if not keras_s1_battery and (not has_tflite or not tflite_s1_battery):
            start = time.time()
            keras_s2_recycle, keras_s2_prob, _ = predict_keras_stage2(keras_s2, image, stage2_threshold)
            keras_s2_time = time.time() - start
            keras_times_s2.append(keras_s2_time * 1000)
            
            expected_recycle = (true_category == 'waste_recyclable')
            if keras_s2_recycle == expected_recycle:
                keras_correct_s2 += 1
                keras_s2_result = "✓"
            else:
                keras_s2_result = "✗"
            
            print(f"  S2 Keras: {keras_s2_result} ({keras_s2_prob:.4f}) - {keras_s2_time*1000:.1f}ms")
            
            if has_tflite:
                start = time.time()
                tflite_s2_recycle, tflite_s2_prob, _ = predict_tflite_stage2(tflite_s2, image, stage2_threshold)
                tflite_s2_time = time.time() - start
                tflite_times_s2.append(tflite_s2_time * 1000)
                
                if tflite_s2_recycle == expected_recycle:
                    tflite_correct_s2 += 1
                    tflite_s2_result = "✓"
                else:
                    tflite_s2_result = "✗"
                
                if keras_s2_recycle == tflite_s2_recycle:
                    print(f"  S2 TFLite: {tflite_s2_result} ({tflite_s2_prob:.4f}) - {tflite_s2_time*1000:.1f}ms ✓")
                else:
                    print(f"  S2 TFLite: {tflite_s2_result} ({tflite_s2_prob:.4f}) - {tflite_s2_time*1000:.1f}ms ✗ DISAGREE")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nStage 1 (Battery Detection):")
    print(f"  Keras:  {keras_correct_s1}/{len(test_images)} ({100*keras_correct_s1/len(test_images):.1f}%)")
    if has_tflite:
        print(f"  TFLite: {tflite_correct_s1}/{len(test_images)} ({100*tflite_correct_s1/len(test_images):.1f}%)")
        print(f"  Agreement: {agreements}/{len(test_images)} ({100*agreements/len(test_images):.1f}%)")
    
    if keras_times_s2 or tflite_times_s2:
        total_s2 = keras_correct_s2 + tflite_correct_s2 if has_tflite else keras_correct_s2
        s2_samples = max(len(keras_times_s2), len(tflite_times_s2))
        
        print(f"\nStage 2 (Waste Classification):")
        print(f"  Keras:  {keras_correct_s2}/{s2_samples} ({100*keras_correct_s2/s2_samples:.1f}%)" if s2_samples > 0 else "  Keras:  0/0 (N/A)")
        if has_tflite:
            print(f"  TFLite: {tflite_correct_s2}/{s2_samples} ({100*tflite_correct_s2/s2_samples:.1f}%)" if s2_samples > 0 else "  TFLite: 0/0 (N/A)")
    
    if keras_times_s1:
        print(f"\n  Inference Time (Stage 1):")
        print(f"    Keras:  {np.mean(keras_times_s1):.2f} ± {np.std(keras_times_s1):.2f} ms")
        if has_tflite and tflite_times_s1:
            print(f"    TFLite: {np.mean(tflite_times_s1):.2f} ± {np.std(tflite_times_s1):.2f} ms")
            speedup = np.mean(keras_times_s1) / np.mean(tflite_times_s1)
            print(f"    Speedup: {speedup:.1f}x")
    
    if keras_times_s2:
        print(f"\n  Inference Time (Stage 2):")
        print(f"    Keras:  {np.mean(keras_times_s2):.2f} ± {np.std(keras_times_s2):.2f} ms")
        if has_tflite and tflite_times_s2:
            print(f"    TFLite: {np.mean(tflite_times_s2):.2f} ± {np.std(tflite_times_s2):.2f} ms")
            speedup = np.mean(keras_times_s2) / np.mean(tflite_times_s2)
            print(f"    Speedup: {speedup:.1f}x")
    
    print("\n✓ Comparison complete!")

if __name__ == "__main__":
    main()
