"""
Stage 1 Demo: Visualize predictions on 10 random test images

Shows battery vs non-battery classification with model confidence.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

# Serializable preprocessing layer (must match train script)
class MobileNetV3PreprocessingLayer(keras.layers.Layer):
    """Serializable preprocessing layer for MobileNetV3"""
    def call(self, x):
        """Apply MobileNetV3 preprocessing"""
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        return preprocess_input(tf.cast(x, tf.float32))
    
    def get_config(self):
        return super().get_config()

# Constants
TARGET_SIZE = (224, 224)
RUNS_DIR = Path(__file__).parent.parent.parent / 'runs'
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_all_test_images_with_labels():
    """Load all test images and their labels"""
    images = []
    labels = []
    sources = []
    filepaths = []
    
    source_configs = [
        ('battery_recybat', 1),
        ('battery_singapore', 1),
        ('battery_original', 1),
        ('recyclable_glass', 0),
        ('recyclable_metal', 0),
        ('recyclable_paper', 0),
        ('recyclable_plastic', 0),
        ('recyclable_cardboard', 0),
        ('trash_biological', 0),
        ('trash_clothes', 0),
        ('trash_shoes', 0),
        ('trash_trash', 0),
    ]
    
    for source_name, label in source_configs:
        source_folder = Path('data/proper_splits/test') / source_name
        if not source_folder.exists():
            continue
        
        for img_path in sorted(source_folder.glob('*')):
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(TARGET_SIZE)
                img_array = np.array(img, dtype=np.uint8)
                images.append(img_array)
                labels.append(label)
                sources.append(source_name)
                filepaths.append(str(img_path))
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels), np.array(sources), np.array(filepaths)

def main():
    print("="*70)
    print("STAGE 1 DEMO: 10 RANDOM TEST PREDICTIONS")
    print("="*70)
    
    # Load model
    stage1_models = sorted(RUNS_DIR.glob("stage1_battery_detector_honest_*.keras"))
    if not stage1_models:
        print("Error: No Stage 1 model found.")
        sys.exit(1)
    
    model_path = stage1_models[-1]
    print(f"\nLoading model: {model_path.name}")
    model = keras.models.load_model(model_path, custom_objects={'MobileNetV3PreprocessingLayer': MobileNetV3PreprocessingLayer})
    
    # Load all test set images
    print("\nLoading test set...")
    X_test, y_test, sources_test, filepaths_test = load_all_test_images_with_labels()
    print(f"Test set: {len(X_test)} images")
    
    # Separate battery and non-battery images
    battery_indices = np.where(y_test == 1)[0]
    non_battery_indices = np.where(y_test == 0)[0]
    
    print(f"  Batteries: {len(battery_indices)}")
    print(f"  Non-batteries: {len(non_battery_indices)}")
    
    # Sample: 10 batteries, 10 non-batteries
    sample_battery_indices = np.random.choice(battery_indices, size=min(10, len(battery_indices)), replace=False)
    sample_non_battery_indices = np.random.choice(non_battery_indices, size=min(10, len(non_battery_indices)), replace=False)
    
    sample_indices = np.concatenate([sample_battery_indices, sample_non_battery_indices])
    np.random.shuffle(sample_indices)
    
    # Get predictions
    # Load threshold from config
    config_path = Path(__file__).parent.parent.parent / 'src' / 'production' / 'stage1_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    threshold = config['threshold']
    
    print("\nGenerating predictions...")
    X_sample = X_test[sample_indices]
    probs = model.predict(X_sample, verbose=0)[:, 0]  # P(battery)
    preds = (probs >= threshold).astype(int)
    
    # Display results
    print("\n" + "="*70)
    print(f"PREDICTIONS (using threshold={threshold})")
    print("="*70)
    
    predictions = []
    
    # Create figure to display images - 4 rows x 5 columns = 20 images
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_indices, 1):
        true_label = y_test[idx]
        pred_label = preds[i-1]
        confidence = probs[i-1]
        source = sources_test[idx]
        filepath = filepaths_test[idx]
        filename = Path(filepath).name
        
        true_class = "BATTERY" if true_label == 1 else "NON-BATTERY"
        pred_class = "BATTERY" if pred_label == 1 else "NON-BATTERY"
        confidence_pct = 100.0 * confidence if pred_label == 1 else 100.0 * (1 - confidence)
        
        correct = "[OK]" if pred_label == true_label else "[MISS]"
        
        print(f"\n{i}. {filename}")
        print(f"   Source: {source}")
        print(f"   TRUE: {true_class:12s} | PRED: {pred_class:12s} {correct}")
        print(f"   Confidence: {confidence_pct:.1f}%")
        
        # Load and display image
        img = Image.open(filepath).convert('RGB')
        ax = axes[i-1]
        ax.imshow(img)
        
        # Title with prediction
        title_color = 'green' if pred_label == true_label else 'red'
        title = f"{pred_class}\n{confidence_pct:.0f}%"
        ax.set_title(title, fontsize=11, fontweight='bold', color=title_color)
        ax.axis('off')
        
        predictions.append({
            'image': filename,
            'source': source,
            'true_label': int(true_label),
            'predicted_label': int(pred_label),
            'battery_probability': float(confidence),
            'confidence_pct': float(confidence_pct),
            'correct': int(pred_label == true_label)
        })
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f"demo_stage1_predictions_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n[OK] Demo results saved: {results_file.name}")
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    correct_count = sum(1 for p in predictions if p['correct'])
    print(f"Correct: {correct_count}/20")
    print(f"Accuracy (on sample): {100*correct_count/20:.0f}%")
    print("\nTo run again: python src/demo_stage1_predictions.py")
    
    # Show images
    plt.suptitle(f"Stage 1 Battery Detector Demo - 20 Test Images (Threshold={threshold})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
