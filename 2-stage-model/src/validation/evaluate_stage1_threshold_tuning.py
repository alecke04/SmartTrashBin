"""
Stage 1 Threshold Tuning + Per-Source Performance Analysis

After training, use this script to:
1. Find optimal battery detection threshold on VAL set
2. Analyze per-source performance (recybat vs singapore vs original)
3. Lock threshold before final test evaluation
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

# Serializable preprocessing layer (must match train script)
class MobileNetV3PreprocessingLayer(keras.layers.Layer):
    """Serializable preprocessing layer for MobileNetV3"""
    def call(self, x):
        """Apply MobileNetV3 preprocessing"""
        return preprocess_input(tf.cast(x, tf.float32))
    
    def get_config(self):
        return super().get_config()

sys.path.insert(0, str(Path(__file__).parent.parent))

# Constants
TARGET_SIZE = (224, 224)
RUNS_DIR = Path(__file__).parent.parent / 'runs'

def load_images_with_source_labels(split_dir):
    """Load images and track their source (battery_recybat, battery_singapore, etc)"""
    images = []
    labels = []
    sources = []
    
    # Define source folders and their class labels
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
        source_folder = Path(split_dir) / source_name
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
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels), np.array(sources)

def main():
    print("="*70)
    print("STAGE 1: THRESHOLD TUNING + PER-SOURCE ANALYSIS")
    print("="*70)
    
    # Find latest model
    stage1_models = sorted(RUNS_DIR.glob("stage1_battery_detector_honest_*.keras"))
    if not stage1_models:
        print("Error: No Stage 1 model found. Train first with: python src/train_stage1_honest.py")
        sys.exit(1)
    
    # Try loading models from newest to oldest (in case latest is corrupted)
    model_path = None
    model = None
    for candidate_path in reversed(stage1_models):
        try:
            print(f"\nAttempting to load: {candidate_path.name}")
            model = keras.models.load_model(candidate_path, custom_objects={'MobileNetV3PreprocessingLayer': MobileNetV3PreprocessingLayer})
            model_path = candidate_path
            print(f"Successfully loaded: {model_path.name}")
            break
        except Exception as e:
            print(f"  Failed to load {candidate_path.name}: {type(e).__name__}")
            continue
    
    if model is None:
        print("Error: Could not load any Stage 1 model. All models corrupted. Retrain with: python src/train_stage1_honest.py")
        sys.exit(1)
    
    # Load validation set
    print("\nLoading validation set...")
    X_val, y_val, sources_val = load_images_with_source_labels('data/proper_splits/val')
    print(f"Validation set: {len(X_val)} images")
    
    # Get predictions (probabilities)
    print("\nGenerating predictions on validation set...")
    probs = model.predict(X_val, verbose=0)[:, 0]  # P(battery)
    
    # Threshold sweep
    print("\n" + "="*70)
    print("THRESHOLD SWEEP ON VALIDATION SET")
    print("="*70)
    
    # Safety-first: sweep from very low thresholds (minimize false negatives)
    thresholds = np.arange(0.05, 0.95, 0.02)
    results = []
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        tp = np.sum((preds == 1) & (y_val == 1))
        fn = np.sum((preds == 0) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))
        tn = np.sum((preds == 0) & (y_val == 0))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / len(y_val)
        
        results.append({
            'threshold': float(threshold),
            'recall': float(recall),
            'precision': float(precision),
            'accuracy': float(accuracy),
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp),
            'tn': int(tn),
        })
        
        print(f"Threshold {threshold:.2f}: Recall={recall:.3f}, Precision={precision:.3f}, FN={fn} (CRITICAL)")
    
    # Find optimal threshold (maximize recall, secondary: minimize FN)
    best_result = max(results, key=lambda x: (x['recall'], -x['fn']))
    best_threshold = best_result['threshold']
    
    print(f"\n{'='*70}")
    print(f"OPTIMAL THRESHOLD: {best_threshold:.2f}")
    print(f"Recall: {best_result['recall']:.3f} (catches {int(best_result['tp'])} out of {int(best_result['tp'] + best_result['fn'])} batteries)")
    print(f"False Negatives: {int(best_result['fn'])} (MISSED BATTERIES - BAD)")
    print(f"False Positives: {int(best_result['fp'])} (acceptable for safety)")
    print(f"{'='*70}")
    
    # Per-source analysis
    print("\nPER-SOURCE PERFORMANCE (on validation set, using best threshold):")
    print("="*70)
    
    preds_best = (probs >= best_threshold).astype(int)
    
    unique_sources = np.unique(sources_val)
    source_results = {}
    
    for source in unique_sources:
        mask = sources_val == source
        y_source = y_val[mask]
        preds_source = preds_best[mask]
        
        tp = np.sum((preds_source == 1) & (y_source == 1))
        fn = np.sum((preds_source == 0) & (y_source == 1))
        fp = np.sum((preds_source == 1) & (y_source == 0))
        tn = np.sum((preds_source == 0) & (y_source == 0))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / len(y_source)
        
        source_results[source] = {
            'recall': float(recall),
            'accuracy': float(accuracy),
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp),
            'total_positives': int(tp + fn),
            'total': int(np.sum(mask))
        }
        
        # Print with explicit FN count (this is the critical safety metric)
        total_positives = int(tp + fn)
        print(f"{source:25s} | Recall: {recall:.3f} | FN: {fn:2d}/{total_positives:2d} | Total: {np.sum(mask):3d}")
    
    # Save threshold and results
    timestamp = model_path.stem.split('_')[-1]
    results_file = RUNS_DIR / f"stage1_threshold_tuning_{timestamp}.json"
    
    output = {
        'model': model_path.name,
        'best_threshold': best_threshold,
        'best_result': best_result,
        'all_thresholds': results,
        'per_source_analysis': source_results,
        'notes': [
            'Threshold optimized for RECALL (safety first)',
            'Recall is the critical metric for battery detection',
            'Use best_threshold in inference pipeline',
            'Test set evaluation should use this locked threshold'
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[OK] Results saved: {results_file.name}")
    print("\n[THRESHOLD LOCKED]")
    print(f"Use threshold={best_threshold:.2f} in production inference")
    print("\nNext: python src/evaluate_stage1_test_honest.py")
    print("      to get final metrics on LOCKED test set (run once!)")

if __name__ == "__main__":
    main()
