"""
Stage 2: Threshold Tuning + Per-Source Analysis

Run on validation set to find optimal threshold for recyclable classification.
NEVER use this for test evaluation.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from PIL import Image
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Custom preprocessing layer
class MobileNetV3PreprocessingLayer(keras.layers.Layer):
    """Serializable preprocessing layer for MobileNetV3"""
    def call(self, x):
        return preprocess_input(tf.cast(x, tf.float32))
    def get_config(self):
        return super().get_config()

# Custom metric
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

RUNS_DIR = Path(__file__).parent.parent / 'runs'
RUNS_DIR.mkdir(parents=True, exist_ok=True)

DATA_SPLIT_PATH = Path(__file__).parent.parent / 'data' / 'proper_splits'

def load_images_with_source_labels(split_dir):
    """Load images and track their source (class)"""
    images = []
    labels = []
    sources = []
    filepaths = []
    
    source_configs = [
        ('recyclable_glass', 1),
        ('recyclable_metal', 1),
        ('trash_biological', 0),
        ('trash_other', 0),
    ]
    
    for source_name, label in source_configs:
        source_folder = Path(split_dir) / source_name
        if not source_folder.exists():
            continue
        
        for img_path in sorted(source_folder.glob('*')):
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img, dtype=np.uint8)  # Raw 0-255 uint8 (preprocess_input in model handles normalization)
                images.append(img_array)
                labels.append(label)
                sources.append(source_name)
                filepaths.append(str(img_path))
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels), np.array(sources), np.array(filepaths)

def main():
    print("="*70)
    print("STAGE 2: THRESHOLD TUNING + PER-SOURCE ANALYSIS")
    print("="*70)
    print("\nTuning on validation set (not test set)")
    
    # Find latest model
    RUNS_DIR = Path('runs')
    stage2_models = sorted(RUNS_DIR.glob("stage2_waste_classifier_honest_*.keras"))
    if not stage2_models:
        print("Error: No Stage 2 model found.")
        sys.exit(1)
    
    model_path = stage2_models[-1]
    print(f"\nLoading model: {model_path.name}")
    model = keras.models.load_model(
        model_path,
        custom_objects={
            'MobileNetV3PreprocessingLayer': MobileNetV3PreprocessingLayer,
            'TrashRecall': TrashRecall
        }
    )
    
    # Load validation set
    print("\nLoading validation set...")
    X_val, y_val, sources_val, filepaths_val = load_images_with_source_labels(
        DATA_SPLIT_PATH / 'val'
    )
    
    recyclable_count = np.sum(y_val)
    trash_count = len(y_val) - recyclable_count
    
    print(f"Validation set: {len(X_val)} images")
    print(f"  Recyclable: {int(recyclable_count)}, Trash: {int(trash_count)}")
    
    # Generate predictions
    print("\nGenerating predictions on validation set...")
    probs = model.predict(X_val, verbose=0)[:, 0]  # P(recyclable)
    
    # Threshold sweep
    print("\n" + "="*70)
    print("THRESHOLD SWEEP ON VALIDATION SET")
    print("="*70)
    
    thresholds = np.arange(0.05, 0.96, 0.02)
    best_threshold = 0.5
    best_metric = -np.inf
    threshold_results = {}
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        
        tp = np.sum((preds == 1) & (y_val == 1))
        fn = np.sum((preds == 0) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))
        tn = np.sum((preds == 0) & (y_val == 0))
        
        recyclable_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        trash_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_recall = (recyclable_recall + trash_recall) / 2
        
        threshold_results[f"{threshold:.2f}"] = {
            'recyclable_recall': float(recyclable_recall),
            'trash_recall': float(trash_recall),
            'balanced_recall': float(balanced_recall),
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp),
            'tn': int(tn)
        }
        
        # Metric: balanced recall (treat both classes equally)
        metric = balanced_recall
        
        print(f"Threshold {threshold:.2f}: RecycleRecall={recyclable_recall:.3f}, TrashRecall={trash_recall:.3f}, Balanced={balanced_recall:.3f}, FN={int(fn)}, FP={int(fp)}")
        
        if metric > best_metric:
            best_metric = metric
            best_threshold = threshold
    
    # Best threshold
    print("\n" + "="*70)
    print(f"OPTIMAL THRESHOLD: {best_threshold:.2f}")
    print(f"Balanced Recall: {best_metric:.3f} (equal weight to both classes)")
    print("="*70)
    
    preds_best = (probs >= best_threshold).astype(int)
    tp_best = np.sum((preds_best == 1) & (y_val == 1))
    fn_best = np.sum((preds_best == 0) & (y_val == 1))
    fp_best = np.sum((preds_best == 1) & (y_val == 0))
    tn_best = np.sum((preds_best == 0) & (y_val == 0))
    
    recyclable_recall_best = tp_best / (tp_best + fn_best) if (tp_best + fn_best) > 0 else 0
    trash_recall_best = tn_best / (tn_best + fp_best) if (tn_best + fp_best) > 0 else 0
    
    print(f"\nRecyclable Recall: {recyclable_recall_best:.1%} (catches {int(tp_best)} recyclables, misses {int(fn_best)})")
    print(f"Trash Recall: {trash_recall_best:.1%} (catches {int(tn_best)} trash, misses {int(fp_best)})")
    
    # Per-source analysis
    print("\n" + "="*70)
    print("PER-SOURCE PERFORMANCE (on validation set, using best threshold):")
    print("="*70)
    
    unique_sources = np.unique(sources_val)
    source_results = {}
    
    for source in sorted(unique_sources):
        mask = sources_val == source
        y_source = y_val[mask]
        preds_source = preds_best[mask]
        
        tp_src = np.sum((preds_source == 1) & (y_source == 1))
        fn_src = np.sum((preds_source == 0) & (y_source == 1))
        fp_src = np.sum((preds_source == 1) & (y_source == 0))
        tn_src = np.sum((preds_source == 0) & (y_source == 0))
        
        if y_source[0] == 1:  # Recyclable source
            recall_src = tp_src / (tp_src + fn_src) if (tp_src + fn_src) > 0 else 0
            fn_count = fn_src
            print(f"{source:25s} | Recall: {recall_src:.1%} | FN: {int(fn_count):3d}/{int(tp_src + fn_src):3d}")
        else:  # Trash source
            recall_src = tn_src / (tn_src + fp_src) if (tn_src + fp_src) > 0 else 0
            fp_count = fp_src
            print(f"{source:25s} | Recall: {recall_src:.1%} | FP: {int(fp_count):3d}/{int(tn_src + fp_src):3d}")
        
        source_results[source] = {
            'recall': float(recall_src),
            'tp': int(tp_src),
            'fn': int(fn_src),
            'fp': int(fp_src),
            'tn': int(tn_src)
        }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RUNS_DIR / f"stage2_threshold_tuning_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'best_threshold': float(best_threshold),
        'best_metric': 'balanced_recall',
        'metric_value': float(best_metric),
        'model': model_path.name,
        'validation_set_size': len(X_val),
        'threshold_sweep': threshold_results,
        'per_source': source_results,
        'interpretation': [
            f'Threshold {best_threshold:.2f} maximizes balanced recall ({best_metric:.3f})',
            f'Recyclable recall: {recyclable_recall_best:.1%}',
            f'Trash recall: {trash_recall_best:.1%}',
            'Use this threshold for final test evaluation'
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Also update stage2_config.json with locked threshold and model name
    config_file = Path(__file__).parent / 'stage2_config.json'
    config = {
        "stage": 2,
        "status": "LOCKED",
        "threshold": float(best_threshold),
        "model_name": model_path.name,
        "reason": f"Optimal threshold from validation sweep - balanced recall {best_metric:.3f}",
        "timestamp_set": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "validation_metrics": {
            "recyclable_recall": float(recyclable_recall_best),
            "trash_recall": float(trash_recall_best),
            "balanced_recall": float(best_metric)
        },
        "notes": [
            "LOCKED: Do not modify after this point",
            f"Model: {model_path.name}",
            f"Threshold: {best_threshold:.2f}",
            "Ready for final test eval: python src/evaluate_stage2_test_honest.py"
        ]
    }
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[OK] Results saved: {results_file.name}")
    print(f"[OK] Config locked: {config_file.name}")
    
    print("\n[OK] THRESHOLD LOCKED")
    print(f"Use threshold={best_threshold:.2f} in production inference")
    print(f"Model: {model_path.name}")
    print("\nNext: python src/evaluate_stage2_test_honest.py")
    print("      to get final metrics on LOCKED test set (run once!)")

if __name__ == "__main__":
    main()

