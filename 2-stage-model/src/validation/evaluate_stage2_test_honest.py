"""
Stage 2 Final Test Evaluation (LOCKED TEST SET)

Run this ONCE after threshold is locked.
Do NOT use this to tune threshold (VAL only).
Do NOT run multiple times to find better threshold.

This is the final honest metric.
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

def load_stage2_config():
    """Load locked Stage 2 configuration (threshold + other params)"""
    config_file = Path(__file__).parent / 'stage2_config.json'
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        print("Run threshold tuning first: python src/evaluate_stage2_threshold_tuning.py")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config

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
    print("STAGE 2: FINAL TEST EVALUATION (LOCKED TEST SET)")
    print("="*70)
    print("\n⚠️  WARNING: This runs ONCE per threshold.")
    print("Do NOT use this to tune anything.")
    print("Do NOT run multiple times.\n")
    
    # Load config with locked threshold
    config = load_stage2_config()
    threshold = config['threshold']
    print(f"Loaded config:")
    print(f"  Threshold: {threshold:.3f} (locked from VAL tuning)")
    print(f"  Reason: {config.get('reason', 'N/A')}")
    
    # Use model name from config (reproducible)
    model_name = config.get('model_name')
    if not model_name:
        print("ERROR: Config does not have model_name. Run threshold tuning first.")
        sys.exit(1)
    
    RUNS_DIR = Path('runs')
    model_path = RUNS_DIR / model_name
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)
    
    print(f"\nLoading model: {model_path.name}")
    model = keras.models.load_model(
        model_path,
        custom_objects={
            'MobileNetV3PreprocessingLayer': MobileNetV3PreprocessingLayer,
            'TrashRecall': TrashRecall
        }
    )
    
    # Load TEST set (LOCKED - never touched before)
    print("\nLoading TEST set (locked)...")
    X_test, y_test, sources_test, filepaths_test = load_images_with_source_labels(
        DATA_SPLIT_PATH / 'test'
    )
    recyclable_count = np.sum(y_test)
    trash_count = len(y_test) - recyclable_count
    
    print(f"TEST set: {len(X_test)} images")
    print(f"  Recyclable: {int(recyclable_count)}, Trash: {int(trash_count)}")
    
    # Generate predictions
    print("\nGenerating predictions on TEST set...")
    probs = model.predict(X_test, verbose=0)[:, 0]  # P(recyclable)
    
    # Apply locked threshold
    preds = (probs >= threshold).astype(int)
    
    # Confusion matrix
    tp = np.sum((preds == 1) & (y_test == 1))
    fn = np.sum((preds == 0) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    tn = np.sum((preds == 0) & (y_test == 0))
    
    recyclable_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    trash_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / len(y_test)
    
    # Print overall results
    print("\n" + "="*70)
    print("OVERALL TEST SET RESULTS")
    print("="*70)
    print(f"Threshold: {threshold:.3f}")
    print(f"\nRecyclable Detection:")
    print(f"  TRUE POSITIVES:  {int(tp)} (caught recyclables)")
    print(f"  FALSE NEGATIVES: {int(fn)} (MISCLASSIFIED AS TRASH)")
    print(f"  → Recyclable Recall: {recyclable_recall:.1%}")
    print(f"\nTrash Detection:")
    print(f"  TRUE NEGATIVES:  {int(tn)} (caught trash correctly)")
    print(f"  FALSE POSITIVES: {int(fp)} (MISCLASSIFIED AS RECYCLABLE - contamination)")
    print(f"  → Trash Recall: {trash_recall:.1%}")
    print(f"\nOverall Accuracy: {accuracy:.1%}")
    
    # Per-source analysis
    print("\n" + "="*70)
    print("PER-SOURCE ANALYSIS")
    print("="*70)
    
    unique_sources = np.unique(sources_test)
    source_results = {}
    
    for source in sorted(unique_sources):
        mask = sources_test == source
        y_source = y_test[mask]
        preds_source = preds[mask]
        
        tp_src = np.sum((preds_source == 1) & (y_source == 1))
        fn_src = np.sum((preds_source == 0) & (y_source == 1))
        fp_src = np.sum((preds_source == 1) & (y_source == 0))
        tn_src = np.sum((preds_source == 0) & (y_source == 0))
        
        source_results[source] = {
            'tp': int(tp_src),
            'fn': int(fn_src),
            'fp': int(fp_src),
            'tn': int(tn_src)
        }
        
        if y_source[0] == 1:  # Recyclable source
            recall_src = tp_src / (tp_src + fn_src) if (tp_src + fn_src) > 0 else 0
            print(f"\n{source:25s}")
            print(f"  Recall: {recall_src:.1%} ({int(tp_src)}/{int(tp_src+fn_src)} caught as recyclable)")
            print(f"  Misclassified: {int(fn_src)} FN (sent to trash by mistake)")
        else:  # Trash source
            recall_src = tn_src / (tn_src + fp_src) if (tn_src + fp_src) > 0 else 0
            print(f"\n{source:25s}")
            print(f"  Recall: {recall_src:.1%} ({int(tn_src)}/{int(tn_src+fp_src)} caught as trash)")
            print(f"  Contamination: {int(fp_src)} FP (sent to recycling by mistake)")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RUNS_DIR / f"stage2_test_results_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'threshold': float(threshold),
        'model': model_path.name,
        'test_set_size': len(X_test),
        'overall': {
            'recyclable_recall': float(recyclable_recall),
            'trash_recall': float(trash_recall),
            'accuracy': float(accuracy),
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp),
            'tn': int(tn),
        },
        'per_source': source_results,
        'interpretation': [
            f'Recyclable recall: {recyclable_recall:.1%} (caught {int(tp)}/{int(tp+fn)} recyclables)',
            f'Trash contamination: {trash_recall:.1%} (missed {int(fp)} trash items)',
            'This is the FINAL locked test metric - do not retrain based on this'
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved: {results_file.name}")
    
    # Summary for reporting
    print("\n" + "="*70)
    print("REPORTING SUMMARY")
    print("="*70)
    print(f"\n📊 Stage 2 Performance:")
    print(f"Recyclable Recall: {recyclable_recall:.1%} ({int(tp)}/{int(tp+fn)})")
    print(f"Trash Recall: {trash_recall:.1%} ({int(tn)}/{int(tn+fp)})")
    
    print(f"\n📊 Per-source breakdown:")
    for source in sorted(unique_sources):
        if 'recyclable' in source:
            tp_src = source_results[source]['tp']
            fn_src = source_results[source]['fn']
            total = tp_src + fn_src
            print(f"  {source:25s}: {int(tp_src)}/{int(total)} caught")
        else:
            tn_src = source_results[source]['tn']
            fp_src = source_results[source]['fp']
            total = tn_src + fp_src
            print(f"  {source:25s}: {int(tn_src)}/{int(total)} correct")
    
    print("\n✅ DONE: Test set evaluation complete.")
    print("This is your final honest metric. Do not retrain based on this.")

if __name__ == "__main__":
    main()
