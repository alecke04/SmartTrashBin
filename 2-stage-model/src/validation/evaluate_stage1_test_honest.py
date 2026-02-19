"""
Stage 1 Final Test Evaluation (LOCKED TEST SET)

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
from PIL import Image
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load constants
CONFIG_FILE = Path(__file__).parent / 'stage1_config.json'

def load_stage1_config():
    """Load locked Stage 1 configuration (threshold + other params)"""
    if not CONFIG_FILE.exists():
        print(f"ERROR: Config file not found: {CONFIG_FILE}")
        print("Run threshold tuning first: python src/evaluate_stage1_threshold_tuning.py")
        sys.exit(1)
    
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    return config

def load_images_with_source_labels(split_dir):
    """Load images and track their source"""
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
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
                labels.append(label)
                sources.append(source_name)
                filepaths.append(str(img_path))
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels), np.array(sources), np.array(filepaths)

def main():
    print("="*70)
    print("STAGE 1: FINAL TEST EVALUATION (LOCKED TEST SET)")
    print("="*70)
    print("\nWARNING: This runs ONCE per threshold.")
    print("Do NOT use this to tune anything.")
    print("Do NOT run multiple times.\n")
    
    # Load config with locked threshold
    config = load_stage1_config()
    threshold = config['threshold']
    print(f"Loaded config:")
    print(f"  Threshold: {threshold:.3f} (locked from VAL tuning)")
    print(f"  Reason: {config.get('reason', 'N/A')}")
    
    # Find latest model
    RUNS_DIR = Path('runs')
    stage1_models = sorted(RUNS_DIR.glob("stage1_battery_detector_honest_*.keras"))
    if not stage1_models:
        print("Error: No Stage 1 model found.")
        sys.exit(1)
    
    model_path = stage1_models[-1]
    print(f"\nLoading model: {model_path.name}")
    model = keras.models.load_model(model_path)
    
    # Load TEST set (LOCKED - never touched before)
    print("\nLoading TEST set (locked)...")
    X_test, y_test, sources_test, filepaths_test = load_images_with_source_labels('data/proper_splits/test')
    print(f"TEST set: {len(X_test)} images")
    print(f"  Batteries: {np.sum(y_test)}, Non-batteries: {len(y_test) - np.sum(y_test)}")
    
    # Generate predictions
    print("\nGenerating predictions on TEST set...")
    probs = model.predict(X_test, verbose=0)[:, 0]  # P(battery)
    
    # Apply locked threshold
    preds = (probs >= threshold).astype(int)
    
    # Confusion matrix
    tp = np.sum((preds == 1) & (y_test == 1))
    fn = np.sum((preds == 0) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    tn = np.sum((preds == 0) & (y_test == 0))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / len(y_test)
    
    # Print overall results
    print("\n" + "="*70)
    print("OVERALL TEST SET RESULTS")
    print("="*70)
    print(f"Threshold: {threshold:.3f}")
    print(f"\nBattery Detection:")
    print(f"  TRUE POSITIVES:  {int(tp)} (caught batteries)")
    print(f"  FALSE NEGATIVES: {int(fn)} (MISSED BATTERIES - BAD)")
    print(f"  → Battery Recall: {recall:.1%}")
    print(f"\nNon-Battery Classification:")
    print(f"  TRUE NEGATIVES:  {int(tn)}")
    print(f"  FALSE POSITIVES: {int(fp)} (false alarms)")
    print(f"  → Precision: {precision:.1%}")
    print(f"\nOverall Accuracy: {accuracy:.1%} (reference only)")
    
    # Per-source analysis
    print("\n" + "="*70)
    print("PER-SOURCE ANALYSIS (CRITICAL FOR DOMAIN SHIFT)")
    print("="*70)
    
    unique_sources = np.unique(sources_test)
    source_results = {}
    
    for source in unique_sources:
        mask = sources_test == source
        y_source = y_test[mask]
        preds_source = preds[mask]
        probs_source = probs[mask]
        filepaths_source = filepaths_test[mask]
        
        tp_src = np.sum((preds_source == 1) & (y_source == 1))
        fn_src = np.sum((preds_source == 0) & (y_source == 1))
        fp_src = np.sum((preds_source == 1) & (y_source == 0))
        tn_src = np.sum((preds_source == 0) & (y_source == 0))
        
        recall_src = tp_src / (tp_src + fn_src) if (tp_src + fn_src) > 0 else 0
        precision_src = tp_src / (tp_src + fp_src) if (tp_src + fp_src) > 0 else 0
        
        source_results[source] = {
            'recall': float(recall_src),
            'precision': float(precision_src),
            'tp': int(tp_src),
            'fn': int(fn_src),
            'fp': int(fp_src),
            'tn': int(tn_src),
            'total_batteries': int(tp_src + fn_src)
        }
        
        # Collect false negatives for this source
        false_negative_paths = filepaths_source[(preds_source == 0) & (y_source == 1)]
        
        print(f"\n{source:25s}")
        print(f"  Recall:  {recall_src:.1%} ({int(tp_src)}/{int(tp_src + fn_src)} caught)")
        print(f"  Precision: {precision_src:.1%}")
        print(f"  Missed: {int(fn_src)} FN, {int(fp_src)} FP")
        
        if len(false_negative_paths) > 0:
            print(f"  Top false negatives:")
            for i, fp in enumerate(false_negative_paths[:3], 1):
                print(f"      {i}. {Path(fp).name}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RUNS_DIR / f"stage1_test_results_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'threshold': float(threshold),
        'model': model_path.name,
        'test_set_size': len(X_test),
        'overall': {
            'recall': float(recall),
            'precision': float(precision),
            'accuracy': float(accuracy),
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp),
            'tn': int(tn),
        },
        'per_source': source_results,
        'interpretation': [
            f'Battery recall: {recall:.1%} (caught {int(tp)}/{int(tp+fn)} batteries)',
            f'Missed batteries: {int(fn)} false negatives',
            f'False alarms: {int(fp)} false positives',
            'This is the FINAL locked test metric - do not retrain based on this'
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved: {results_file.name}")
    
    # Summary for reporting
    print("\n" + "="*70)
    print("REPORTING SUMMARY")
    print("="*70)
    print(f"\n📊 Headline metric:")
    print(f"Stage 1 missed {int(fn)} batteries out of {int(tp+fn)} on locked test (recall = {recall:.1%})")
    
    print(f"\n📊 Per-source FN breakdown:")
    for source in sorted(unique_sources):
        if 'battery' in source:
            total_pos = source_results[source]['total_batteries']
            fn_count = source_results[source]['fn']
            print(f"  {source:25s}: {fn_count} FN out of {total_pos}")
    
    print("\nDONE: Test set evaluation complete.")
    print("This is your final honest metric. Do not retrain based on this.")

if __name__ == "__main__":
    main()
