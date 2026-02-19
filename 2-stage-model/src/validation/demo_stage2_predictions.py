"""
Stage 2 Demo: Visualize predictions on 20 random waste images

Shows recyclable vs trash classification with model confidence.
This demo runs AFTER Stage 1 (assumes batteries already filtered out).
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

# Serializable trash recall layer (must match train script)
class TrashRecall(keras.layers.Layer):
    """Custom layer to track trash detection recall"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recall_metric = keras.metrics.Recall()
    
    def call(self, x):
        return x
    
    def get_config(self):
        return super().get_config()

# Constants
TARGET_SIZE = (224, 224)
RUNS_DIR = Path(__file__).parent.parent.parent / 'runs'
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_all_waste_test_images_with_labels():
    """Load all waste test images (recycable + trash, no batteries)"""
    images = []
    labels = []
    sources = []
    filepaths = []
    
    # Only waste categories (no batteries - those are filtered by Stage 1)
    source_configs = [
        ('recyclable_glass', 1),
        ('recyclable_metal', 1),
        ('recyclable_paper', 1),
        ('recyclable_plastic', 1),
        ('recyclable_cardboard', 1),
        ('trash_biological', 0),
        ('trash_clothes', 0),
        ('trash_shoes', 0),
        ('trash_trash', 0),
    ]
    
    for source_name, label in source_configs:
        source_folder = Path(__file__).parent.parent.parent / 'data' / 'proper_splits' / 'test' / source_name
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
    print("STAGE 2 DEMO: 20 RANDOM WASTE CLASSIFICATION PREDICTIONS")
    print("="*70)
    print("\n[NOTE] This demo assumes Stage 1 already filtered out batteries.")
    print("[NOTE] Input images are waste only (recyclable + trash).\n")
    
    # Load model
    stage2_models = sorted(RUNS_DIR.glob("stage2_waste_classifier_honest_*.keras"))
    if not stage2_models:
        print("Error: No Stage 2 model found.")
        sys.exit(1)
    
    model_path = stage2_models[-1]
    print(f"Loading model: {model_path.name}")
    model = keras.models.load_model(
        model_path,
        custom_objects={
            'MobileNetV3PreprocessingLayer': MobileNetV3PreprocessingLayer,
            'TrashRecall': TrashRecall
        }
    )
    
    # Load all waste test images
    print("\nLoading waste test set...")
    X_test, y_test, sources_test, filepaths_test = load_all_waste_test_images_with_labels()
    print(f"Waste test set: {len(X_test)} images")
    
    # Separate recyclable and trash
    recyclable_indices = np.where(y_test == 1)[0]
    trash_indices = np.where(y_test == 0)[0]
    
    print(f"  Recyclable: {len(recyclable_indices)}")
    print(f"  Trash: {len(trash_indices)}")
    
    # Sample: 10 recyclable, 10 trash
    sample_recyclable_indices = np.random.choice(
        recyclable_indices,
        size=min(10, len(recyclable_indices)),
        replace=False
    )
    sample_trash_indices = np.random.choice(
        trash_indices,
        size=min(10, len(trash_indices)),
        replace=False
    )
    
    sample_indices = np.concatenate([sample_recyclable_indices, sample_trash_indices])
    np.random.shuffle(sample_indices)
    
    # Get predictions
    print("\nGenerating predictions...")
    # Load threshold from config
    config_path = Path(__file__).parent.parent.parent / 'src' / 'production' / 'stage2_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    threshold = config['threshold']
    
    X_sample = X_test[sample_indices]
    probs = model.predict(X_sample, verbose=0)[:, 0]  # P(recyclable)
    preds = (probs >= threshold).astype(int)
    
    # Display results
    print("\n" + "="*70)
    print(f"PREDICTIONS (using threshold={threshold})")
    print("="*70)
    
    predictions = []
    
    # Create figure - 4 rows x 5 columns = 20 images
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_indices, 1):
        true_label = y_test[idx]
        pred_label = preds[i-1]
        recyclable_confidence = probs[i-1]
        source = sources_test[idx]
        filepath = filepaths_test[idx]
        filename = Path(filepath).name
        
        true_class = "RECYCLABLE" if true_label == 1 else "TRASH"
        pred_class = "RECYCLABLE" if pred_label == 1 else "TRASH"
        
        # Show confidence in predicted class
        if pred_label == 1:
            confidence_pct = 100.0 * recyclable_confidence
        else:
            confidence_pct = 100.0 * (1 - recyclable_confidence)
        
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
            'recyclable_probability': float(recyclable_confidence),
            'confidence_pct': float(confidence_pct),
            'correct': int(pred_label == true_label)
        })
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f"demo_stage2_predictions_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n[OK] Demo results saved: {results_file.name}")
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    correct_count = sum(1 for p in predictions if p['correct'])
    print(f"Correct: {correct_count}/20")
    print(f"Accuracy (on sample): {100*correct_count/20:.0f}%")
    print("\nTo run again: python src/validation/demo_stage2_predictions.py")
    
    # Show images
    plt.suptitle(
        f"Stage 2 Waste Classifier Demo - 20 Random Waste Images (Threshold={threshold})",
        fontsize=16,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
