"""
Full Pipeline Test: Stage 1 (Battery Detection) -> Stage 2 (Waste Classification)

Tests end-to-end: Load image -> Detect battery -> Classify waste
Shows all steps with predictions and confidence scores.
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
import matplotlib.patches as patches
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

# Serializable preprocessing layer (matches both training scripts)
class MobileNetV3PreprocessingLayer(keras.layers.Layer):
    """Serializable preprocessing layer for MobileNetV3"""
    def call(self, x):
        """Apply MobileNetV3 preprocessing"""
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        return preprocess_input(tf.cast(x, tf.float32))
    
    def get_config(self):
        return super().get_config()

# Custom metric for Stage 2
class TrashRecall(keras.metrics.Metric):
    """Compute recall for negative class (trash). TN/(TN+FP)."""
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
CONFIG_DIR = Path(__file__).parent.parent.parent
DATA_SPLIT_PATH = Path(__file__).parent.parent.parent / 'data' / 'proper_splits'

def load_stage1_model():
    """Load Stage 1 battery detection model and locked threshold"""
    # Find latest Stage 1 model
    stage1_models = sorted(RUNS_DIR.glob('stage1_battery_detector_honest_*.keras'))
    if not stage1_models:
        raise FileNotFoundError("No Stage 1 model found in runs/")
    
    model_path = stage1_models[-1]
    print(f"Loading Stage 1 model: {model_path.name}")
    
    model = keras.models.load_model(
        str(model_path),
        custom_objects={'MobileNetV3PreprocessingLayer': MobileNetV3PreprocessingLayer}
    )
    
    # Load locked threshold from config
    config_path = Path(__file__).parent.parent.parent / 'src' / 'production' / 'stage1_config.json'
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        threshold = config.get('threshold', 0.15)
    except FileNotFoundError:
        print(f"Warning: Config not found at {config_path}, using default threshold=0.15")
        threshold = 0.15
    
    return model, model_path, threshold

def load_stage2_model():
    """Load Stage 2 waste classification model and threshold"""
    # Find latest Stage 2 model
    stage2_models = sorted(RUNS_DIR.glob('stage2_waste_classifier_honest_*.keras'))
    if not stage2_models:
        raise FileNotFoundError("No Stage 2 model found in runs/")
    
    model_path = stage2_models[-1]
    print(f"Loading Stage 2 model: {model_path.name}")
    
    model = keras.models.load_model(
        str(model_path),
        custom_objects={
            'MobileNetV3PreprocessingLayer': MobileNetV3PreprocessingLayer,
            'TrashRecall': TrashRecall
        }
    )
    
    # Load locked threshold from config
    config_path = CONFIG_DIR / 'stage2_config.json'
    if not config_path.exists():
        print(f"Warning: Config not found at {config_path}, using default threshold=0.07")
        threshold = 0.07
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
            threshold = config.get('threshold', 0.07)
    
    print(f"Stage 2 threshold (locked): {threshold}")
    
    return model, model_path, threshold

def load_and_preprocess_image(img_path):
    """Load and preprocess image"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(TARGET_SIZE)
        img_array = np.array(img, dtype=np.uint8)
        return img_array
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

def predict_stage1(model, image, threshold):
    """Stage 1: Detect if image contains battery"""
    batch = np.expand_dims(image, axis=0)
    prob = model.predict(batch, verbose=0)[0, 0]
    
    is_battery = prob >= threshold
    confidence = prob if is_battery else (1.0 - prob)
    
    return is_battery, prob, confidence

def predict_stage2(model, image, threshold):
    """Stage 2: Classify waste as recyclable or trash"""
    batch = np.expand_dims(image, axis=0)
    prob = model.predict(batch, verbose=0)[0, 0]
    
    is_recyclable = prob >= threshold
    confidence = prob if is_recyclable else (1.0 - prob)
    
    return is_recyclable, prob, confidence

def collect_test_images(num_batteries=3, num_trash=3, num_recyclable=4):
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
    
    # Collect all available images by category
    for class_name, category in class_configs:
        class_dir = DATA_SPLIT_PATH / 'test' / class_name
        if not class_dir.exists():
            continue
        
        for img_path in sorted(class_dir.glob('*')):
            try:
                img = load_and_preprocess_image(img_path)
                if img is not None:
                    images_data[category].append({
                        'path': str(img_path),
                        'class': class_name,
                        'category': category,
                        'image': img
                    })
            except Exception as e:
                pass
    
    # Randomly select from each category
    selected = []
    
    if len(images_data['battery']) >= num_batteries:
        selected.extend(random.sample(images_data['battery'], num_batteries))
    else:
        selected.extend(images_data['battery'])
        print(f"[WARN] Only {len(images_data['battery'])} batteries available, requested {num_batteries}")
    
    if len(images_data['waste_trash']) >= num_trash:
        selected.extend(random.sample(images_data['waste_trash'], num_trash))
    else:
        selected.extend(images_data['waste_trash'])
        print(f"[WARN] Only {len(images_data['waste_trash'])} trash items available, requested {num_trash}")
    
    if len(images_data['waste_recyclable']) >= num_recyclable:
        selected.extend(random.sample(images_data['waste_recyclable'], num_recyclable))
    else:
        selected.extend(images_data['waste_recyclable'])
        print(f"[WARN] Only {len(images_data['waste_recyclable'])} recyclable items available, requested {num_recyclable}")
    
    # Shuffle selected images
    random.shuffle(selected)
    
    total = len(selected)
    print(f"Selected {total} images: {num_batteries} batteries, {num_recyclable} recyclable, {num_trash} trash")
    
    return selected

def main():
    print("="*70)
    print("FULL PIPELINE TEST: STAGE 1 + STAGE 2 (10 RANDOM IMAGES)")
    print("="*70)
    
    # Load models
    print("\n[LOADING MODELS]")
    stage1_model, stage1_path, stage1_threshold = load_stage1_model()
    stage2_model, stage2_path, stage2_threshold = load_stage2_model()
    
    # Collect test images (random selection each time)
    print("\n[COLLECTING RANDOM TEST IMAGES]")
    test_images = collect_test_images(num_batteries=3, num_trash=3, num_recyclable=4)
    print(f"Total test images: {len(test_images)}\n")
    
    # Run pipeline
    print("="*70)
    print("PIPELINE PREDICTIONS")
    print("="*70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'stage1_model': stage1_path.name,
        'stage1_threshold': float(stage1_threshold),
        'stage2_model': stage2_path.name,
        'stage2_threshold': float(stage2_threshold),
        'predictions': []
    }
    
    battery_correct = 0
    battery_total = 0
    waste_correct = 0
    waste_total = 0
    
    # Prepare visualization
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, img_data in enumerate(test_images):
        img_path = img_data['path']
        true_class = img_data['class']
        true_category = img_data['category']
        image = img_data['image']
        
        print(f"\n[{i+1}/{len(test_images)}] {Path(img_path).name}")
        print(f"  True class: {true_class} ({true_category})")
        
        # Stage 1: Battery detection
        is_battery, stage1_prob, stage1_conf = predict_stage1(stage1_model, image, stage1_threshold)
        stage1_result = "BATTERY" if is_battery else "NON-BATTERY"
        print(f"  Stage 1: {stage1_result} (prob={stage1_prob:.4f}, conf={stage1_conf:.4f})")
        
        # Check Stage 1 correctness
        expected_battery = (true_category == 'battery')
        if is_battery == expected_battery:
            battery_correct += 1
            print(f"  [OK] Stage 1 CORRECT")
        else:
            print(f"  [ERROR] Stage 1 WRONG (expected {'battery' if expected_battery else 'non-battery'})")
        battery_total += 1
        
        # Stage 2: Only run on non-battery waste
        stage2_result = None
        stage2_correct = None
        if not is_battery:
            is_recyclable, stage2_prob, stage2_conf = predict_stage2(stage2_model, image, stage2_threshold)
            stage2_result = "RECYCLABLE" if is_recyclable else "TRASH"
            print(f"  Stage 2: {stage2_result} (prob={stage2_prob:.4f}, conf={stage2_conf:.4f})")
            
            # Check Stage 2 correctness
            expected_recyclable = (true_category == 'waste_recyclable')
            if is_recyclable == expected_recyclable:
                waste_correct += 1
                print(f"  [OK] Stage 2 CORRECT")
            else:
                print(f"  [ERROR] Stage 2 WRONG (expected {'recyclable' if expected_recyclable else 'trash'})")
            waste_total += 1
        else:
            print(f"  Stage 2: SKIPPED (battery detected)")
        
        # Store result
        results['predictions'].append({
            'filename': Path(img_path).name,
            'true_class': true_class,
            'true_category': true_category,
            'stage1_result': stage1_result,
            'stage1_prob': float(stage1_prob),
            'stage1_correct': bool(is_battery == expected_battery),
            'stage2_result': stage2_result if not is_battery else None,
            'stage2_prob': float(stage2_prob) if not is_battery else None,
            'stage2_correct': bool(is_recyclable == expected_recyclable) if not is_battery else None,
        })
        
        # Add to visualization
        ax = axes[i]
        img_pil = Image.fromarray(image)
        ax.imshow(img_pil)
        
        # Prepare title with predictions
        if is_battery:
            stage1_text = f"S1: BATTERY\n({stage1_prob:.2f})"
            stage2_text = ""
            correct = "[OK]" if is_battery == expected_battery else "[ERROR]"
            color = 'green' if is_battery == expected_battery else 'red'
        else:
            stage1_text = f"S1: NON-BATT\n({stage1_prob:.2f})"
            if stage2_result:
                stage2_text = f"\nS2: {stage2_result}\n({stage2_prob:.2f})"
                correct = "[OK]" if (is_recyclable == expected_recyclable) else "[ERROR]"
                color = 'green' if (is_recyclable == expected_recyclable) else 'red'
            else:
                stage2_text = ""
                correct = "?"
                color = 'gray'
        
        title = f"{correct} {true_class}\n{stage1_text}{stage2_text}"
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.axis('off')
    
    # Save visualization to results directory
    plt.tight_layout()
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    viz_path = results_dir / f"pipeline_test_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {viz_path.name}")
    plt.show()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nStage 1 (Battery Detection):")
    print(f"  Correct: {battery_correct}/{battery_total} ({100*battery_correct/battery_total:.1f}%)")
    
    if waste_total > 0:
        print(f"\nStage 2 (Waste Classification, non-battery only):")
        print(f"  Correct: {waste_correct}/{waste_total} ({100*waste_correct/waste_total:.1f}%)")
    
    print(f"\nModels loaded successfully:")
    print(f"  Stage 1: {stage1_path.name}")
    print(f"  Stage 2: {stage2_path.name}")
    
    # Save results
    results_dir = Path(__file__).parent.parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"pipeline_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path.name}")
    
    # Add title to figure
    fig.suptitle(f"Full Pipeline Test - 10 Random Images (Stage 1 + Stage 2)", fontsize=16, fontweight='bold')
    
    return results

if __name__ == "__main__":
    main()
