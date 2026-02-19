"""
Train Stage 1 (Binary Battery Detector) with proper data splits.

IMPORTANT:
- Trains ONLY on training split
- Uses val split for early stopping and monitoring (no tuning on test)
- Test set is locked and only used for final evaluation
"""

import sys
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from datetime import datetime
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

# Serializable preprocessing layer (avoids Lambda pickling issues)
class MobileNetV3PreprocessingLayer(keras.layers.Layer):
    """Serializable preprocessing layer for MobileNetV3"""
    def call(self, x):
        """Apply MobileNetV3 preprocessing"""
        return preprocess_input(tf.cast(x, tf.float32))
    
    def get_config(self):
        return super().get_config()

# Constants
TARGET_SIZE = (224, 224)
RUNS_DIR = Path(__file__).parent.parent / 'runs'

# Ensure runs directory exists
RUNS_DIR.mkdir(parents=True, exist_ok=True)

def collect_image_paths_and_labels(split_name):
    """
    Collect image file paths and labels (don't load images yet).
    Returns lists of (path, label) tuples.
    """
    split_dir = Path('data/proper_splits') / split_name
    
    paths_and_labels = []
    
    # Collect battery images (label=1)
    print(f"\n  Collecting battery paths from {split_name}...")
    battery_count = 0
    for battery_class in ['battery_recybat', 'battery_singapore', 'battery_original']:
        battery_folder = split_dir / battery_class
        if battery_folder.exists():
            class_paths = sorted(list(battery_folder.glob('*')))
            paths_and_labels.extend([(str(p), 1) for p in class_paths])
            print(f"    {battery_class}: {len(class_paths)} images")
            battery_count += len(class_paths)
    print(f"  Total batteries: {battery_count}")
    
    # Collect non-battery images (label=0)
    print(f"\n  Collecting non-battery paths from {split_name}...")
    non_battery_folders = [
        'recyclable_glass',
        'recyclable_metal',
        'recyclable_paper',
        'recyclable_plastic',
        'recyclable_cardboard',
        'trash_biological',
        'trash_clothes',
        'trash_shoes',
        'trash_trash',
    ]
    
    non_battery_count = 0
    for non_battery_class in non_battery_folders:
        folder = split_dir / non_battery_class
        if folder.exists():
            class_paths = sorted(list(folder.glob('*')))
            paths_and_labels.extend([(str(p), 0) for p in class_paths])
            print(f"    {non_battery_class}: {len(class_paths)} images")
            non_battery_count += len(class_paths)
    print(f"  Total non-batteries: {non_battery_count}")
    
    # Shuffle
    import random
    random.Random(42).shuffle(paths_and_labels)
    
    return paths_and_labels

def load_and_preprocess_image(path_bytes, label):
    """Load and preprocess a single image (for tf.data.Dataset)
    Returns uint8 images (0-255). Model will apply preprocess_input."""
    path = path_bytes.numpy().decode('utf-8')
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.uint8)
        return img_array, np.int32(label)
    except Exception as e:
        # Return a blank image on error (maintains batch structure)
        return np.zeros((224, 224, 3), dtype=np.uint8), np.int32(label)

def create_train_dataset(paths_and_labels, batch_size=32):
    """Create tf.data.Dataset for training with streaming load (memory efficient)
    
    Args:
        paths_and_labels: list of (path_str, label_int) tuples
        batch_size: batch size for training
    """
    paths = np.array([p for p, _ in paths_and_labels], dtype=object)
    labels = np.array([l for _, l in paths_and_labels], dtype=np.int32)
    
    # Create dataset from paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    # Map with set_shape for proper tensor shape tracking
    def _load_fn(path, label):
        img, lab = tf.py_function(load_and_preprocess_image, [path, label], [tf.uint8, tf.int32])
        img.set_shape((224, 224, 3))
        lab.set_shape(())
        return img, lab
    
    dataset = dataset.map(_load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def load_val_test_data_to_memory(split_name):
    """
    Load val/test data fully into memory (they're small: ~4700 and ~4700 images).
    Returns uint8 (0-255). Model will apply preprocess_input.
    """
    paths_and_labels = collect_image_paths_and_labels(split_name)
    images = []
    labels = []
    
    print(f"\n  Loading {len(paths_and_labels)} images from {split_name}...")
    for idx, (path, label) in enumerate(paths_and_labels):
        if idx % 1000 == 0 and idx > 0:
            print(f"    ... loaded {idx} images so far")
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.uint8)
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            pass  # Skip bad images
    
    print(f"\n  Loaded {len(images)} images successfully")
    return np.array(images), np.array(labels)

def build_stage1_model():
    """Build Stage 1 binary battery detector with MobileNetV3"""
    # Load MobileNetV3Small with ImageNet weights
    base_model = keras.applications.MobileNetV3Small(
        input_shape=(*TARGET_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Build model with serializable preprocessing layer
    inputs = keras.Input(shape=(*TARGET_SIZE, 3), dtype=tf.uint8)
    # Apply MobileNetV3 preprocessing (expects uint8 0-255 input)
    x = MobileNetV3PreprocessingLayer()(inputs)
    x = base_model(x, training=False)  # Start frozen
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary: battery or not
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model, base_model

def main():
    print("="*70)
    print("STAGE 1: BINARY BATTERY DETECTOR TRAINING")
    print("="*70)
    print("Using proper train/val/test splits (no data leakage)")
    print("TRAINING: Streamed from disk in batches (memory efficient)")
    print("VALIDATION: Loaded fully (small, used for early stopping)")
    
    # Create training dataset (streams from disk in batches)
    print("\n[1/3] Creating training dataset (streaming)...")
    # Collect paths once and reuse
    train_paths = collect_image_paths_and_labels('train')
    train_dataset = create_train_dataset(train_paths, batch_size=32)
    
    num_train = len(train_paths)
    num_batteries_train = sum(1 for _, l in train_paths if l == 1)
    num_non_batteries_train = num_train - num_batteries_train
    print(f"Training set: {num_train} images")
    print(f"  Batteries: {num_batteries_train}, Non-batteries: {num_non_batteries_train}")
    print(f"  Imbalance ratio: 1:{num_non_batteries_train/num_batteries_train:.1f}")
    print(f"  [VERIFY] should see battery_recybat + battery_singapore + battery_original above")
    
    # Load validation data (small, keep in memory)
    print("\n[2/3] Loading validation data (for early stopping, NOT tuning)...")
    X_val, y_val = load_val_test_data_to_memory('val')
    print(f"Validation set: {len(X_val)} images")
    num_batt_val = np.sum(y_val)
    num_non_batt_val = len(y_val) - num_batt_val
    print(f"  Batteries: {num_batt_val}, Non-batteries: {num_non_batt_val}")
    print(f"  Imbalance ratio: 1:{num_non_batt_val/num_batt_val:.1f}")
    
    # Calculate class weights: weight minority (battery) class higher
    class_weight = {
        0: num_train / (2.0 * num_non_batteries_train),  # Non-battery weight
        1: num_train / (2.0 * num_batteries_train)       # Battery weight (higher if batteries < non-batteries)
    }
    print(f"\nClass weights: {class_weight}")
    
    # Build model
    print("\n[3/3] Building Stage 1 model (MobileNetV3 + preprocess_input)...")
    model, base_model = build_stage1_model()
    
    # Sanity check: predict on one batch before training
    print("\n" + "="*70)
    print("SANITY CHECK: Predictions before training")
    print("="*70)
    sample_probs = model.predict(X_val[:100], verbose=0)[:, 0]
    battery_mask = y_val[:100] == 1
    non_battery_mask = y_val[:100] == 0
    mean_batt = np.mean(sample_probs[battery_mask]) if np.sum(battery_mask) > 0 else None
    mean_non = np.mean(sample_probs[non_battery_mask]) if np.sum(non_battery_mask) > 0 else None
    
    if mean_batt is not None and mean_non is not None:
        print(f"Mean P(battery) on val batteries: {mean_batt:.4f} (range: {np.min(sample_probs[battery_mask]):.4f}–{np.max(sample_probs[battery_mask]):.4f})")
        print(f"Mean P(battery) on val non-batteries: {mean_non:.4f} (range: {np.min(sample_probs[non_battery_mask]):.4f}–{np.max(sample_probs[non_battery_mask]):.4f})")
        print("(Should be ~0.5 before training; if both are constant ~0.5, model is uninitialized correctly)")
    else:
        print("Warning: First 100 samples missing one class, sanity check skipped")
    
    # Custom callback to detect collapse and monitor separation
    class SanityCheckCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch in [0, 1]:  # After epochs 1 and 2
                probs = self.model.predict(X_val, verbose=0)[:, 0]
                battery_mask = y_val == 1
                non_battery_mask = y_val == 0
                mean_batt = np.mean(probs[battery_mask]) if np.sum(battery_mask) > 0 else 0
                mean_non = np.mean(probs[non_battery_mask]) if np.sum(non_battery_mask) > 0 else 0
                separation = mean_batt - mean_non
                
                # Count TP/FN/FP/TN at threshold 0.5
                preds = (probs >= 0.5).astype(int)
                tp = np.sum((preds == 1) & (y_val == 1))
                fn = np.sum((preds == 0) & (y_val == 1))
                fp = np.sum((preds == 1) & (y_val == 0))
                tn = np.sum((preds == 0) & (y_val == 0))
                
                recall_50 = tp / (tp + fn) if (tp + fn) > 0 else 0
                precision_50 = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                print(f"\nEpoch {epoch + 1} - Separation Check (all {len(y_val)} val images):")
                print(f"   Mean P(battery) on batteries: {mean_batt:.4f}")
                print(f"   Mean P(battery) on non-batteries: {mean_non:.4f}")
                print(f"   Separation (should grow): {separation:.4f}")
                print(f"   At threshold 0.5: Recall={recall_50:.3f}, Precision={precision_50:.3f}")
                print(f"   Confusion: TP={int(tp)}, FN={int(fn)}, FP={int(fp)}, TN={int(tn)}")
                
                if separation < 0.02:
                    print(f"   WARNING: Weak separation (only {separation:.4f}), model may not be learning!")
    
    # EarlyStopping callback for Phase 1
    early_stop_phase1 = keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # ModelCheckpoint callback
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = RUNS_DIR / f"stage1_battery_detector_honest_{timestamp}.keras"
    checkpoint = keras.callbacks.ModelCheckpoint(
        str(model_path),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    # =========================================================================
    # PHASE 1: Frozen backbone, head-only training (2 epochs, high LR)
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: HEAD-ONLY TRAINING (backbone frozen)")
    print("="*70)
    
    # Freeze backbone
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # Higher LR for head-only
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.Recall(name='battery_recall'),
            keras.metrics.Precision(name='battery_precision'),
            keras.metrics.AUC(name='auc'),
            'accuracy'
        ]
    )
    
    print("\nModel summary:")
    model.summary()
    
    hist1 = model.fit(
        train_dataset,
        validation_data=(X_val, y_val),
        epochs=2,
        class_weight=class_weight,
        callbacks=[SanityCheckCallback(), early_stop_phase1, checkpoint],
        verbose=1
    )
    
    # =========================================================================
    # PHASE 2: Unfreeze last 50 backbone layers + recompile with lower LR
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 2: BACKBONE FINE-TUNING (last 50 layers unfrozen)")
    print("="*70)
    
    # Freeze all but last 50 layers
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-5),  # Lower LR for fine-tuning
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.Recall(name='battery_recall'),
            keras.metrics.Precision(name='battery_precision'),
            keras.metrics.AUC(name='auc'),
            'accuracy'
        ]
    )
    
    print("Backbone unfrozen, recompiled with LR=3e-5")
    
    # Create fresh EarlyStopping for Phase 2 (avoid callback state carry-over from Phase 1)
    early_stop_phase2 = keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    hist2 = model.fit(
        train_dataset,
        validation_data=(X_val, y_val),
        epochs=50,
        initial_epoch=2,
        class_weight=class_weight,
        callbacks=[SanityCheckCallback(), early_stop_phase2, checkpoint],
        verbose=1
    )
    
    # Combine histories
    history = hist1
    for key in hist1.history:
        if key in hist2.history:
            history.history[key] = hist1.history[key] + hist2.history[key]
    
    # Summary (model already saved via ModelCheckpoint with best weights)
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    if len(history.history['accuracy']) > 0:
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final training battery_recall: {history.history['battery_recall'][-1]:.4f}")
        print(f"Final training battery_precision: {history.history['battery_precision'][-1]:.4f}")
        
        print(f"\nFinal validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"Final validation battery_recall: {history.history['val_battery_recall'][-1]:.4f}")
        print(f"Final validation battery_precision: {history.history['val_battery_precision'][-1]:.4f}")
    
    print("\nIMPORTANT NOTES:")
    print(f"- Class imbalance: {num_non_batteries_train/num_batteries_train:.1f}:1 (waste:battery)")
    print(f"- Class weights applied to handle imbalance")
    print(f"- Battery RECALL is the meaningful safety metric (catch all batteries)")
    print(f"- Next: threshold tune on val set, then test ONCE")
    
    print("\nTEST SET IS STILL LOCKED")
    print("Run: python src/evaluate_stage1_threshold_tuning.py")
    print("   to find optimal threshold on val set")

if __name__ == "__main__":
    main()
