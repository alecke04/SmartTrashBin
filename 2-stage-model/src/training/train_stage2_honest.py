"""
Stage 2: Binary Waste Classifier (RECYCLABLE vs TRASH)

Trains on all non-battery images from proper train/val/test splits.
NOT on "things that passed Stage 1" - that would bias the dataset.

Architecture: MobileNetV3Small with serializable preprocessing layer
Metrics: AUC (separability), per-class recall (track both classes)
Strategy: Streaming pipeline (no in-memory loading) + two-phase training

Data: 19,270 training waste, 4,125 validation waste, 4,139 test waste
Classes: recyclable=1 (glass, metal, paper, plastic, cardboard)
         trash=0 (biological, shoes, clothes, trash)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from PIL import Image
from datetime import datetime
import json

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
RUNS_DIR = Path(__file__).parent.parent / 'runs'
RUNS_DIR.mkdir(parents=True, exist_ok=True)

DATA_SPLIT_PATH = Path(__file__).parent.parent / 'data' / 'proper_splits'
TARGET_SIZE = (224, 224)

# Custom metric: Specificity (Trash Recall) = TN / (TN + FP)
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

def collect_image_paths_and_labels(split_name):
    """Collect waste-only image paths and labels for given split (streaming-friendly)"""
    base_dir = DATA_SPLIT_PATH / split_name
    
    recyclable_classes = [
        'recyclable_glass',
        'recyclable_metal',
        'recyclable_paper',
        'recyclable_plastic',
        'recyclable_cardboard',
    ]
    
    trash_classes = [
        'trash_biological',
        'trash_shoes',
        'trash_clothes',
        'trash_trash',
    ]
    
    image_paths = []
    labels = []
    
    # Load recyclable (label=1)
    for class_name in recyclable_classes:
        class_dir = base_dir / class_name
        if not class_dir.exists():
            print(f"[WARN] Directory not found: {class_dir}")
            continue
        
        for img_path in sorted(class_dir.glob('*')):
            image_paths.append(str(img_path))
            labels.append(1)
    
    # Load trash (label=0)
    for class_name in trash_classes:
        class_dir = base_dir / class_name
        if not class_dir.exists():
            print(f"[WARN] Directory not found: {class_dir}")
            continue
        
        for img_path in sorted(class_dir.glob('*')):
            image_paths.append(str(img_path))
            labels.append(0)
    
    return np.array(image_paths), np.array(labels)

def load_and_preprocess_image(img_path, label):
    """Load and preprocess a single image (for tf.py_function)"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(TARGET_SIZE)
        img_array = np.array(img, dtype=np.uint8)
        return img_array, np.int32(label)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return np.zeros((*TARGET_SIZE, 3), dtype=np.uint8), np.int32(label)

def create_train_dataset(image_paths, labels, batch_size=32):
    """Create streaming tf.data.Dataset for training (no in-memory loading)"""
    def py_func_wrapper(path, label):
        img, lbl = tf.py_function(load_and_preprocess_image, [path, label], [tf.uint8, tf.int32])
        img.set_shape((*TARGET_SIZE, 3))
        lbl.set_shape(())
        return img, lbl
    
    paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((paths_ds, labels_ds))
    
    # Shuffle and map
    dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.map(py_func_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_val_sanity_dataset(image_paths, labels, batch_size=32, max_images=512):
    """Create small sanity check dataset (subset of validation for quick checks)"""
    def py_func_wrapper(path, label):
        img, lbl = tf.py_function(load_and_preprocess_image, [path, label], [tf.uint8, tf.int32])
        img.set_shape((*TARGET_SIZE, 3))
        lbl.set_shape(())
        return img, lbl
    
    # Take first max_images for deterministic subset
    subset_size = min(max_images, len(image_paths))
    subset_paths = image_paths[:subset_size]
    subset_labels = labels[:subset_size]
    
    paths_ds = tf.data.Dataset.from_tensor_slices(subset_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(subset_labels)
    dataset = tf.data.Dataset.zip((paths_ds, labels_ds))
    
    dataset = dataset.map(py_func_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def build_stage2_model():
    """Build Stage 2 model with MobileNetV3 and serializable preprocessing"""
    base_model = keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model._name = "mobilenetv3small_backbone"  # Name for reliable access
    
    # Build model with serializable preprocessing layer
    inputs = keras.Input(shape=(224, 224, 3), dtype=tf.uint8)
    x = MobileNetV3PreprocessingLayer()(inputs)
    x = keras.layers.RandomFlip("horizontal")(x)
    x = keras.layers.RandomRotation(0.1)(x)
    x = keras.layers.RandomZoom(0.1)(x)
    x = keras.layers.RandomBrightness(0.1)(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model, base_model

def main():
    print("="*70)
    print("STAGE 2: WASTE CLASSIFIER (RECYCLABLE vs TRASH)")
    print("="*70)
    print("\nUsing proper train/val/test splits (non-battery only, streaming)")
    print("Classes: recyclable=1 (glass, metal, paper, plastic, cardboard)")
    print("         trash=0 (biological, shoes, clothes, trash)")
    
    # [1/3] Collect image paths (streaming-friendly, no loading yet)
    print("\n[1/3] Collecting image paths...")
    train_paths, train_labels = collect_image_paths_and_labels('train')
    val_paths, val_labels = collect_image_paths_and_labels('val')
    
    recyclable_count = np.sum(train_labels == 1)
    trash_count = np.sum(train_labels == 0)
    
    print(f"Training set: {len(train_paths)} images")
    print(f"  Recyclable: {int(recyclable_count)}")
    print(f"  Trash: {int(trash_count)}")
    
    recyclable_val = np.sum(val_labels == 1)
    trash_val = np.sum(val_labels == 0)
    
    print(f"Validation set: {len(val_paths)} images")
    print(f"  Recyclable: {int(recyclable_val)}")
    print(f"  Trash: {int(trash_val)}")
    
    # Class weights (balanced formula: total / (2 * class_count))
    total = len(train_labels)
    class_weight = {
        0: total / (2 * trash_count),
        1: total / (2 * recyclable_count)
    }
    print(f"\nClass weights (balanced): {class_weight}")
    
    # [2/3] Create streaming datasets
    print("\n[2/3] Creating streaming datasets...")
    train_dataset = create_train_dataset(train_paths, train_labels, batch_size=32)
    val_sanity_dataset = create_val_sanity_dataset(val_paths, val_labels, batch_size=32, max_images=512)
    
    print(f"  Training pipeline: streaming with shuffle")
    print(f"  Validation sanity subset: first 512 images")
    
    # [3/3] Build model
    print("\n[3/3] Building Stage 2 model...")
    model, base_model = build_stage2_model()
    
    print("\nModel summary:")
    model.summary()
    
    # Sanity check on first sanity batch
    print("\n" + "="*70)
    print("SANITY CHECK: Model predictions on first batch (before training)")
    print("="*70)
    for batch_images, batch_labels in val_sanity_dataset.take(1):
        batch_probs = model.predict(batch_images, verbose=0)[:, 0]
        print(f"Sample predictions: min={batch_probs.min():.4f}, max={batch_probs.max():.4f}, mean={batch_probs.mean():.4f}")
        print(f"(Should be ~0.5 before training starts)")
        break
    
    # Callbacks
    class LightSanityCheckCallback(keras.callbacks.Callback):
        """Lightweight sanity check on small subset (not full validation)"""
        def __init__(self, sanity_dataset, first_n_epochs=15):
            super().__init__()
            self.sanity_dataset = sanity_dataset
            self.first_n_epochs = first_n_epochs
        
        def on_epoch_end(self, epoch, logs=None):
            if epoch < self.first_n_epochs:
                # Quick check on first batch only
                for batch_images, batch_labels in self.sanity_dataset.take(1):
                    batch_probs = self.model.predict(batch_images, verbose=0)[:, 0]
                    batch_preds = (batch_probs >= 0.5).astype(int)
                    batch_accuracy = np.mean(batch_preds == batch_labels.numpy())
                    
                    print(f"\nEpoch {epoch + 1} - Sanity (first batch, 512 images):")
                    print(f"   Prediction range: {batch_probs.min():.4f} to {batch_probs.max():.4f}")
                    print(f"   Batch accuracy: {batch_accuracy:.3f}")
    
    # Training
    print("\n" + "="*70)
    print("TRAINING (2-PHASE: FROZEN BACKBONE -> FINE-TUNE)")
    print("="*70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = RUNS_DIR / f"stage2_waste_classifier_honest_{timestamp}.keras"
    
    # PHASE 1: Train head only (backbone frozen, 2 epochs, LR=1e-3)
    print("\n[PHASE 1] Training head only (backbone frozen, epochs 1-2, LR=1e-3)")
    base_model.trainable = False
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Recall(name='recyclable_recall'),
            keras.metrics.Precision(name='recyclable_precision'),
            keras.metrics.AUC(name='auc'),
            TrashRecall(name='trash_recall')
        ]
    )
    
    history_phase1 = model.fit(
        train_dataset,
        validation_data=val_sanity_dataset,
        epochs=2,
        class_weight=class_weight,
        callbacks=[
            LightSanityCheckCallback(val_sanity_dataset, first_n_epochs=15),
            keras.callbacks.ModelCheckpoint(
                str(model_path),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ],
        verbose=1
    )
    
    # PHASE 2: Fine-tune last 50 backbone layers (unfrozen, epochs 3-50, LR=3e-5)
    print("\n[PHASE 2] Fine-tuning last 50 backbone layers (epochs 3-50, LR=3e-5)")
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    # IMPORTANT: Recompile with new LR + new trainable state
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-5),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Recall(name='recyclable_recall'),
            keras.metrics.Precision(name='recyclable_precision'),
            keras.metrics.AUC(name='auc'),
            TrashRecall(name='trash_recall')
        ]
    )
    
    history_phase2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        initial_epoch=2,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[
            SanityCheckCallback(),
            keras.callbacks.EarlyStopping(
                monitor='val_auc',
                mode='max',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                str(model_path),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ],
        verbose=1
    )
    
    # Use Phase 2 history (final phase) for metrics
    h = history_phase2.history
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE (2-phase: frozen head + fine-tuned backbone)")
    print("="*70)
    
    final_train_acc = h['accuracy'][-1]
    final_train_recall = h['recyclable_recall'][-1]
    final_train_precision = h['recyclable_precision'][-1]
    
    final_val_acc = h['val_accuracy'][-1]
    final_val_recall = h['val_recyclable_recall'][-1]
    final_val_precision = h['val_recyclable_precision'][-1]
    
    print(f"\nFinal training accuracy: {final_train_acc:.4f}")
    print(f"Final training recyclable_recall: {final_train_recall:.4f}")
    print(f"Final training recyclable_precision: {final_train_precision:.4f}")
    
    print(f"\nFinal validation accuracy: {final_val_acc:.4f}")
    print(f"Final validation recyclable_recall: {final_val_recall:.4f}")
    print(f"Final validation recyclable_precision: {final_val_precision:.4f}")
    
    print("\nIMPORTANT NOTES:")
    print("- Training set may not be 50/50 (real-world distribution)")
    print("- Recyclable RECALL is the meaningful metric (minimize false trash)")
    print("- Trash RECALL also matters (minimize false recyclables)")
    print("- Next: threshold tune on val set, then test ONCE")
    
    print("\nTEST SET IS STILL LOCKED")
    print("Run: python src/evaluate_stage2_threshold_tuning.py")
    print("   to find optimal threshold on val set")

if __name__ == "__main__":
    main()
