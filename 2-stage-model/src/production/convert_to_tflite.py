"""
Convert TensorFlow Keras models to TFLite with quantization.

Creates optimized .tflite versions for edge deployment.
No retraining required - just format conversion + quantization.
"""

import tensorflow as tf
from pathlib import Path
import json

# Constants
RUNS_DIR = Path(__file__).parent.parent / 'runs'
OUTPUT_DIR = RUNS_DIR / 'tflite'

def convert_model_to_tflite(keras_model_path, output_path):
    """
    Convert Keras model to TFLite with quantization.
    
    Args:
        keras_model_path: Path to .keras model file
        output_path: Where to save .tflite file
    
    Returns:
        (original_size_mb, tflite_size_mb, compression_ratio)
    """
    print(f"\n[CONVERTING] {Path(keras_model_path).name}")
    print(f"  Loading Keras model...")
    
    # Load the Keras model with custom objects
    custom_objects = {}
    try:
        from tensorflow.keras.layers import Layer
        
        class MobileNetV3PreprocessingLayer(Layer):
            def call(self, x):
                from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
                return preprocess_input(tf.cast(x, tf.float32))
            def get_config(self):
                return super().get_config()
        
        class TrashRecall(tf.keras.metrics.Metric):
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
        
        custom_objects = {
            'MobileNetV3PreprocessingLayer': MobileNetV3PreprocessingLayer,
            'TrashRecall': TrashRecall
        }
    except Exception as e:
        print(f"  Warning: Could not load custom objects: {e}")
    
    model = tf.keras.models.load_model(str(keras_model_path), custom_objects=custom_objects)
    
    # Get original model size
    original_size_mb = Path(keras_model_path).stat().st_size / (1024 * 1024)
    print(f"  Original model size: {original_size_mb:.2f} MB")
    
    # Convert to TFLite with quantization
    print(f"  Converting with quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable quantization (dynamic range quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    compression_ratio = original_size_mb / tflite_size_mb
    
    print(f"  TFLite model size: {tflite_size_mb:.2f} MB")
    print(f"  Compression: {compression_ratio:.1f}x smaller")
    print(f"  ✓ Saved to: {output_path.name}")
    
    return original_size_mb, tflite_size_mb, compression_ratio

def main():
    print("="*70)
    print("CONVERT KERAS MODELS TO TFLITE WITH QUANTIZATION")
    print("="*70)
    
    # Find latest Stage 1 model
    stage1_models = sorted(RUNS_DIR.glob('stage1_battery_detector_honest_*.keras'))
    if not stage1_models:
        print("ERROR: No Stage 1 model found!")
        return
    
    stage1_keras = stage1_models[-1]
    stage1_tflite = OUTPUT_DIR / f"{stage1_keras.stem}_quantized.tflite"
    
    # Find latest Stage 2 model
    stage2_models = sorted(RUNS_DIR.glob('stage2_waste_classifier_honest_*.keras'))
    if not stage2_models:
        print("ERROR: No Stage 2 model found!")
        return
    
    stage2_keras = stage2_models[-1]
    stage2_tflite = OUTPUT_DIR / f"{stage2_keras.stem}_quantized.tflite"
    
    print(f"\nStage 1 model: {stage1_keras.name}")
    print(f"Stage 2 model: {stage2_keras.name}")
    
    # Convert both models
    try:
        print("\n" + "="*70)
        print("STAGE 1 CONVERSION")
        print("="*70)
        s1_orig, s1_tflite_size, s1_ratio = convert_model_to_tflite(stage1_keras, stage1_tflite)
        
        print("\n" + "="*70)
        print("STAGE 2 CONVERSION")
        print("="*70)
        s2_orig, s2_tflite_size, s2_ratio = convert_model_to_tflite(stage2_keras, stage2_tflite)
        
        # Summary
        print("\n" + "="*70)
        print("CONVERSION SUMMARY")
        print("="*70)
        print(f"\nStage 1:")
        print(f"  Original: {s1_orig:.2f} MB")
        print(f"  TFLite:   {s1_tflite_size:.2f} MB")
        print(f"  Ratio:    {s1_ratio:.1f}x smaller")
        
        print(f"\nStage 2:")
        print(f"  Original: {s2_orig:.2f} MB")
        print(f"  TFLite:   {s2_tflite_size:.2f} MB")
        print(f"  Ratio:    {s2_ratio:.1f}x smaller")
        
        total_orig = s1_orig + s2_orig
        total_tflite = s1_tflite_size + s2_tflite_size
        total_ratio = total_orig / total_tflite
        
        print(f"\nTotal:")
        print(f"  Original: {total_orig:.2f} MB")
        print(f"  TFLite:   {total_tflite:.2f} MB")
        print(f"  Ratio:    {total_ratio:.1f}x smaller")
        
        print(f"\n✓ All models converted successfully!")
        print(f"✓ TFLite models saved in: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
