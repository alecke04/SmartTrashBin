"""
inference_2stage.py

Two-Stage Waste Classification Inference

Loads trained models and runs end-to-end prediction pipeline:
  Stage 1: Is it a battery? → REJECT or continue
  Stage 2: Is it recyclable? → RECYCLABLE or TRASH

Usage:
  python src/inference_2stage.py <image_path> [--stage1-model PATH] [--stage2-model PATH]
  python src/inference_2stage.py --batch <folder_path> [--stage1-model PATH] [--stage2-model PATH]
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import random

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory (2-stage-model root) to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TARGET_SIZE,
    RUNS_DIR,
    IMG_EXTS,
    STAGE1_BATTERY_THRESHOLD,
    STAGE2_RECYCLABLE_THRESHOLD,
    OUTPUTS,
)


class TwoStageInferencePipeline:
    """
    Two-stage waste classification pipeline.
    
    Stage 1: Detects batteries (binary classification)
    Stage 2: Classifies waste as recyclable or trash
    """
    
    def __init__(self, stage1_model_path=None, stage2_model_path=None):
        """Load both models."""
        self.stage1_model = None
        self.stage2_model = None
        
        if stage1_model_path:
            self.load_stage1_model(stage1_model_path)
        if stage2_model_path:
            self.load_stage2_model(stage2_model_path)
    
    def load_stage1_model(self, model_path):
        """Load Stage 1 battery detector model."""
        try:
            print(f"Loading Stage 1 model: {model_path}")
            self.stage1_model = tf.keras.models.load_model(model_path)
            print("  [OK] Stage 1 model loaded")
        except Exception as e:
            print(f"  [ERROR] Failed to load Stage 1 model: {e}")
            raise
    
    def load_stage2_model(self, model_path):
        """Load Stage 2 waste classifier model."""
        try:
            print(f"Loading Stage 2 model: {model_path}")
            self.stage2_model = tf.keras.models.load_model(model_path)
            print("  [OK] Stage 2 model loaded")
        except Exception as e:
            print(f"  [ERROR] Failed to load Stage 2 model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """Load and preprocess image for model input.
        
        Note: Load as uint8 [0-255]. preprocess_input layers in models handle normalization.
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize(TARGET_SIZE)
        img_array = np.array(img, dtype=np.uint8)  # Raw 0-255 uint8, models contain preprocess_input layers
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch, img
    
    def stage1_detect_battery(self, img_batch):
        """
        Stage 1: Detect if item is a battery.
        
        Returns:
            (is_battery: bool, confidence: float)
        """
        if self.stage1_model is None:
            raise ValueError("Stage 1 model not loaded")
        
        probs = self.stage1_model.predict(img_batch, verbose=0)
        battery_confidence = float(probs[0, 0])
        is_battery = battery_confidence >= STAGE1_BATTERY_THRESHOLD
        
        return is_battery, battery_confidence
    
    def stage2_classify_waste(self, img_batch):
        """
        Stage 2: Classify waste as recyclable or trash.
        
        Model outputs P(recyclable) via sigmoid (recyclable=1, trash=0).
        
        Returns:
            (waste_type: str, confidence: float)
        """
        if self.stage2_model is None:
            raise ValueError("Stage 2 model not loaded")
        
        probs = self.stage2_model.predict(img_batch, verbose=0)
        recyclable_prob = float(probs[0, 0])  # P(recyclable) from sigmoid
        
        # Classify based on threshold
        # if P(recyclable) >= threshold → RECYCLABLE, else TRASH
        is_recyclable = recyclable_prob >= STAGE2_RECYCLABLE_THRESHOLD
        waste_type = 'RECYCLABLE' if is_recyclable else 'TRASH'
        confidence = recyclable_prob if is_recyclable else (1.0 - recyclable_prob)
        
        return waste_type, confidence
    
    def predict(self, image_path):
        """
        Run full 2-stage pipeline on a single image.
        
        Returns:
            {
                'image': str (path),
                'final_decision': str (RECYCLABLE | TRASH | REJECT),
                'stage1_is_battery': bool,
                'stage1_confidence': float,
                'stage2_waste_type': str or None,
                'stage2_confidence': float or None,
                'output_location': str,
            }
        """
        # Preprocess
        img_batch, img_display = self.preprocess_image(image_path)
        
        # Stage 1: Battery detection
        is_battery, stage1_conf = self.stage1_detect_battery(img_batch)
        
        result = {
            'image': str(image_path),
            'stage1_is_battery': bool(is_battery),
            'stage1_confidence': float(stage1_conf),
            'stage2_waste_type': None,
            'stage2_confidence': None,
        }
        
        if is_battery:
            result['final_decision'] = 'REJECT'
        else:
            # Stage 2: Waste classification
            waste_type, stage2_conf = self.stage2_classify_waste(img_batch)
            result['stage2_waste_type'] = waste_type
            result['stage2_confidence'] = float(stage2_conf)
            result['final_decision'] = waste_type
        
        result['output_location'] = OUTPUTS[result['final_decision']]
        return result
    
    def predict_batch(self, folder_path, output_json=None):
        """
        Run pipeline on all images in a folder.
        
        Args:
            folder_path: Path to folder with images
            output_json: Optional path to save results as JSON
        
        Returns:
            List of prediction results
        """
        folder = Path(folder_path)
        results = []
        
        # Collect all images
        image_files = []
        for ext in IMG_EXTS:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
        
        image_files = sorted(image_files)
        print(f"\nProcessing {len(image_files)} images...\n")
        
        for i, img_path in enumerate(image_files, 1):
            try:
                result = self.predict(img_path)
                results.append(result)
                
                decision = result['final_decision']
                s1_conf = result['stage1_confidence']
                s2_conf = result['stage2_confidence']
                
                if result['stage1_is_battery']:
                    print(f"[{i:3d}] REJECT (Battery detected, conf={s1_conf:.3f}) | {img_path.name}")
                else:
                    print(f"[{i:3d}] {decision:12s} (conf={s2_conf:.3f}) | {img_path.name}")
            
            except Exception as e:
                print(f"[{i:3d}] ERROR: {e} | {img_path.name}")
                results.append({
                    'image': str(img_path),
                    'error': str(e)
                })
        
        # Summary
        recyclable_count = sum(1 for r in results if r.get('final_decision') == 'RECYCLABLE')
        trash_count = sum(1 for r in results if r.get('final_decision') == 'TRASH')
        reject_count = sum(1 for r in results if r.get('final_decision') == 'REJECT')
        error_count = sum(1 for r in results if 'error' in r)
        
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  Recyclable: {recyclable_count}")
        print(f"  Trash: {trash_count}")
        print(f"  Reject (Battery): {reject_count}")
        print(f"  Errors: {error_count}")
        print(f"  Total: {len(results)}")
        
        # Save results
        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved: {output_path}")
        
        return results
    
    def display_results_grid(self, image_results, title="2-Stage Model Predictions", rows=3, cols=10):
        """Display prediction results in a matplotlib grid like verifying_pictures_improved.py.
        
        Shuffles results to show mixed predictions (not all batteries, then all recyclables, etc)
        """
        # Shuffle results for mixed display
        shuffled = image_results.copy()
        random.shuffle(shuffled)
        
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        num_display = min(len(shuffled), rows * cols)
        
        for idx in range(num_display):
            ax = fig.add_subplot(rows, cols, idx + 1)
            result = shuffled[idx]
            
            try:
                # Load and display image
                img = Image.open(result['image']).convert('RGB')
                ax.imshow(img)
                
                # Create title with results
                if result.get('stage1_is_battery'):
                    pred_text = "BATTERY"
                    decision = "REJECT"
                    color = 'red'
                else:
                    pred_text = result.get('stage2_waste_type', 'UNKNOWN')
                    decision = result.get('final_decision', 'UNKNOWN')
                    color = 'green' if decision == 'RECYCLABLE' else 'orange'
                
                title_text = f"Pred: {pred_text}\n{decision}"
                conf = result.get('stage1_confidence' if result.get('stage1_is_battery') else 'stage2_confidence', 0)
                conf_text = f"\n(conf={conf:.2f})"
                
                ax.set_title(title_text + conf_text, fontsize=9, fontweight='bold', color=color)
                ax.axis('off')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}', ha='center', va='center')
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()


def find_latest_models():
    """Find latest trained models in runs directory."""
    stage1_models = sorted(RUNS_DIR.glob("stage1_battery_detector_*.keras"))
    stage2_models = sorted(RUNS_DIR.glob("stage2_waste_classifier_*.keras"))
    
    stage1_path = stage1_models[-1] if stage1_models else None
    stage2_path = stage2_models[-1] if stage2_models else None
    
    return stage1_path, stage2_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2-stage waste classification inference")
    parser.add_argument('input', nargs='?', help='Image file or folder path')
    parser.add_argument('--batch', action='store_true', help='Process entire folder (input is folder path)')
    parser.add_argument('--stage1-model', type=str, help='Path to Stage 1 model')
    parser.add_argument('--stage2-model', type=str, help='Path to Stage 2 model')
    parser.add_argument('--output-json', type=str, help='Path to save results as JSON')
    
    args = parser.parse_args()
    
    # Find models
    stage1_path, stage2_path = find_latest_models()
    
    if args.stage1_model:
        stage1_path = args.stage1_model
    if args.stage2_model:
        stage2_path = args.stage2_model
    
    if not stage1_path or not stage2_path:
        print("Error: Could not find trained models")
        print("Please train models first with: python src/train_2stage.py")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = TwoStageInferencePipeline(stage1_path, stage2_path)
    
    if not args.input:
        print("Usage: python src/inference_2stage.py <image_path>")
        print("       python src/inference_2stage.py <folder_path> --batch")
        print("\nOptions:")
        print("  --stage1-model PATH  Custom Stage 1 model path")
        print("  --stage2-model PATH  Custom Stage 2 model path")
        print("  --output-json PATH   Save batch results as JSON")
        sys.exit(1)
    
    input_path = Path(args.input)
    
    # Single image
    if input_path.is_file():
        result = pipeline.predict(input_path)
        print(f"\nImage: {result['image']}")
        print(f"Stage 1 (Battery?): {result['stage1_is_battery']} (conf={result['stage1_confidence']:.3f})")
        if result['stage2_waste_type']:
            print(f"Stage 2 ({result['stage2_waste_type']}): conf={result['stage2_confidence']:.3f}")
        print(f"\nFinal Decision: {result['final_decision']}")
        print(f"Output: {result['output_location']}")
    
    # Batch folder
    elif input_path.is_dir():
        output_json = args.output_json or (
            RUNS_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        results = pipeline.predict_batch(input_path, output_json=output_json)
        
        # Display grid of results
        if results:
            pipeline.display_results_grid(results, title=f"2-Stage Predictions - {input_path.name}")
    
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)
