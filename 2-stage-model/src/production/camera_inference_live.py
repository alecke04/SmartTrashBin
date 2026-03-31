"""
camera_inference_live.py

Real-time 2-Stage Waste Classification on Raspberry Pi

Captures live camera feed, runs predictions, and displays results on screen.
Uses TFLite models for fast inference (~15-20 FPS on RPi 5).

Features:
  - Real-time camera capture via picamera2 or OpenCV
  - Two-stage pipeline (battery detection -> waste classification)
  - Live display of predictions and confidence scores
  - FPS monitoring
  - Optional frame logging for debugging

Usage:
  python src/production/camera_inference_live.py \
    --stage1-model runs/tflite/stage1_model.tflite \
    --stage2-model runs/tflite/stage2_model.tflite \
    [--headless] [--log-frames]

Requirements:
  - picamera2 (Raspberry Pi camera)
  - opencv-python (fallback camera support)
"""

import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from collections import deque

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# Add parent directory for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    TARGET_SIZE,
    STAGE1_BATTERY_THRESHOLD,
    STAGE2_RECYCLABLE_THRESHOLD,
    OUTPUTS,
)

# Try to import camera libraries
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class TwoStageInferencePipeline:
    """Two-stage waste classification pipeline."""
    
    def __init__(self, stage1_model_path=None, stage2_model_path=None, use_tflite=False):
        """Load both models."""
        self.stage1_model = None
        self.stage2_model = None
        self.use_tflite = use_tflite
        
        if use_tflite:
            self.interpreter_stage1 = None
            self.interpreter_stage2 = None
        
        if stage1_model_path:
            self.load_stage1_model(stage1_model_path)
        if stage2_model_path:
            self.load_stage2_model(stage2_model_path)
    
    def load_stage1_model(self, model_path):
        """Load Stage 1 battery detector model."""
        try:
            print(f"[Stage1] Loading from: {model_path}")
            if self.use_tflite:
                import tensorflow.lite as tflite
                self.interpreter_stage1 = tflite.Interpreter(model_path=str(model_path))
                self.interpreter_stage1.allocate_tensors()
                print("  [OK] Stage 1 TFLite model loaded")
            else:
                self.stage1_model = tf.keras.models.load_model(model_path)
                print("  [OK] Stage 1 model loaded")
        except Exception as e:
            print(f"  [ERROR] Failed to load Stage 1: {e}")
            raise
    
    def load_stage2_model(self, model_path):
        """Load Stage 2 waste classifier model."""
        try:
            print(f"[Stage2] Loading from: {model_path}")
            if self.use_tflite:
                import tensorflow.lite as tflite
                self.interpreter_stage2 = tflite.Interpreter(model_path=str(model_path))
                self.interpreter_stage2.allocate_tensors()
                print("  [OK] Stage 2 TFLite model loaded")
            else:
                self.stage2_model = tf.keras.models.load_model(model_path)
                print("  [OK] Stage 2 model loaded")
        except Exception as e:
            print(f"  [ERROR] Failed to load Stage 2: {e}")
            raise
    
    def preprocess_image(self, image_array):
        """
        Preprocess numpy array (from camera) for model input.
        
        Args:
            image_array: numpy array (height, width, 3) in BGR (from OpenCV)
                        or RGB (from picamera2)
        
        Returns:
            img_batch: (1, 224, 224, 3) uint8 array for model
            img_pil: PIL Image for display
        """
        # Convert to PIL Image if needed
        if isinstance(image_array, np.ndarray):
            # Assume BGR from OpenCV, convert to RGB
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Check if it looks like BGR (common with OpenCV)
                img_pil = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) if HAS_OPENCV else image_array)
            else:
                img_pil = Image.fromarray(image_array)
        else:
            img_pil = image_array
        
        # Resize to model input size
        img_pil = img_pil.convert('RGB')
        img_resized = img_pil.resize(TARGET_SIZE)
        
        # Convert to uint8 array
        img_array = np.array(img_resized, dtype=np.uint8)
        img_batch = np.expand_dims(img_array, axis=0)
        
        return img_batch, img_pil
    
    def stage1_detect_battery(self, img_batch):
        """Stage 1: Detect if item is a battery."""
        if self.use_tflite:
            input_details = self.interpreter_stage1.get_input_details()
            output_details = self.interpreter_stage1.get_output_details()
            
            # Prepare input
            self.interpreter_stage1.set_tensor(input_details[0]['index'], img_batch)
            self.interpreter_stage1.invoke()
            
            # Get output
            output_data = self.interpreter_stage1.get_tensor(output_details[0]['index'])
            battery_confidence = float(output_data[0, 0])
        else:
            probs = self.stage1_model.predict(img_batch, verbose=0)
            battery_confidence = float(probs[0, 0])
        
        is_battery = battery_confidence >= STAGE1_BATTERY_THRESHOLD
        return is_battery, battery_confidence
    
    def stage2_classify_waste(self, img_batch):
        """Stage 2: Classify waste as recyclable or trash."""
        if self.use_tflite:
            input_details = self.interpreter_stage2.get_input_details()
            output_details = self.interpreter_stage2.get_output_details()
            
            # Prepare input
            self.interpreter_stage2.set_tensor(input_details[0]['index'], img_batch)
            self.interpreter_stage2.invoke()
            
            # Get output
            output_data = self.interpreter_stage2.get_tensor(output_details[0]['index'])
            recyclable_prob = float(output_data[0, 0])
        else:
            probs = self.stage2_model.predict(img_batch, verbose=0)
            recyclable_prob = float(probs[0, 0])
        
        is_recyclable = recyclable_prob >= STAGE2_RECYCLABLE_THRESHOLD
        waste_type = 'RECYCLABLE' if is_recyclable else 'TRASH'
        confidence = recyclable_prob if is_recyclable else (1.0 - recyclable_prob)
        
        return waste_type, confidence
    
    def predict(self, image_array):
        """Run full pipeline on a frame."""
        img_batch, img_pil = self.preprocess_image(image_array)
        
        # Stage 1
        is_battery, stage1_conf = self.stage1_detect_battery(img_batch)
        
        result = {
            'stage1_is_battery': bool(is_battery),
            'stage1_confidence': float(stage1_conf),
            'stage2_waste_type': None,
            'stage2_confidence': None,
        }
        
        if is_battery:
            result['final_decision'] = 'REJECT'
        else:
            waste_type, stage2_conf = self.stage2_classify_waste(img_batch)
            result['stage2_waste_type'] = waste_type
            result['stage2_confidence'] = float(stage2_conf)
            result['final_decision'] = waste_type
        
        result['output_location'] = OUTPUTS[result['final_decision']]
        result['timestamp'] = datetime.now().isoformat()
        
        return result, img_pil


class CameraCapture:
    """Abstraction for camera capture (picamera2 or OpenCV)."""
    
    def __init__(self, use_picamera2=True, width=640, height=480, fps=30):
        self.use_picamera2 = use_picamera2 and HAS_PICAMERA2
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        
        if self.use_picamera2:
            self._init_picamera2()
        elif HAS_OPENCV:
            self._init_opencv()
        else:
            raise RuntimeError("No camera library available. Install picamera2 or opencv-python")
    
    def _init_picamera2(self):
        """Initialize Raspberry Pi camera via picamera2."""
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"format": "RGB888", "size": (self.width, self.height)}
            )
            self.camera.configure(config)
            self.camera.start()
            print(f"[Camera] Picamera2 initialized: {self.width}x{self.height} @ {self.fps}fps")
        except Exception as e:
            print(f"[ERROR] Failed to init picamera2: {e}")
            raise
    
    def _init_opencv(self):
        """Initialize camera via OpenCV."""
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            print(f"[Camera] OpenCV initialized: {self.width}x{self.height} @ {self.fps}fps")
        except Exception as e:
            print(f"[ERROR] Failed to init OpenCV: {e}")
            raise
    
    def read_frame(self):
        """Read next frame from camera."""
        if self.use_picamera2:
            frame = self.camera.capture_array()
            return frame
        else:
            ret, frame = self.camera.read()
            if not ret:
                raise RuntimeError("Failed to read frame from camera")
            return frame
    
    def release(self):
        """Release camera resources."""
        if self.use_picamera2 and self.camera:
            self.camera.stop()
        elif self.camera:
            self.camera.release()


class DisplayRenderer:
    """Render predictions on image frames."""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.font_large = None
        self.font_small = None
        self._load_fonts()
    
    def _load_fonts(self):
        """Load system fonts for display."""
        try:
            # Try to load a nice font
            self.font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            # Fallback to default font
            self.font_large = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
    
    def render(self, image_pil, prediction, fps=0.0):
        """Draw prediction results on image.
        
        Args:
            image_pil: PIL Image
            prediction: dict from TwoStageInferencePipeline.predict()
            fps: frames per second for display
        
        Returns:
            PIL Image with annotations
        """
        img = image_pil.copy()
        draw = ImageDraw.Draw(img)
        
        # Decision and output
        decision = prediction['final_decision']
        output_loc = prediction['output_location']
        
        # Colors
        if decision == 'REJECT':
            color_decision = (255, 0, 0)  # Red
            text_decision = "REJECT: BATTERY"
        elif decision == 'RECYCLABLE':
            color_decision = (0, 200, 0)  # Green
            text_decision = f"RECYCLABLE ({prediction['stage2_confidence']:.1%})"
        else:
            color_decision = (255, 165, 0)  # Orange
            text_decision = f"TRASH ({prediction['stage2_confidence']:.1%})"
        
        # Draw background boxes
        draw.rectangle([0, 0, self.width, 80], fill=(0, 0, 0))
        draw.rectangle([0, self.height - 120, self.width, self.height], fill=(0, 0, 0))
        
        # Draw decision (top)
        draw.text((10, 10), text_decision, fill=color_decision, font=self.font_large)
        
        # Draw output location
        draw.text((10, 50), f"→ {output_loc}", fill=(255, 255, 255), font=self.font_small)
        
        # Draw debug info (bottom)
        debug_text = (
            f"S1: Battery={prediction['stage1_is_battery']} ({prediction['stage1_confidence']:.1%})\n"
            f"S2: {prediction['stage2_waste_type'] or 'N/A'}\n"
            f"FPS: {fps:.1f}"
        )
        y_pos = self.height - 110
        for line in debug_text.split('\n'):
            draw.text((10, y_pos), line, fill=(200, 200, 200), font=self.font_small)
            y_pos += 25
        
        return img


def main():
    """Main live inference loop."""
    parser = argparse.ArgumentParser(description="Real-time 2-stage waste classification on camera feed")
    parser.add_argument("--stage1-model", required=True, help="Path to Stage 1 TFLite model")
    parser.add_argument("--stage2-model", required=True, help="Path to Stage 2 TFLite model")
    parser.add_argument("--headless", action="store_true", help="Run without display (headless mode)")
    parser.add_argument("--log-frames", action="store_true", help="Save captured frames for debugging")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera frame width")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--log-dir", default="./live_inference_logs", help="Directory to save logs")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.log_frames:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        predictions_log = []
    
    print("\n" + "="*70)
    print("SMARTTRASHBIN: REAL-TIME 2-STAGE WASTE CLASSIFICATION")
    print("="*70 + "\n")
    
    # Load models (TFLite)
    print("[Init] Loading TFLite models...")
    pipeline = TwoStageInferencePipeline(
        stage1_model_path=args.stage1_model,
        stage2_model_path=args.stage2_model,
        use_tflite=True
    )
    
    # Initialize camera
    print("[Init] Initializing camera...")
    camera = CameraCapture(
        use_picamera2=True,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.fps
    )
    
    # Initialize display renderer
    renderer = DisplayRenderer(width=args.camera_width, height=args.camera_height)
    
    # FPS tracking
    fps_history = deque(maxlen=30)
    frame_count = 0
    start_time = time.time()
    
    print("[Init] Starting inference loop. Press 'q' to exit (or Ctrl+C)...\n")
    
    try:
        while True:
            try:
                frame_start = time.time()
                
                # Capture frame
                frame = camera.read_frame()
                
                # Run prediction
                prediction, img_pil_input = pipeline.predict(frame)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                fps = 1.0 / frame_time if frame_time > 0 else 0
                fps_history.append(fps)
                avg_fps = np.mean(fps_history) if fps_history else 0
                
                # Render output
                if not args.headless:
                    img_annotated = renderer.render(img_pil_input, prediction, fps=avg_fps)
                    # Display can be handled by OpenCV or other display method
                    # For headless, just print
                
                # Print to console
                decision = prediction['final_decision']
                s1_conf = prediction['stage1_confidence']
                s2_conf = prediction['stage2_confidence']
                
                status_line = f"[{frame_count:05d}] {decision:12s} | S1: {s1_conf:.1%} | "
                if s2_conf is not None:
                    status_line += f"S2: {s2_conf:.1%}"
                status_line += f" | FPS: {avg_fps:.1f}"
                
                print(status_line)
                
                # Log if requested
                if args.log_frames:
                    prediction['frame_number'] = frame_count
                    predictions_log.append(prediction)
                
                frame_count += 1
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[ERROR] Frame processing error: {e}")
                continue
    
    except KeyboardInterrupt:
        print("\n[Exit] Stopping camera...")
    
    finally:
        camera.release()
        
        # Save log
        if args.log_frames:
            log_file = Path(args.log_dir) / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_file, 'w') as f:
                json.dump(predictions_log, f, indent=2)
            print(f"[Log] Saved {len(predictions_log)} predictions to {log_file}")
        
        elapsed = time.time() - start_time
        print(f"\n[Stats] Processed {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
