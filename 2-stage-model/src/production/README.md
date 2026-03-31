# Production Scripts Guide

This folder contains the deployment code for running the SmartTrashBin on Raspberry Pi 5 with TensorFlow Lite models.

## Setup

```bash
cd 2-stage-model
pip install -r requirements.txt
```

## Scripts Overview

### camera_inference_live.py
Real-time camera inference using TFLite models. Processes live camera feed, classifies waste, and outputs predictions.

**Key features:**
- TFLite models (fast: 15-20 FPS on RPi 5)
- Picamera2 or OpenCV camera support
- FPS monitoring
- Optional logging to JSON
- Headless mode for server deployment

**Run (with LCD display):**
```bash
python src/production/camera_inference_live.py \
  --stage1-model runs/tflite/stage1_model.tflite \
  --stage2-model runs/tflite/stage2_model.tflite
```

**Run (headless - no display, console only):**
```bash
python src/production/camera_inference_live.py \
  --stage1-model runs/tflite/stage1_model.tflite \
  --stage2-model runs/tflite/stage2_model.tflite \
  --headless
```

**Output:**
```
[00000] RECYCLABLE | S1:  8.2% | S2: 85.3% | FPS: 16.2
[00001] TRASH      | S1: 12.1% | S2: 38.7% | FPS: 16.5
[00002] REJECT     | S1: 92.8% | S2: N/A  | FPS: 16.1
```

**Options:**
- `--headless`: No display output (console only). Omit this flag to display on LCD/HDMI
- `--log-frames`: Save predictions to JSON in live_inference_logs/
- `--camera-width`, `--camera-height`: Adjust frame size (default 640x480)
- `--fps`: Target frames per second (default 30)



## Models

Both TFLite models in `runs/tflite/`:
- `stage1_model.tflite` (5.2 MB) - Battery detection
- `stage2_model.tflite` (5.3 MB) - Waste classification

**Performance:**
- FPS: 15-20 on Raspberry Pi 5
- Latency: ~50-65ms per frame
- Accuracy: Stage 1 99.8% recall, Stage 2 90% balanced recall

## Testing Without Hardware

Run in dry-run mode (simulated) on your development machine:

```bash
# Test camera inference (simulated, no display)
python src/production/camera_inference_live.py \
  --stage1-model runs/tflite/stage1_model.tflite \
  --stage2-model runs/tflite/stage2_model.tflite \
  --headless
```

The camera script automatically detects hardware and falls back to simulation mode if not available.

**On Raspberry Pi with 7" LCD screen:**
Just remove `--headless` to see the live predictions displayed on the LCD.

## Logging & Debugging

Enable frame logging while displaying on LCD:
```bash
python src/production/camera_inference_live.py \
  --stage1-model runs/tflite/stage1_model.tflite \
  --stage2-model runs/tflite/stage2_model.tflite \
  --log-frames
```

Or log without display output:
```bash
python src/production/camera_inference_live.py \
  --stage1-model runs/tflite/stage1_model.tflite \
  --stage2-model runs/tflite/stage2_model.tflite \
  --headless \
  --log-frames
```

Predictions saved to `live_inference_logs/predictions_TIMESTAMP.json`:
```json
{
  "frame_number": 0,
  "final_decision": "RECYCLABLE",
  "stage1_is_battery": false,
  "stage1_confidence": 0.082,
  "stage2_waste_type": "RECYCLABLE",
  "stage2_confidence": 0.853,
  "output_location": "Chute 1 (Recyclables)",
  "timestamp": "2026-03-30T14:23:45.123456"
}
```

## Performance Tips

1. **For better FPS:** Reduce camera resolution
   ```bash
   --camera-width 320 --camera-height 240 --fps 15
   ```

2. **For headless operation:** Use `--headless` flag (saves CPU)

3. **For logging:** Enable `--log-frames` to debug predictions

4. **Monitor CPU:** Watch performance with `top` command

## Troubleshooting

### Camera not detected
```bash
# Check camera is connected
libcamera-hello --list-cameras
```

### Models not loading
- Verify file paths: `ls -lh runs/tflite/`
- Use absolute paths if needed

### Low FPS
- Reduce resolution with `--camera-width` and `--camera-height`
- Run in headless mode (`--headless`)
- Check for other processes: `top`

## Next Steps

1. Install and test camera inference
2. Validate classification accuracy with real samples
3. Monitor performance and adjust resolution/FPS if needed
