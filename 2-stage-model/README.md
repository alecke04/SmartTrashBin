# 2-Stage Waste Classification Model

A cleaner, two-stage approach to waste classification focusing on safety-first battery rejection.

## Architecture

```
Image Input
    ↓
┌─ STAGE 1: Binary Battery Detector ─┐
│  Q: Is this a battery?             │
│  Output: BATTERY or NOT-BATTERY    │
└─────────────────────────────────────┘
    ↓
    ├─ If BATTERY → REJECT (Hazardous)
    │
    └─ If NOT-BATTERY
        ↓
    ┌─ STAGE 2: Waste Classifier ─────┐
    │  Q: Is it recyclable or trash?  │
    │  Output: RECYCLABLE or TRASH    │
    └─────────────────────────────────┘
        ↓
        ├─ RECYCLABLE → Chute 1
        └─ TRASH → Chute 2
```

## Data Usage

- **Battery**: 3,020 images (full dataset from `data/Battery/`)
- **Recyclable**: 7,613 images (full dataset from `data/RECYCLABLE/`)
- **Trash**: 4,921 images (full dataset from `data/TRASH/`)
- **Total**: 15,554 images

## Training

Install dependencies:
```bash
pip install -r requirements.txt
```

Train both stages:
```bash
python src/train_2stage.py
```

Train individual stages:
```bash
python src/train_2stage.py --stage 1          # Stage 1 only
python src/train_2stage.py --stage 2          # Stage 2 only
```

Customize training:
```bash
python src/train_2stage.py --epochs 50 --batch-size 32
```

Models are saved in `runs/` with timestamps.

## Testing/Inference

Single image:
```bash
python src/inference_2stage.py <image_path>
```

Batch process a folder:
```bash
python src/inference_2stage.py <folder_path> --batch
```

Save results as JSON:
```bash
python src/inference_2stage.py <folder_path> --batch --output-json results.json
```

Specify custom models:
```bash
python src/inference_2stage.py <image_path> \
    --stage1-model path/to/stage1.keras \
    --stage2-model path/to/stage2.keras
```

## Configuration

Edit `config.py` to adjust:
- `STAGE1_BATTERY_THRESHOLD`: Battery detection confidence threshold (default: 0.40 = conservative)
- `STAGE2_RECYCLABLE_THRESHOLD`: Recyclable classification threshold (default: 0.50)
- Epochs, batch sizes, and model paths

## Key Differences vs João's 3-Stage Model

| Aspect | This Model | João's Model |
|--------|-----------|--------------|
| **Stages** | 2 (Battery, then Waste) | 3 (Battery, Type, then Waste) |
| **Battery Type Classification** | Not implemented (all batteries rejected) | Implemented (Alkaline vs Non-alkaline) |
| **Classes** | 3 decision points (REJECT / RECYCLABLE / TRASH) | 10 classes simultaneously |
| **Complexity** | Simpler, faster | More detailed, slower |
| **Safety** | Conservative (all batteries rejected) | Nuanced (some alkaline allowed to trash) |

## Output

Each prediction returns:
```json
{
  "image": "path/to/image.jpg",
  "final_decision": "RECYCLABLE",
  "stage1_is_battery": false,
  "stage1_confidence": 0.95,
  "stage2_waste_type": "RECYCLABLE",
  "stage2_confidence": 0.87,
  "output_location": "Chute 1 (Recyclables)"
}
```

## Notes

- Uses `MobileNetV3Small` for speed and efficiency
- Shared data folder: uses `../data/` (parent directory data)
- No local data subfolder needed
- Models default to safety-first: conservative thresholds for battery detection
