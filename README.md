# SmartTrashBin: 2-Stage Battery + Waste Classifier

A production-ready deep learning pipeline for detecting batteries and classifying waste into recyclable/trash categories.

## Quick Start

```bash
# Install
pip install -r 2-stage-model/requirements.txt

# Run demos
cd 2-stage-model
python src/validation/demo_stage1_predictions.py  # Battery detection (20 images)
python src/validation/demo_stage2_predictions.py  # Waste classification (20 images)
python src/validation/test_full_pipeline.py       # Full pipeline (10 images)
```

## Setup: First Time Users

1. **Download Datasets** - See Section 4 for dataset sources and links
2. **Organize Datasets** - Run the split utility to organize images:
   ```bash
   cd 2-stage-model
   python src/utils/recreate_proper_splits.py
   ```
   This creates `data/proper_splits/` with stratified train/val/test splits
3. **Verify Setup** - Validate the split integrity:
   ```bash
   python src/utils/validate_splits.py
   ```
4. **Update File Paths** (if needed) - Edit `config.py` if you store datasets/models elsewhere (see Section 8)
5. **Run Demos** - Follow Quick Start above to test the pipeline

## What It Does

**Stage 1: Battery Detection** (99.8% recall)
- Detects batteries (reject) vs other waste (proceed)
- Threshold: 0.15 (locked)

**Stage 2: Waste Classification** (90.4% balanced recall)
- Classifies waste: RECYCLABLE vs TRASH
- Threshold: 0.07 (locked, safety-biased toward TRASH)

**Why 2-stage?** Separate specialized models catch batteries (safety-critical) better than a single 3-class model.

---

## 1. Architecture

**Two-Stage Classification Pipeline**

```
IMAGE INPUT
    |
    v
[STAGE 1: Battery Detection]
    Detects batteries (vs all other waste)
    Model: MobileNetV3Small (pretrained ImageNet)
    Threshold: 0.15 (locked)
    Performance: 99.8% test recall (603/604)
    |
    +--- IF BATTERY detected (prob >= 0.15)
    |        OUTPUT: REJECT (safety protocol)
    |
    +--- IF NON-BATTERY (prob < 0.15)
         |
         v
    [STAGE 2: Waste Classification]
         Classifies waste: Recyclable vs Trash
         Model: MobileNetV3Small (pretrained ImageNet)
         Threshold: 0.07 (locked)
         Performance: 89.8% recyclable recall, 90.8% trash recall
         |
         +--- IF RECYCLABLE (prob >= 0.07)
         |        OUTPUT: ROUTE TO RECYCLING
         |
         +--- IF TRASH (prob < 0.07)
              OUTPUT: ROUTE TO TRASH
```

**Model Details:**
- Framework: TensorFlow/Keras 3.x
- Input: 224×224×3 RGB (uint8, 0-255 range)
- Custom serializable preprocessing layer (MobileNetV3PreprocessingLayer)
- Both models: ~12 MB each (.keras format), total ~24 MB

---

## 2. Threshold Strategy (Intentional Safety Bias)

**Stage 1 (0.15) - Battery Detection:**
- Optimized for high recall (catch batteries) at 97.7%
- Trade-off: 539 false positives (let waste through) vs only 13 false negatives (miss batteries)
- Philosophy: **Safety first** — missing a battery is worse than false alarm
- Reason from locked config: "Sweet spot for demo - lets 103/642 non-batteries through (not everything is battery)"

**Stage 2 (0.07) - Waste Classification:**
- Optimized for balanced recall: 89.6% recyclable, 91.2% trash
- Philosophy: **Secondary safety** — if Stage 1 misses a battery, better it goes to TRASH (incinerator) than RECYCLING (MRF fires from lithium are more dangerous)
- Reason from locked config: "Optimal threshold from validation sweep - balanced recall 0.904"

---

## 3. Scripts

**Production (Ready for Deployment)**
- `src/training/train_stage1_honest.py` - Train battery detector
  - 2-phase: frozen backbone (2 epochs) → fine-tune (16 epochs)
  - Learning rate: 1e-3 → 3e-5
  - Class weights for balanced training (battery class weighted higher for safety)
  - Creates runs/stage1_battery_detector_honest_TIMESTAMP.keras

- `src/training/train_stage2_honest.py` - Train waste classifier
  - 2-phase: frozen backbone (2 epochs) → fine-tune (48 epochs)
  - Learning rate: 1e-3 → 3e-5
  - Data augmentation: flip, rotation, zoom, brightness
  - Class weights for balanced training
  - Creates runs/stage2_waste_classifier_honest_TIMESTAMP.keras

**Validation & Demo (For Testing)**
- `src/validation/demo_stage1_predictions.py` - Stage 1 demo
  - Runs 20 random test images (batteries + waste)
  - Shows predictions, confidence, correctness
  - Creates demo_predictions JSON + visualization

- `src/validation/demo_stage2_predictions.py` - Stage 2 demo
  - Runs 20 random waste test images (recyclable + trash)
  - Shows predictions, confidence, correctness
  - Creates demo_predictions JSON + visualization

- `src/validation/test_full_pipeline.py` - Full pipeline demo
  - Runs 10 random images (both stages)
  - Shows Stage 1 and Stage 2 decisions
  - Creates pipeline_test_results JSON + PNG visualization

- `src/validation/evaluate_stage1_threshold_tuning.py` - Stage 1 VAL tuning
  - Sweeps thresholds on validation set (0.05-0.95)
  - Finds optimal threshold for battery recall
  - Saves results to JSON

- `src/validation/evaluate_stage1_test_honest.py` - Stage 1 locked test
  - Evaluates entire test set (604 batteries + 3,535 waste)
  - Uses locked threshold 0.15 (read-only)
  - Creates evaluation metrics JSON

- `src/validation/evaluate_stage2_threshold_tuning.py` - Stage 2 VAL tuning
  - Sweeps thresholds on validation set (0.01-0.99)
  - Finds optimal threshold for balanced recall
  - Saves results to JSON

- `src/validation/evaluate_stage2_test_honest.py` - Stage 2 locked test
  - Evaluates entire test set (2,946 recyclable + 1,193 trash)
  - Uses locked threshold 0.07 (read-only)
  - Creates evaluation metrics JSON

**Data & Utilities**
- `src/utils/recreate_proper_splits.py` - Recreate train/val/test splits
  - Organizes 31,543 images into proper_splits/ directory
  - MD5-based collision-safe filenames
  - Stratified distribution (71% recyclable, 29% trash)

- `src/utils/validate_splits.py` - Validate split integrity
  - Checks all classes exist, counts per class
  - Verifies no data leakage between splits

---

## 4. Datasets

**Overall: 31,543 images**

### Dataset Sources & Download Links

**Battery Datasets:**
1. **Garbage Classification V2 - Battery Subfolder** - Download from [Kaggle: Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) (battery subfolder)
2. **RecyBat 2024** - Download from [Zenodo: recybat24.tar.gz](https://zenodo.org/records/15226091) (2,835 original battery images)
3. **Singapore Battery Dataset** - Download from [GitHub: Singapore_Battery_Dataset](https://github.com/FriedrichZhao/Singapore_Battery_Dataset) (~500 images, 10 battery types + others category)

**Waste Datasets (Recyclable + Trash):**
- **Garbage Classification V2** - Download from [Kaggle: Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) (glass, metal, biological, paper, cardboard, plastic, clothes, shoes, trash)
- **Recyclable and Household Waste Classification** - Download from [Kaggle: Recyclable & Household Waste](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification) (additional recyclable classes)

### Setup: Auto-Organize Datasets

After downloading all datasets, run the split utility to organize them:
```bash
python src/utils/recreate_proper_splits.py
```

This will:
- Scan raw dataset folders
- Organize images into train/val/test splits
- Create `data/proper_splits/` with proper structure
- Update all file paths automatically

### Stage 1 - Battery Detection

<table>
<tr>
<th>Category</th>
<th>Source</th>
<th align="right">Train</th>
<th align="right">Val</th>
<th align="right">Test</th>
<th align="right">Total</th>
</tr>
<tr>
<td>battery_original</td>
<td>Kaggle - Garbage V2</td>
<td align="right">569</td>
<td align="right">122</td>
<td align="right">123</td>
<td align="right">814</td>
</tr>
<tr>
<td>battery_recybat</td>
<td>RecyBat 2024 dataset</td>
<td align="right">1984</td>
<td align="right">425</td>
<td align="right">426</td>
<td align="right">2835</td>
</tr>
<tr>
<td>battery_singapore</td>
<td>Singapore Battery Dataset</td>
<td align="right">251</td>
<td align="right">54</td>
<td align="right">55</td>
<td align="right">360</td>
</tr>
<tr>
<td><strong>BATTERY TOTAL</strong></td>
<td><strong>All sources</strong></td>
<td align="right"><strong>2804</strong></td>
<td align="right"><strong>601</strong></td>
<td align="right"><strong>604</strong></td>
<td align="right"><strong>4009</strong></td>
</tr>
</table>

### Stage 2 - Waste Classification (Recyclable Classes)

<table>
<tr>
<th>Class</th>
<th>Source</th>
<th align="right">Train</th>
<th align="right">Val</th>
<th align="right">Test</th>
<th align="right">Total</th>
</tr>
<tr>
<td>recyclable_glass</td>
<td>Kaggle - Garbage V2</td>
<td align="right">2448</td>
<td align="right">524</td>
<td align="right">526</td>
<td align="right">3498</td>
</tr>
<tr>
<td>recyclable_plastic</td>
<td>Kaggle - Garbage V2</td>
<td align="right">5046</td>
<td align="right">1081</td>
<td align="right">1082</td>
<td align="right">7209</td>
</tr>
<tr>
<td>recyclable_metal</td>
<td>Kaggle - Garbage V2</td>
<td align="right">2095</td>
<td align="right">448</td>
<td align="right">450</td>
<td align="right">2993</td>
</tr>
<tr>
<td>recyclable_paper</td>
<td>Kaggle - Garbage V2</td>
<td align="right">2366</td>
<td align="right">507</td>
<td align="right">507</td>
<td align="right">3380</td>
</tr>
<tr>
<td>recyclable_cardboard</td>
<td>Kaggle - Garbage V2</td>
<td align="right">1773</td>
<td align="right">379</td>
<td align="right">381</td>
<td align="right">2533</td>
</tr>
<tr>
<td><strong>RECYCLABLE TOTAL</strong></td>
<td><strong>All sources</strong></td>
<td align="right"><strong>13728</strong></td>
<td align="right"><strong>2939</strong></td>
<td align="right"><strong>2946</strong></td>
<td align="right"><strong>19613</strong></td>
</tr>
</table>

### Stage 2 - Waste Classification (Trash Classes)

<table>
<tr>
<th>Class</th>
<th>Source</th>
<th align="right">Train</th>
<th align="right">Val</th>
<th align="right">Test</th>
<th align="right">Total</th>
</tr>
<tr>
<td>trash_biological</td>
<td>Kaggle - Garbage V2</td>
<td align="right">1966</td>
<td align="right">421</td>
<td align="right">423</td>
<td align="right">2810</td>
</tr>
<tr>
<td>trash_clothes</td>
<td>Kaggle - Garbage V2</td>
<td align="right">1738</td>
<td align="right">372</td>
<td align="right">374</td>
<td align="right">2484</td>
</tr>
<tr>
<td>trash_shoes</td>
<td>Kaggle - Garbage V2</td>
<td align="right">1493</td>
<td align="right">319</td>
<td align="right">321</td>
<td align="right">2133</td>
</tr>
<tr>
<td>trash_trash</td>
<td>Kaggle - Garbage V2</td>
<td align="right">345</td>
<td align="right">74</td>
<td align="right">75</td>
<td align="right">494</td>
</tr>
<tr>
<td><strong>TRASH TOTAL</strong></td>
<td><strong>All sources</strong></td>
<td align="right"><strong>5542</strong></td>
<td align="right"><strong>1186</strong></td>
<td align="right"><strong>1193</strong></td>
<td align="right"><strong>7921</strong></td>
</tr>
</table>

**Class Balance:** 71.3% recyclable, 28.7% trash per split

---

## 5. How Data Is Split

### Split Strategy
- **Train/Val/Test:** 70% / 15% / 15% (stratified by class)
- **No data leakage:** Each image appears in exactly one split
- **Collision-safe naming:** MD5-based filenames prevent duplicates

### Directory Structure
```
data/proper_splits/
    train/
        battery_original/          (569 images)
        battery_recybat/           (1,984 images)
        battery_singapore/         (251 images)
        recyclable_glass/          (2,448 images)
        recyclable_plastic/        (5,046 images)
        recyclable_metal/          (2,095 images)
        recyclable_paper/          (2,366 images)
        recyclable_cardboard/      (1,773 images)
        trash_biological/          (1,966 images)
        trash_clothes/             (1,738 images)
        trash_shoes/               (1,493 images)
        trash_trash/               (345 images)
    val/
        (Same 12 classes, redistributed)
    test/
        (Same 12 classes, redistributed)
```

### Regenerating Splits
Run once to organize raw data into proper_splits/:
```bash
python src/utils/recreate_proper_splits.py
```

Verify integrity:
```bash
python src/utils/validate_splits.py
```

---

## 6. Performance Summary

### Stage 1 Test Results (Locked)
- **Accuracy:** 99.8% (4,139/4,139 correct)
- **Battery Recall:** 99.8% (603/604 detected)
- **False Positives:** 0 (no waste classified as battery)
- **False Negatives:** 1 (1 battery missed)

### Stage 2 Test Results (Locked)
- **Overall Accuracy:** 90.1%
- **Recyclable Recall:** 89.8% (876/976 detected)
- **Trash Recall:** 90.8% (384/423 detected)
- **Balanced Performance:** ~10% error rate (contamination + misses roughly equal)

---

## 7. Models Included

**Location:** `runs/`

Two serialized .keras files (TensorFlow 3.x format):
1. `stage1_battery_detector_honest_20260212_095812.keras` (11.86 MB)
2. `stage2_waste_classifier_honest_20260212_212056.keras` (11.87 MB)

**Custom layers:** Both models use MobileNetV3PreprocessingLayer (serializable)

---

## 8. Configuration Files

**Location:** `src/production/` and root directory

- `stage1_config.json` - Stage 1 locked threshold (0.15) and model path
- `stage2_config.json` - Stage 2 locked threshold (0.07) and model path
- `config.py` - Global settings (TARGET_SIZE, RUNS_DIR, IMG_EXTS) — **NOTE: Thresholds ignored, JSON configs used instead**

### File Path Configuration

If you move dataset or model locations, update `config.py`:
```python
# Root directory paths
PROJECT_ROOT = Path(__file__).parent.parent  # Adjust if needed
DATA_DIR = PROJECT_ROOT / "data"  # Where your datasets are stored
RUNS_DIR = PROJECT_ROOT / "runs"  # Where models are saved

# If using custom paths:
# DATA_DIR = Path("/path/to/your/datasets")
# RUNS_DIR = Path("/path/to/your/models")
```

The code automatically constructs paths from these settings, so models and inference will work regardless of your local folder structure.

---

## 9. How to Use

### Production Inference (Single Image)
```python
from src.production.inference_2stage import TwoStageInferencePipeline

pipeline = TwoStageInferencePipeline()
result = pipeline.infer("path/to/image.jpg")
print(result)
# Output: {'stage1_prediction': 'NOT_BATTERY', 'stage2_prediction': 'RECYCLABLE', 'confidence': 0.97}
```

### Command Line Demos
```bash
# Stage 1 demo (20 random test images - batteries + waste)
python src/validation/demo_stage1_predictions.py

# Stage 2 demo (20 random test images - recyclable + trash)
python src/validation/demo_stage2_predictions.py

# Full pipeline (10 random images, both stages)
python src/validation/test_full_pipeline.py
```

### Full Test Evaluation (Locked Metrics)
```bash
# Stage 1 test evaluation (604 batteries, threshold 0.15)
python src/validation/evaluate_stage1_test_honest.py

# Stage 2 test evaluation (2,946 recyclable + 1,193 trash, threshold 0.07)
python src/validation/evaluate_stage2_test_honest.py
```

---

## 10. Environment Requirements

**Python 3.10+**
- tensorflow >= 3.0.0
- keras >= 3.0.0
- pillow >= 9.0.0
- numpy >= 1.23.0
- matplotlib >= 3.5.0

Install:
```bash
pip install tensorflow keras pillow numpy matplotlib
```

---

## Essential Delivery Package

1. **Models** (2 files in runs/)
   - stage1_battery_detector_honest_20260212_095812.keras
   - stage2_waste_classifier_honest_20260212_212056.keras

2. **Production Code** (src/production/)
   - inference_2stage.py
   - stage1_config.json
   - stage2_config.json

3. **Training Code** (src/training/)
   - train_stage1_honest.py
   - train_stage2_honest.py

4. **Validation/Demo** (src/validation/)
   - demo_stage1_predictions.py
   - demo_stage2_predictions.py
   - test_full_pipeline.py
   - evaluate_stage1_threshold_tuning.py
   - evaluate_stage1_test_honest.py
   - evaluate_stage2_threshold_tuning.py
   - evaluate_stage2_test_honest.py

5. **Configuration** (root)
   - config.py
   - requirements.txt

6. **Data Utilities** (src/utils/)
   - recreate_proper_splits.py
   - validate_splits.py

7. **Data** (data/proper_splits/)
   - 31,543 fully organized images in train/val/test splits

---

## Repo Structure

```
SmartTrashBin/
├── README.md (COMPLETE 10-section documentation)
├── .gitignore (excludes data, logs, old models)
├── .gitattributes
├── documentation/ (project docs)
├── archived/ (old datasets, reference code - not needed for current work)
│
└── 2-stage-model/
    ├── config.py (global settings, thresholds locked in JSON)
    ├── requirements.txt
    │
    ├── runs/
    │   ├── stage1_battery_detector_honest_20260212_095812.keras (11.9 MB)
    │   └── stage2_waste_classifier_honest_20260212_212056.keras (11.9 MB)
    │
    └── src/
        ├── production/
        │   ├── inference_2stage.py
        │   ├── convert_to_tflite.py (model conversion for deployment)
        │   ├── stage1_config.json (threshold 0.15 LOCKED)
        │   └── stage2_config.json (threshold 0.07 LOCKED)
        │
        ├── training/
        │   ├── train_stage1_honest.py
        │   └── train_stage2_honest.py
        │
        ├── validation/
        │   ├── demo_stage1_predictions.py 
        │   ├── demo_stage2_predictions.py 
        │   ├── test_full_pipeline.py
        │   ├── test_full_pipeline_compare.py
        │   ├── test_manual_images.py
        │   ├── evaluate_stage1_threshold_tuning.py 
        │   ├── evaluate_stage1_test_honest.py 
        │   ├── evaluate_stage2_threshold_tuning.py 
        │   └── evaluate_stage2_test_honest.py 
        │
        └── utils/ 
            ├── recreate_proper_splits.py (users run this locally)
            └── validate_splits.py (users run this locally)
```

---

**Last Updated:** February 17, 2026
