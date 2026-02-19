# File Manifest: Splits, Configurations, Data Handling

This document maps which files handle different aspects of the pipeline.

## Core Data Handling

### 🔧 Creating/Rebuilding Splits
**File:** `2-stage-model/recreate_proper_splits.py`

**Purpose:** 
- Downloads and integrates data from all sources
- Creates 70/15/15 train/val/test splits
- Maps Kaggle categories to recyclable/trash classes

**What it loads:**
1. **Original_dataset/** (10 classes)
   - Recyclable: glass/, metal/, paper/, plastic/, cardboard/
   - Trash: biological/, shoes/, clothes/, trash/
   - Battery: battery/

2. **Singapore_Battery_Dataset/** (battery data)
   - Used only for Stage 1 training

3. **recybat24/** (battery data)
   - Used only for Stage 1 training

4. **Kaggle dataset** (optional, 15k images)
   - Maps categories based on subfolder names
   - Both "default" and "real_world" variants

5. **Output:** `data/proper_splits/` structure:
   ```
   proper_splits/
   ├── train/
   ├── val/
   └── test/
   ```

**How to run:**
```bash
python 2-stage-model/recreate_proper_splits.py
```

---

## Stage 1: Battery Detection

### 📋 Configuration
**File:** `2-stage-model/src/stage1_config.json`

**Contains:**
- `threshold`: 0.15 (locked threshold for battery detection)
- `validation_metrics`: Recall, precision, FN count
- `per_source_validation`: Metrics per battery source
- `status`: "LOCKED" (don't modify without retraining)

**What it controls:**
- Battery detection cutoff (P(battery) >= threshold → BATTERY)
- Used by all Stage 1 evaluation scripts

### 🧠 Training
**File:** `2-stage-model/src/train_stage1_honest.py`

**What it does:**
1. Loads data from `data/proper_splits/`
   - **Batteries (label=1):** battery_original, battery_recybat, battery_singapore
   - **Non-batteries (label=0):** recyclable_* + trash_* (9 classes)

2. Trains binary classifier
   - Input: 224×224 RGB images
   - Output: P(battery)
   - Backbone: MobileNetV3Small
   - Training: 30 epochs with early stopping on val_battery_recall

3. Saves best model: `runs/stage1_battery_detector_honest_*.keras`

4. Creates training log in stdout

**How to run:**
```bash
python 2-stage-model/src/train_stage1_honest.py
```

### 🎯 Threshold Tuning
**File:** `2-stage-model/src/evaluate_stage1_threshold_tuning.py`

**What it does:**
1. Loads best Stage 1 model
2. Runs on VAL set at various thresholds (0.05 to 0.95)
3. Finds threshold that maximizes battery_recall while maintaining usability
4. Saves swept results to `runs/stage1_threshold_tuning_*.json`
5. Updates `src/stage1_config.json` with locked threshold

**How to run:**
```bash
python 2-stage-model/src/evaluate_stage1_threshold_tuning.py
```

### ✅ Test Evaluation (ONCE ONLY)
**File:** `2-stage-model/src/evaluate_stage1_test_honest.py`

**What it does:**
1. Loads locked threshold from `src/stage1_config.json`
2. Loads best Stage 1 model
3. Evaluates on held-out TEST set
4. Reports per-source metrics (original, recybat, singapore)
5. Saves results to `runs/stage1_test_results_*.json`
6. Logs to `test_eval_runs.log`

**How to run:**
```bash
python 2-stage-model/src/evaluate_stage1_test_honest.py
```

---

## Stage 2: Waste Classification (Recyclable vs Trash)

### 📋 Configuration
**File:** `2-stage-model/src/stage2_config.json`

**Contains:**
- `threshold`: TBD (recyclable detection cutoff)
- `model_path`: Path to best model
- `status`: "PENDING" (needs threshold tuning)

**What it controls:**
- Recyclable detection cutoff (P(recyclable) >= threshold → RECYCLABLE)
- Used by all Stage 2 evaluation scripts

### 🧠 Training
**File:** `2-stage-model/src/train_stage2_honest.py`

**What it does:**
1. Loads data from `data/proper_splits/`
   - **Recyclable (label=1):** recyclable_glass, recyclable_metal, recyclable_paper, recyclable_plastic, recyclable_cardboard
   - **Trash (label=0):** trash_biological, trash_shoes, trash_clothes, trash_trash

2. Trains binary classifier (2-phase)
   - Phase 1 (epochs 1-2): Frozen backbone, LR=1e-3
   - Phase 2 (epochs 3-50): Unfrozen last 50 backbone layers, LR=3e-5
   - Input: 224×224 RGB images
   - Output: P(recyclable)
   - Backbone: MobileNetV3Small

3. Saves best model: `runs/stage2_waste_classifier_honest_*.keras`

**How to run:**
```bash
python 2-stage-model/src/train_stage2_honest.py
```

### 🎯 Threshold Tuning
**File:** `2-stage-model/src/evaluate_stage2_threshold_tuning.py`

**What it does:**
1. Loads best Stage 2 model
2. Runs on VAL set at various thresholds
3. Finds threshold that maximizes balanced recall (recyclable_recall + trash_recall) / 2
4. Saves swept results to `runs/stage2_threshold_tuning_*.json`
5. Updates `src/stage2_config.json` with locked threshold

**How to run:**
```bash
python 2-stage-model/src/evaluate_stage2_threshold_tuning.py
```

### ✅ Test Evaluation (ONCE ONLY)
**File:** `2-stage-model/src/evaluate_stage2_test_honest.py`

**What it does:**
1. Loads locked threshold from `src/stage2_config.json`
2. Loads best Stage 2 model
3. Evaluates on held-out TEST set
4. Reports per-class metrics (recyclable recall, trash recall)
5. Saves results to `runs/stage2_test_results_*.json`
6. Logs to `test_eval_runs.log`

**How to run:**
```bash
python 2-stage-model/src/evaluate_stage2_test_honest.py
```

---

## Inference Pipeline

### 🚀 Live Predictions
**File:** `2-stage-model/src/inference_2stage.py`

**What it does:**
1. Loads locked Stage 1 + Stage 2 configs
2. Loads both models from `runs/`
3. Runs 2-stage pipeline:
   - Stage 1: P(battery) → if >= 0.15, predict BATTERY
   - Stage 2: P(recyclable) → if >= threshold, predict RECYCLABLE, else TRASH

4. Returns: `{'class': 'BATTERY|RECYCLABLE|TRASH', 'confidence': 0.0-1.0}`

---

## Central Configuration

### ⚙️ Main Config
**File:** `2-stage-model/config.py`

**Contains:**
```python
TARGET_SIZE = (224, 224)           # Input image size
SEED = 42                          # Random seed for reproducibility
STAGE1_BATTERY_THRESHOLD = 0.15    # Stage 1 threshold
STAGE1_EPOCHS = 30
STAGE1_BATCH_SIZE = 16
STAGE2_RECYCLABLE_THRESHOLD = TBD  # Stage 2 threshold (set after tuning)
```

**Usage:** Imported by all training/eval scripts

---

## Data Flow Diagram

```
Original_dataset/    Singapore_Battery/    recybat24/    Kaggle Dataset
       ↓                     ↓                  ↓              ↓
       └─────────────────────┼──────────────────┼──────────────┘
                             ↓
            recreate_proper_splits.py
                   (maps categories)
                             ↓
            proper_splits/ [train/val/test]
           ↙ (batteries)           ↘ (waste)
   Stage 1 Training         Stage 2 Training
        ↓                        ↓
  Battery Detector      Waste Classifier
        ↓                        ↓
   Threshold Tuning      Threshold Tuning
        ↓                        ↓
   Test Evaluation      Test Evaluation
        ↓                        ↓
  stage1_config.json    stage2_config.json
        ↓                        ↓
        └────────────────────────┘
                   ↓
          inference_2stage.py
                   ↓
        BATTERY | RECYCLABLE | TRASH
```

---

## File Reference Table

| Purpose | File | Input | Output |
|---------|------|-------|--------|
| **Data** | `recreate_proper_splits.py` | Original_dataset + Kaggle | `proper_splits/` |
| **S1 Config** | `src/stage1_config.json` | - | Locked threshold (0.15) |
| **S1 Train** | `src/train_stage1_honest.py` | `proper_splits/` | `stage1_*.keras` |
| **S1 Tune** | `src/evaluate_stage1_threshold_tuning.py` | Model + VAL | Updated config |
| **S1 Test** | `src/evaluate_stage1_test_honest.py` | Model + TEST | `stage1_test_results_*.json` |
| **S2 Config** | `src/stage2_config.json` | - | Locked threshold (TBD) |
| **S2 Train** | `src/train_stage2_honest.py` | `proper_splits/` | `stage2_*.keras` |
| **S2 Tune** | `src/evaluate_stage2_threshold_tuning.py` | Model + VAL | Updated config |
| **S2 Test** | `src/evaluate_stage2_test_honest.py` | Model + TEST | `stage2_test_results_*.json` |
| **Inference** | `src/inference_2stage.py` | Image + configs | Prediction |
| **Global** | `config.py` | - | Shared settings |

---

## Execution Sequence

### First Time (After Fresh Data)
```
1. python recreate_proper_splits.py          → Create splits
2. python src/train_stage1_honest.py          → Train Stage 1
3. python src/evaluate_stage1_threshold_tuning.py  → Lock Stage 1 threshold
4. python src/evaluate_stage1_test_honest.py  → Test Stage 1 (FINAL)
5. python src/train_stage2_honest.py          → Train Stage 2
6. python src/evaluate_stage2_threshold_tuning.py  → Lock Stage 2 threshold
7. python src/evaluate_stage2_test_honest.py  → Test Stage 2 (FINAL)
8. python src/inference_2stage.py             → Run inference
```

### Modifying Training (Advanced)
- Edit hyperparameters in `config.py` or individual training scripts
- Rerun steps 2-4 (Stage 1) or 5-7 (Stage 2)
- **DO NOT** re-run step 1 (splits) unless retraining from scratch

---

## Key Principles

1. **Splits are immutable per training cycle**
   - Once `recreate_proper_splits.py` runs, don't run it again during training/tuning
   - Rebuild only when changing data sources

2. **Threshold must be locked before testing**
   - Find on VAL set → Lock in config → Test once on TEST set
   - Don't re-test with different thresholds (overfitting to test)

3. **Configs are golden**
   - `stage1_config.json` locked after threshold tuning
   - `stage2_config.json` locked after threshold tuning
   - Don't modify manually

4. **All models saved to `runs/`**
   - Keep best models for later inference
   - Old models safe to delete (script finds latest)

5. **Logs document decisions**
   - `test_eval_runs.log`: When/why test was run
   - Training logs: Loss, metrics per epoch
   - Threshold tuning logs: All thresholds tried
