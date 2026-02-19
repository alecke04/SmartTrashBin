"""
config.py

Configuration for 2-Stage Waste Classification Model

Architecture: Two-Stage Safety-First Pipeline
  Stage 1: Binary Battery Detection → REJECT or continue
  Stage 2: Waste Classification (if not battery) → RECYCLABLE or TRASH

Data sources:
  - Full Battery dataset (3,020 images)
  - Full RECYCLABLE dataset (7,613 images)
  - Full TRASH dataset (4,921 images)
  Total: 15,554 images
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Parent data folder (shared with smartbin-cv)
DATA_DIR = PROJECT_ROOT.parent / "data"

# Runs directory
RUNS_DIR = PROJECT_ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

# Image settings
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_SIZE = (224, 224)
SEED = 42

# ============================================================================
# STAGE 1: BINARY BATTERY DETECTION
# ============================================================================
# Class 0: BATTERY → REJECT
# Class 1: NOT-BATTERY (recyclable/trash) → Continue to Stage 2

STAGE1_BATTERY_THRESHOLD = 0.40  # Conservative: catch batteries first
STAGE1_EPOCHS = 30
STAGE1_BATCH_SIZE = 16

# ============================================================================
# STAGE 2: WASTE CLASSIFICATION
# ============================================================================
# Class 0: RECYCLABLE → Chute 1
# Class 1: TRASH → Chute 2

STAGE2_RECYCLABLE_THRESHOLD = 0.50
STAGE2_EPOCHS = 30
STAGE2_BATCH_SIZE = 16

# ============================================================================
# OUTPUT ROUTING
# ============================================================================

OUTPUTS = {
    'RECYCLABLE': 'Chute 1 (Recyclables)',
    'TRASH': 'Chute 2 (Trash)',
    'REJECT': 'Reject Gate (Hazardous Battery)',
}
