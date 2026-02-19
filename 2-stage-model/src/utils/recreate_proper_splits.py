"""
Recreate proper_splits with CORRECT class organization for Stage 1 & 2.

DATA SOURCES:
=============
BATTERIES (Stage 1 input only):
  - Original_dataset/battery/                    814 images
  - Singapore_Battery_Dataset/                   360 images
  - recybat24/train/ + val/                    2,835 images
  TOTAL BATTERIES: 4,009

WASTE - Original_dataset:
  - Recyclable: glass, metal, paper, plastic, cardboard = 7,603 images
  - Trash: biological, shoes, clothes, trash = 4,921 images
  TOTAL ORIGINAL WASTE: 12,534

WASTE - Kaggle (RecycleHQ dataset):
  - 30 categories mapped to 9 waste classes = 15,000 images
  TOTAL KAGGLE WASTE: 15,000

GRAND TOTAL: 31,543 images (4,009 battery + 27,534 waste)

STAGE 1 USES: All batteries + all waste (31,543 total)
STAGE 2 USES: Waste only (27,534 total) - binary: recyclable vs trash

SPLITS: 70/15/15 train/val/test per class
"""

import os
import sys
import io
import hashlib
from pathlib import Path
import shutil
import numpy as np
from collections import defaultdict

# Fix encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent
# Navigate up to SmartTrashBin root (3 levels: utils -> src -> 2-stage-model -> root)
SMART_TRASH_BIN_ROOT = PROJECT_ROOT.parent.parent.parent
ORIGINAL_DATA = SMART_TRASH_BIN_ROOT / "Original_dataset"
PROPER_SPLITS = PROJECT_ROOT / "data" / "proper_splits"

# Define class mappings
RECYCLABLE_CLASSES = ['glass', 'metal', 'paper', 'plastic', 'cardboard']
TRASH_CLASSES = ['biological', 'shoes', 'clothes', 'trash']
BATTERY_CLASSES = ['battery_original', 'battery_recybat', 'battery_singapore']

# ============================================================================
# KAGGLE CATEGORY MAPPING
# ============================================================================
# Maps 30 Kaggle categories to 9 waste classes
# NOTE: styrofoam → recyclable_plastic is a POLICY CHOICE (some MRFs treat as trash)

KAGGLE_TO_CLASS = {
    # RECYCLABLE - PLASTIC (11 categories)
    'plastic_water_bottles': 'recyclable_plastic',
    'plastic_soda_bottles': 'recyclable_plastic',
    'plastic_detergent_bottles': 'recyclable_plastic',
    'plastic_shopping_bags': 'recyclable_plastic',
    'plastic_trash_bags': 'recyclable_plastic',
    'plastic_food_containers': 'recyclable_plastic',
    'disposable_plastic_cutlery': 'recyclable_plastic',
    'plastic_straws': 'recyclable_plastic',
    'plastic_cup_lids': 'recyclable_plastic',
    'styrofoam_cups': 'recyclable_plastic',  # POLICY: some MRFs treat as trash
    'styrofoam_food_containers': 'recyclable_plastic',  # POLICY: some MRFs treat as trash
    
    # RECYCLABLE - PAPER & CARDBOARD
    'newspaper': 'recyclable_paper',
    'office_paper': 'recyclable_paper',
    'magazines': 'recyclable_paper',
    'paper_cups': 'recyclable_paper',
    'cardboard_boxes': 'recyclable_cardboard',
    'cardboard_packaging': 'recyclable_cardboard',
    
    # RECYCLABLE - GLASS (3 categories)
    'glass_beverage_bottles': 'recyclable_glass',
    'glass_cosmetic_containers': 'recyclable_glass',
    'glass_food_jars': 'recyclable_glass',
    
    # RECYCLABLE - METAL (4 categories)
    'aerosol_cans': 'recyclable_metal',
    'aluminum_food_cans': 'recyclable_metal',
    'aluminum_soda_cans': 'recyclable_metal',
    'steel_food_cans': 'recyclable_metal',
    
    # TRASH - BIOLOGICAL (4 categories)
    'coffee_grounds': 'trash_biological',
    'eggshells': 'trash_biological',
    'tea_bags': 'trash_biological',
    'food_waste': 'trash_biological',
    
    # TRASH - TEXTILES (2 categories)
    'clothing': 'trash_clothes',
    'shoes': 'trash_shoes',
}


def load_kaggle_dataset():
    """Load Kaggle waste classification dataset from local directory"""
    kaggle_path = SMART_TRASH_BIN_ROOT / "RecyclableHouseholdWaste" / "images" / "images"
    
    if not kaggle_path.exists():
        raise FileNotFoundError(f"FATAL: Kaggle dataset not found at {kaggle_path}")
    
    print(f"\nFound Kaggle dataset at {kaggle_path}")
    return kaggle_path


def unique_filename(src_path, source_prefix, class_name):
    """
    Create a unique, Windows-safe filename that won't collide even if multiple sources
    have files with the same name.
    
    Format: source__class__hash__sanitized_originalname
    Example: kaggle__glass_beverage_bottles__a1b2c3d4__Image_001.jpg
    
    Sanitization removes/replaces:
    - Unicode/special characters (keep only A-Z a-z 0-9 . - _)
    - Names longer than 200 chars (Windows path limit)
    """
    # Hash the full path to ensure uniqueness
    path_hash = hashlib.md5(str(src_path).encode('utf-8')).hexdigest()[:8]
    
    # Sanitize the original filename: keep only safe characters
    original_name = src_path.name
    # Replace unsafe chars with underscore, keep alphanumeric, dots, hyphens, underscores
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in original_name)
    # Replace spaces with underscores
    safe_name = safe_name.replace(" ", "_")
    # Limit length to prevent Windows path issues
    safe_name = safe_name[:100]
    
    return f"{source_prefix}__{class_name}__{path_hash}__{safe_name}"


def create_proper_splits():
    """Recreate proper_splits with correct labels from all sources"""
    
    # Create all directories
    for split in ['train', 'val', 'test']:
        for cls in RECYCLABLE_CLASSES:
            (PROPER_SPLITS / split / f'recyclable_{cls}').mkdir(parents=True, exist_ok=True)
        for cls in TRASH_CLASSES:
            (PROPER_SPLITS / split / f'trash_{cls}').mkdir(parents=True, exist_ok=True)
        for cls in BATTERY_CLASSES:
            (PROPER_SPLITS / split / cls).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("REBUILDING PROPER_SPLITS WITH CORRECT CLASS ORGANIZATION")
    print("="*80)
    
    # Collect all images by class (store tuples of (path, source_prefix))
    all_images = defaultdict(list)
    
    # ========================================================================
    # LOAD ORIGINAL_DATASET WASTE CLASSES
    # ========================================================================
    print("\n[1/3] Loading Original_dataset waste classes...")
    for cls in RECYCLABLE_CLASSES:
        cls_dir = ORIGINAL_DATA / cls
        if cls_dir.exists():
            for img in sorted(cls_dir.glob('*')):
                if img.is_file() and img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                    all_images[f'recyclable_{cls}'].append((img, 'original'))
    
    for cls in TRASH_CLASSES:
        cls_dir = ORIGINAL_DATA / cls
        if cls_dir.exists():
            for img in sorted(cls_dir.glob('*')):
                if img.is_file() and img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                    all_images[f'trash_{cls}'].append((img, 'original'))
    
    original_waste_count = sum(len(imgs) for cls_name, imgs in all_images.items() if 'recycl' in cls_name or 'trash' in cls_name)
    print(f"  Loaded {original_waste_count} Original_dataset waste images")
    
    # ========================================================================
    # LOAD BATTERY DATA (Stage 1 input)
    # ========================================================================
    print("\n[1.5/3] Loading battery data...")
    
    # Battery from Original_dataset
    battery_dir = ORIGINAL_DATA / 'battery'
    if battery_dir.exists():
        for img in sorted(battery_dir.glob('*')):
            if img.is_file() and img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                all_images['battery_original'].append((img, 'original'))
    
    # Battery from Singapore_Battery_Dataset (may have nested structure)
    singapore_dir = ORIGINAL_DATA.parent / 'Singapore_Battery_Dataset'
    if singapore_dir.exists():
        for img in singapore_dir.rglob('*'):
            if img.is_file() and img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                all_images['battery_singapore'].append((img, 'singapore'))
    
    # Battery from RecyBat dataset
    recybat_dir = ORIGINAL_DATA.parent / 'recybat24'
    if recybat_dir.exists():
        for img in recybat_dir.rglob('*'):
            if img.is_file() and img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                all_images['battery_recybat'].append((img, 'recybat'))
    
    battery_total = sum(len(all_images[cls]) for cls in BATTERY_CLASSES)
    print(f"  Loaded {battery_total} battery images")
    print(f"    - battery_original: {len(all_images['battery_original'])}")
    print(f"    - battery_singapore: {len(all_images['battery_singapore'])}")
    print(f"    - battery_recybat: {len(all_images['battery_recybat'])}")
    
    # ========================================================================
    # LOAD KAGGLE DATASET
    # ========================================================================
    print("\n[2/3] Loading Kaggle dataset...")
    kaggle_path = load_kaggle_dataset()
    kaggle_count = 0
    categories_found = 0
    categories_missing = []
    
    for category_dir in sorted(kaggle_path.glob('*')):
        if category_dir.is_dir():
            category_name = category_dir.name
            
            # Check if this category is in our mapping
            if category_name not in KAGGLE_TO_CLASS:
                categories_missing.append(category_name)
                continue
            
            categories_found += 1
            target_class = KAGGLE_TO_CLASS[category_name]
            
            # Load from both 'default' and 'real_world' subdirectories
            for variant_dir in [category_dir / 'default', category_dir / 'real_world']:
                if variant_dir.exists():
                    for img in sorted(variant_dir.glob('*')):
                        if img.is_file() and img.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                            all_images[target_class].append((img, 'kaggle'))
                            kaggle_count += 1
    
    print(f"  ✓ Loaded {kaggle_count} Kaggle images from {categories_found} categories")
    if categories_missing:
        print(f"  ⚠ Warning: {len(categories_missing)} unmapped categories: {categories_missing}")
    
    # ========================================================================
    # VERIFY COUNTS BEFORE SPLITTING
    # ========================================================================
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION (BEFORE SPLITTING)")
    print("="*80)
    
    total_images = 0
    battery_images = 0
    waste_images = 0
    
    print("\nBATTERY CLASSES:")
    for cls in BATTERY_CLASSES:
        count = len(all_images[cls])
        battery_images += count
        total_images += count
        print(f"  {cls:30s}: {count:6d}")
    
    print("\nRECYCLABLE CLASSES:")
    for cls in RECYCLABLE_CLASSES:
        count = len(all_images[f'recyclable_{cls}'])
        waste_images += count
        total_images += count
        print(f"  recyclable_{cls:20s}: {count:6d}")
    
    print("\nTRASH CLASSES:")
    for cls in TRASH_CLASSES:
        count = len(all_images[f'trash_{cls}'])
        waste_images += count
        total_images += count
        print(f"  trash_{cls:20s}: {count:6d}")
    
    print(f"\n{'='*80}")
    print(f"BATTERY TOTAL:  {battery_images:6d}")
    print(f"WASTE TOTAL:    {waste_images:6d}")
    print(f"GRAND TOTAL:    {total_images:6d}")
    print(f"{'='*80}")
    
    # ========================================================================
    # SPLIT INTO TRAIN/VAL/TEST (70/15/15) PER CLASS
    # ========================================================================
    print("\nSplitting into train/val/test (70/15/15 per class)...")
    split_counts = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    
    files_copied = 0
    files_failed = 0
    collisions_prevented = 0
    
    for cls_name in sorted(all_images.keys()):
        img_list = all_images[cls_name]
        n = len(img_list)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        
        # Shuffle and split
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # Copy files to proper splits
        for split_name, idx_list in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            for idx in idx_list:
                src_path, source_prefix = img_list[idx]
                
                # Generate unique filename to prevent collisions
                safe_filename = unique_filename(src_path, source_prefix, cls_name)
                dst = PROPER_SPLITS / split_name / cls_name / safe_filename
                
                # Sanity check: destination should not already exist
                if dst.exists():
                    collisions_prevented += 1
                    # This should never happen with proper hashing, but guard against it
                    continue
                
                try:
                    shutil.copy2(src_path, dst)
                    files_copied += 1
                    if files_copied % 5000 == 0:
                        print(f"  Progress: {files_copied} files copied...")
                except Exception as e:
                    files_failed += 1
                    if files_failed <= 10:  # Print first 10 errors for visibility
                        print(f"  ERROR [{split_name}/{cls_name}] {src_path.name}: {type(e).__name__}: {e}")
                    continue
                
                split_counts[cls_name][split_name] += 1
    
    # ========================================================================
    # FINAL STATISTICS
    # ========================================================================
    print("\n" + "="*80)
    print("SPLIT RESULTS")
    print("="*80)
    
    print("\nPer-class split distribution:")
    total_train = total_val = total_test = 0
    for cls_name in sorted(split_counts.keys()):
        counts = split_counts[cls_name]
        print(f"  {cls_name:30s}: train={counts['train']:4d}, val={counts['val']:4d}, test={counts['test']:4d}")
        total_train += counts['train']
        total_val += counts['val']
        total_test += counts['test']
    
    grand_total = total_train + total_val + total_test
    print(f"\n  {'TOTAL':30s}: train={total_train:4d}, val={total_val:4d}, test={total_test:4d}")
    print(f"  Ratio: {total_train}/{grand_total} train, {total_val}/{grand_total} val, {total_test}/{grand_total} test")
    print(f"\nFile operations: {files_copied} copied, {files_failed} failed")
    if collisions_prevented > 0:
        print(f"  {collisions_prevented} potential collisions prevented via unique naming")
    
    # Summary by type
    print("\n" + "="*80)
    print("SUMMARY BY TYPE")
    print("="*80)
    
    recyclable_train = sum(split_counts[f'recyclable_{c}']['train'] for c in RECYCLABLE_CLASSES)
    recyclable_val = sum(split_counts[f'recyclable_{c}']['val'] for c in RECYCLABLE_CLASSES)
    recyclable_test = sum(split_counts[f'recyclable_{c}']['test'] for c in RECYCLABLE_CLASSES)
    
    trash_train = sum(split_counts[f'trash_{c}']['train'] for c in TRASH_CLASSES)
    trash_val = sum(split_counts[f'trash_{c}']['val'] for c in TRASH_CLASSES)
    trash_test = sum(split_counts[f'trash_{c}']['test'] for c in TRASH_CLASSES)
    
    battery_train = sum(split_counts[c]['train'] for c in BATTERY_CLASSES)
    battery_val = sum(split_counts[c]['val'] for c in BATTERY_CLASSES)
    battery_test = sum(split_counts[c]['test'] for c in BATTERY_CLASSES)
    
    print(f"\nRECYCLABLE (glass, metal, paper, plastic, cardboard):")
    print(f"  Train: {recyclable_train}, Val: {recyclable_val}, Test: {recyclable_test}")
    print(f"  Total: {recyclable_train + recyclable_val + recyclable_test}")
    
    print(f"\nTRASH (biological, shoes, clothes, trash):")
    print(f"  Train: {trash_train}, Val: {trash_val}, Test: {trash_test}")
    print(f"  Total: {trash_train + trash_val + trash_test}")
    
    print(f"\nBATTERY (original, recybat, singapore):")
    print(f"  Train: {battery_train}, Val: {battery_val}, Test: {battery_test}")
    print(f"  Total: {battery_train + battery_val + battery_test}")
    
    print(f"\nSTAGE 1 (Battery vs Waste):")
    waste_train = recyclable_train + trash_train
    waste_val = recyclable_val + trash_val
    waste_test = recyclable_test + trash_test
    print(f"  Battery: {battery_train} train, {battery_val} val, {battery_test} test = {battery_train + battery_val + battery_test} total")
    print(f"  Waste:   {waste_train} train, {waste_val} val, {waste_test} test = {waste_train + waste_val + waste_test} total")
    print(f"  IMBALANCE: {battery_train}:{waste_train} ≈ 1:{waste_train/battery_train:.1f}")
    
    print(f"\nSTAGE 2 (Recyclable vs Trash):")
    print(f"  Recyclable: {recyclable_train} train, {recyclable_val} val, {recyclable_test} test = {recyclable_train + recyclable_val + recyclable_test} total")
    print(f"  Trash:      {trash_train} train, {trash_val} val, {trash_test} test = {trash_train + trash_val + trash_test} total")
    
    print(f"\n{'='*80}")
    print(f"ALL DATA: {total_train} train, {total_val} val, {total_test} test = {grand_total} TOTAL")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    create_proper_splits()
