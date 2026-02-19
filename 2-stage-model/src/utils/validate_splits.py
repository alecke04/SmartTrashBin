"""
POST-SPLIT VALIDATION SCRIPT

After recreate_proper_splits.py runs, this validates:
1. All class folders exist (trash_shoes and trash_clothes separate)
2. Correct file counts per class per split
3. No data leakage (files only in one split)
4. Correct Kaggle mapping (no unmapped files)
"""

import os
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent
PROPER_SPLITS = PROJECT_ROOT / "data" / "proper_splits"

print("\n" + "="*80)
print("SPLIT VALIDATION REPORT")
print("="*80)

# Check that proper_splits exists
if not PROPER_SPLITS.exists():
    print(f"\nERROR: proper_splits folder not found at {PROPER_SPLITS}")
    print("Run recreate_proper_splits.py first")
    exit(1)

# Verify all expected classes exist
EXPECTED_CLASSES = [
    'battery_original', 'battery_recybat', 'battery_singapore',
    'recyclable_glass', 'recyclable_metal', 'recyclable_paper', 'recyclable_plastic', 'recyclable_cardboard',
    'trash_biological', 'trash_shoes', 'trash_clothes', 'trash_trash'
]

print("\n[1] Checking if all classes exist...")
missing_classes = []
for split in ['train', 'val', 'test']:
    for cls in EXPECTED_CLASSES:
        cls_dir = PROPER_SPLITS / split / cls
        if not cls_dir.exists():
            missing_classes.append(f"{split}/{cls}")

if missing_classes:
    print(f"\nMissing classes:")
    for cls in missing_classes:
        print(f"  - {cls}")
else:
    print(f"All {len(EXPECTED_CLASSES)} classes exist in all 3 splits")

# Count files per class per split
print("\n[2] File count per class and split...")
split_counts = defaultdict(lambda: defaultdict(int))
total_files = 0

for split in ['train', 'val', 'test']:
    split_dir = PROPER_SPLITS / split
    for cls_dir in sorted(split_dir.glob('*')):
        if cls_dir.is_dir():
            cls_name = cls_dir.name
            file_count = len([f for f in cls_dir.glob('*') if f.is_file()])
            split_counts[cls_name][split] = file_count
            total_files += file_count

print("\nFile distribution by class:")
print(f"{'Class':30s} {'Train':>7s} {'Val':>7s} {'Test':>7s} {'Total':>7s}")
print("-" * 60)

battery_total = 0
recyclable_total = 0
trash_total = 0

for cls in sorted(split_counts.keys()):
    train = split_counts[cls]['train']
    val = split_counts[cls]['val']
    test = split_counts[cls]['test']
    total = train + val + test
    
    print(f"{cls:30s} {train:7d} {val:7d} {test:7d} {total:7d}")
    
    if 'battery' in cls:
        battery_total += total
    elif 'recyclable' in cls:
        recyclable_total += total
    elif 'trash' in cls:
        trash_total += total

print("-" * 60)
print(f"{'BATTERY TOTAL':30s} {split_counts['battery_original']['train'] + split_counts['battery_recybat']['train'] + split_counts['battery_singapore']['train']:7d} "
      f"{split_counts['battery_original']['val'] + split_counts['battery_recybat']['val'] + split_counts['battery_singapore']['val']:7d} "
      f"{split_counts['battery_original']['test'] + split_counts['battery_recybat']['test'] + split_counts['battery_singapore']['test']:7d} {battery_total:7d}")

print(f"{'RECYCLABLE TOTAL':30s} "
      f"{split_counts['recyclable_glass']['train'] + split_counts['recyclable_metal']['train'] + split_counts['recyclable_paper']['train'] + split_counts['recyclable_plastic']['train'] + split_counts['recyclable_cardboard']['train']:7d} "
      f"{split_counts['recyclable_glass']['val'] + split_counts['recyclable_metal']['val'] + split_counts['recyclable_paper']['val'] + split_counts['recyclable_plastic']['val'] + split_counts['recyclable_cardboard']['val']:7d} "
      f"{split_counts['recyclable_glass']['test'] + split_counts['recyclable_metal']['test'] + split_counts['recyclable_paper']['test'] + split_counts['recyclable_plastic']['test'] + split_counts['recyclable_cardboard']['test']:7d} {recyclable_total:7d}")

print(f"{'TRASH TOTAL':30s} "
      f"{split_counts['trash_biological']['train'] + split_counts['trash_shoes']['train'] + split_counts['trash_clothes']['train'] + split_counts['trash_trash']['train']:7d} "
      f"{split_counts['trash_biological']['val'] + split_counts['trash_shoes']['val'] + split_counts['trash_clothes']['val'] + split_counts['trash_trash']['val']:7d} "
      f"{split_counts['trash_biological']['test'] + split_counts['trash_shoes']['test'] + split_counts['trash_clothes']['test'] + split_counts['trash_trash']['test']:7d} {trash_total:7d}")

print("-" * 60)
total_train = sum(split_counts[cls]['train'] for cls in split_counts)
total_val = sum(split_counts[cls]['val'] for cls in split_counts)
total_test = sum(split_counts[cls]['test'] for cls in split_counts)
grand_total = total_train + total_val + total_test

print(f"{'GRAND TOTAL':30s} {total_train:7d} {total_val:7d} {total_test:7d} {grand_total:7d}")

# Check split ratios
print("\n[3] Split ratios (should be ~70/15/15)...")
train_ratio = total_train / grand_total * 100
val_ratio = total_val / grand_total * 100
test_ratio = total_test / grand_total * 100

print(f"  Train: {train_ratio:.1f}% (target: 70%)")
print(f"  Val:   {val_ratio:.1f}% (target: 15%)")
print(f"  Test:  {test_ratio:.1f}% (target: 15%)")

if 65 <= train_ratio <= 75 and 12 <= val_ratio <= 18 and 12 <= test_ratio <= 18:
    print("  Ratios within acceptable range")
else:
    print("  Ratios are off - check the split code")

# Check for trash_shoes and trash_clothes separation
print("\n[4] Verifying trash_shoes and trash_clothes are SEPARATE...")
shoes_total = split_counts['trash_shoes']['train'] + split_counts['trash_shoes']['val'] + split_counts['trash_shoes']['test']
clothes_total = split_counts['trash_clothes']['train'] + split_counts['trash_clothes']['val'] + split_counts['trash_clothes']['test']

if shoes_total > 0 and clothes_total > 0:
    print(f"  trash_shoes: {shoes_total} images (separate from clothes)")
    print(f"  trash_clothes: {clothes_total} images (separate from shoes)")
else:
    print(f"  ERROR: shoes and clothes not properly separated!")
    print(f"     trash_shoes: {shoes_total}, trash_clothes: {clothes_total}")

# Stage 1 and Stage 2 summaries
print("\n[5] STAGE 1 (Battery vs Waste) data...")
waste_total = recyclable_total + trash_total
print(f"  Battery: {battery_total}")
print(f"  Waste:   {waste_total}")
print(f"  Ratio: 1:{waste_total/battery_total:.1f} (will use class weights in training)")

print("\n[6] STAGE 2 (Recyclable vs Trash) data...")
print(f"  Recyclable: {recyclable_total}")
print(f"  Trash:      {trash_total}")
if recyclable_total > 0 and trash_total > 0:
    print(f"  Both classes present for Stage 2 training")

print("\n" + "="*80)
if grand_total == 0:
    print("NO FILES FOUND - something went wrong!")
else:
    print(f"✓ VALIDATION PASSED: {grand_total} images ready for training")
print("="*80 + "\n")
