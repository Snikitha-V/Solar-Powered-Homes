# ============================================
# STEP 1: PREPARE MERGED DATASET
# merge_datasets.py
# ============================================

import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm
import cv2

def merge_roboflow_datasets():
    """
    Merge your two downloaded Roboflow datasets into single YOLO format
    """
    
    print("=" * 80)
    print("MERGING ROBOFLOW DATASETS")
    print("=" * 80)
    
    # Your dataset paths (MODIFY THESE TO MATCH YOUR ACTUAL PATHS)
    dataset1_path = Path("datasets/lsgi547-project-3")
    dataset2_path = Path("datasets/solar-panels-ba8ty-1")
    
    # Output path
    output_dir = Path("data/merged_solar_dataset")
    
    # Verify datasets exist
    if not dataset1_path.exists():
        raise FileNotFoundError(f"Dataset 1 not found: {dataset1_path}")
    if not dataset2_path.exists():
        raise FileNotFoundError(f"Dataset 2 not found: {dataset2_path}")
    
    print(f"\nDataset 1: {dataset1_path}")
    print(f"Dataset 2: {dataset2_path}")
    
    # Create output structure
    for split in ['train', 'valid', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy files from both datasets
    for dataset_idx, dataset_path in enumerate([dataset1_path, dataset2_path], 1):
        print(f"\n{'='*60}")
        print(f"Processing Dataset {dataset_idx}: {dataset_path.name}")
        print('='*60)
        
        for split in ['train', 'valid', 'test']:
            img_src = dataset_path / split / 'images'
            lbl_src = dataset_path / split / 'labels'
            
            if not img_src.exists():
                print(f"⚠️  {split} split not found, skipping...")
                continue
            
            # Get all images
            images = list(img_src.glob('*.jpg')) + list(img_src.glob('*.png')) + list(img_src.glob('*.jpeg'))
            
            print(f"\n{split.upper()} split: {len(images)} images")
            
            copied = 0
            skipped = 0
            
            for img_path in tqdm(images, desc=f"Copying {split}"):
                # Create unique filename using dataset name prefix
                new_name = f"ds{dataset_idx}_{img_path.name}"
                
                # Check if corresponding label exists
                label_path = lbl_src / f"{img_path.stem}.txt"
                
                if label_path.exists():
                    # Copy image
                    shutil.copy2(
                        img_path,
                        output_dir / split / 'images' / new_name
                    )
                    
                    # Copy label
                    shutil.copy2(
                        label_path,
                        output_dir / split / 'labels' / f"{Path(new_name).stem}.txt"
                    )
                    copied += 1
                else:
                    skipped += 1
            
            print(f"  ✓ Copied: {copied}, Skipped (no label): {skipped}")
    
    # Count final dataset
    train_count = len(list((output_dir / 'train' / 'images').glob('*')))
    valid_count = len(list((output_dir / 'valid' / 'images').glob('*')))
    test_count = len(list((output_dir / 'test' / 'images').glob('*')))
    
    # Create data.yaml
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['solar_panel']
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n{'='*80}")
    print("MERGE COMPLETE")
    print('='*80)
    print(f"Train:      {train_count:,} images")
    print(f"Validation: {valid_count:,} images")
    print(f"Test:       {test_count:,} images")
    print(f"Total:      {train_count + valid_count + test_count:,} images")
    print(f"\nDataset config: {yaml_path}")
    print(f"Dataset location: {output_dir.absolute()}")
    
    return yaml_path

if __name__ == "__main__":
    merge_roboflow_datasets()