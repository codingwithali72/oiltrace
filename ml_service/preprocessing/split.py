import os
import csv
import random
from pathlib import Path

# Paths relative to project root
DATA_ROOT = Path("dataset_raw")
OUTPUT_DIR = Path("ml-service/data_manifests")

CLASSES = {
    "Oil": {
        "img_dir": DATA_ROOT / "oil_images" / "Oil",
        "mask_dir": DATA_ROOT / "oil_mask" / "Mask_oil"
    },
    "No-Oil": {
        "img_dir": DATA_ROOT / "no_oil_images" / "No_oil",
        "mask_dir": DATA_ROOT / "no_oil_mask" / "Mask_no_oil"
    },
    "Look-alike": {
        "img_dir": DATA_ROOT / "lookalike_images" / "Lookalike",
        "mask_dir": DATA_ROOT / "lookalike_mask" / "Mask_lookalike"
    }
}

SEED = 42
TRAIN_RATIO = 0.8

def build_split():
    random.seed(SEED)
    
    train_rows = []
    val_rows = []
    
    # Process each class
    for cls_name, dirs in CLASSES.items():
        img_dir = dirs["img_dir"]
        mask_dir = dirs["mask_dir"]
        
        # We only record paths if the directory exists (might not exist locally yet)
        if not img_dir.exists() or not mask_dir.exists():
            print(f"Warning: Directory not found for class {cls_name}. Skipping.")
            continue
            
        images = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif') or f.endswith('.tiff')])
        masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif') or f.endswith('.tiff')])
        
        # Assuming names match exactly
        assert len(images) == len(masks), f"Mismatch in counts for {cls_name}"
        
        pairs = []
        for img_name in images:
            # We use POSIX paths for cross-platform compatibility in the CSV
            img_path = (img_dir / img_name).as_posix()
            mask_path = (mask_dir / img_name).as_posix()
            
            # File ID is just the stem (e.g., 00000)
            file_id = Path(img_name).stem
            
            pairs.append({
                "class": cls_name,
                "id": file_id,
                "image_path": img_path,
                "mask_path": mask_path
            })
            
        # Shuffle for the split
        random.shuffle(pairs)
        
        split_idx = int(len(pairs) * TRAIN_RATIO)
        cls_train = pairs[:split_idx]
        cls_val = pairs[split_idx:]
        
        for row in cls_train:
            row["split"] = "train"
        for row in cls_val:
            row["split"] = "val"
            
        train_rows.extend(cls_train)
        val_rows.extend(cls_val)
        
    return train_rows, val_rows

def save_manifest(rows, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename
    
    columns = ["split", "class", "id", "image_path", "mask_path"]
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} records to {filepath}")

if __name__ == "__main__":
    print("Building dataset splits...")
    train_rows, val_rows = build_split()
    
    if train_rows or val_rows:
        save_manifest(train_rows, "train_manifest.csv")
        save_manifest(val_rows, "val_manifest.csv")
        print("Done.")
    else:
        print("No data found. Ensure dataset_raw is present in the project root.")
