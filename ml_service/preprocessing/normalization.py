import os
import json
import csv
import numpy as np
import tifffile
from pathlib import Path

MANIFEST_PATH = Path("ml-service/data_manifests/train_manifest.csv")
STATS_OUTPUT_PATH = Path("ml-service/data_manifests/normalization_stats.json")

def compute_percentiles(manifest_path, sample_fraction=1.0, seed=42):
    """
    Computes P1 and P99 percentiles for VV (channel 0) and VH (channel 1) bands
    using ONLY the training data to prevent leakage.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Read training manifest
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        train_rows = [row for row in reader if row['split'] == 'train']

    if not train_rows:
        raise ValueError("No training data found in manifest.")

    # Random subsample for faster computation if requested (can be 1.0 for full)
    np.random.seed(seed)
    if sample_fraction < 1.0:
        n_samples = max(1, int(len(train_rows) * sample_fraction))
        train_rows = np.random.choice(train_rows, n_samples, replace=False)

    vv_pixels = []
    vh_pixels = []

    print(f"Computing percentiles over {len(train_rows)} training images...")

    for i, row in enumerate(train_rows):
        img_path = Path(row['image_path'])
        if not img_path.exists():
            continue
            
        # Images are (2, H, W) or (H, W, 2). Assuming rasterio/tifffile standard.
        # Often Sentinel-1 is (2, H, W).
        img = tifffile.imread(img_path)
        
        # Determine channel dimension
        if img.ndim == 3:
            if img.shape[0] == 2:
                # Shape is (C, H, W)
                vv = img[0].flatten()
                vh = img[1].flatten()
            elif img.shape[-1] == 2:
                # Shape is (H, W, C)
                vv = img[..., 0].flatten()
                vh = img[..., 1].flatten()
            else:
                continue
        else:
            continue

        # Using Reservoir sampling or just appending if memory allows. 
        # Since images are large, we can subsample pixels from each image.
        # 512x512 is 262k pixels. 2048x2048 is 4M pixels.
        # Let's subsample 10,000 pixels per image to fit in memory easily.
        subsample_size = min(10000, len(vv))
        idx = np.random.choice(len(vv), subsample_size, replace=False)
        
        vv_pixels.append(vv[idx])
        vh_pixels.append(vh[idx])

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(train_rows)} images...")

    vv_all = np.concatenate(vv_pixels)
    vh_all = np.concatenate(vh_pixels)

    vv_p1, vv_p99 = np.percentile(vv_all, [1, 99])
    vh_p1, vh_p99 = np.percentile(vh_all, [1, 99])

    stats = {
        "VV": {"P1": float(vv_p1), "P99": float(vv_p99)},
        "VH": {"P1": float(vh_p1), "P99": float(vh_p99)}
    }

    STATS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_OUTPUT_PATH, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"Saved normalization stats to {STATS_OUTPUT_PATH}")
    print(json.dumps(stats, indent=2))
    return stats


class Normalizer:
    def __init__(self, stats_path):
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
            
        self.vv_p1 = self.stats["VV"]["P1"]
        self.vv_p99 = self.stats["VV"]["P99"]
        self.vh_p1 = self.stats["VH"]["P1"]
        self.vh_p99 = self.stats["VH"]["P99"]

    def __call__(self, img):
        """
        Applies linear stretch to [0,1] for VV and VH bands.
        Assumes img is a numpy array of shape (2, H, W).
        """
        out = np.empty_like(img, dtype=np.float32)
        
        # VV
        vv = img[0]
        vv = np.clip(vv, self.vv_p1, self.vv_p99)
        out[0] = (vv - self.vv_p1) / (self.vv_p99 - self.vv_p1 + 1e-8)
        
        # VH
        vh = img[1]
        vh = np.clip(vh, self.vh_p1, self.vh_p99)
        out[1] = (vh - self.vh_p1) / (self.vh_p99 - self.vh_p1 + 1e-8)
        
        return out


if __name__ == "__main__":
    if MANIFEST_PATH.exists():
        compute_percentiles(MANIFEST_PATH)
    else:
        print(f"Manifest not found at {MANIFEST_PATH}. Run split.py first.")
