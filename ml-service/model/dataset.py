import json
import random
import numpy as np
# pyrefly: ignore [missing-import]
import torch
from torch.utils.data import Dataset
import tifffile
import imagecodecs  # required for LZW-compressed float32 TIFFs
from pathlib import Path

# Assume normalizer is imported from preprocessing
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.normalization import Normalizer


class OilSpillDataset(Dataset):
    def __init__(self, manifest_rows, mode="train", patch_size=512, stats_path=None, root_dir=""):
        """
        Args:
            manifest_rows: list of dicts from the manifest CSV
            mode: "train" or "val"
            patch_size: size of the extracted patches (512x512)
            stats_path: path to normalization JSON
            root_dir: prefix to prepend to relative paths
        """
        self.manifest = manifest_rows
        self.mode = mode
        self.patch_size = patch_size
        self.root_dir = Path(root_dir)
        
        if stats_path is None:
            stats_path = Path("ml-service/data_manifests/normalization_stats.json")
        self.normalizer = Normalizer(stats_path)
        
        # Validation setup: precompute deterministic grid
        self.val_patches = []
        if self.mode == "val":
            for row in self.manifest:
                # We assume 2048x2048 images yielding exactly 16 patches
                # but we'll calculate it dynamically if needed. 
                # For lazy loading, we'll just queue up coordinates (0, 0), (0, 512)...
                # 2048 / 512 = 4 blocks per dimension
                for r in range(4):
                    for c in range(4):
                        self.val_patches.append({
                            "row": row,
                            "y": r * self.patch_size,
                            "x": c * self.patch_size
                        })

    def __len__(self):
        if self.mode == "train":
            # For training, epoch size can be arbitrary since we sample on-the-fly.
            # Let's define an epoch as 10 * len(manifest) patches.
            return len(self.manifest) * 10
        else:
            return len(self.val_patches)

    def __getitem__(self, idx):
        if self.mode == "train":
            # 1. Oil-aware sampling
            # Bias selection towards 'Oil' class (e.g., 50% chance)
            if random.random() < 0.5:
                oil_rows = [r for r in self.manifest if r['class'] == 'Oil']
                row = random.choice(oil_rows) if oil_rows else random.choice(self.manifest)
            else:
                row = random.choice(self.manifest)
                
            img_path = self.root_dir / row['image_path']
            mask_path = self.root_dir / row['mask_path']
            
            # Read full TIFF lazily
            img_full = tifffile.imread(img_path)
            mask_full = tifffile.imread(mask_path)
            
            # Standardize shape if needed
            if img_full.ndim == 3 and img_full.shape[-1] == 2:
                img_full = np.moveaxis(img_full, -1, 0)
            
            _, H, W = img_full.shape
            
            # For oil-aware sampling inside the image, we ideally want to pick a patch containing oil.
            # Since generating an oil-density map is complex, we use a simple rejection sampling approach:
            y, x = 0, 0
            if row['class'] == 'Oil' and random.random() < 0.8:
                # Try to find a patch with oil
                for _ in range(10):
                    y = random.randint(0, H - self.patch_size)
                    x = random.randint(0, W - self.patch_size)
                    mask_patch = mask_full[y:y+self.patch_size, x:x+self.patch_size]
                    if np.any(mask_patch > 0):
                        break
            else:
                # Random crop
                y = random.randint(0, H - self.patch_size)
                x = random.randint(0, W - self.patch_size)
                
            img_patch = img_full[:, y:y+self.patch_size, x:x+self.patch_size]
            mask_patch = mask_full[y:y+self.patch_size, x:x+self.patch_size]
            
            # Geometric Augmentation
            if random.random() < 0.5:
                img_patch = np.flip(img_patch, axis=1) # Horizontal
                mask_patch = np.flip(mask_patch, axis=0)
            if random.random() < 0.5:
                img_patch = np.flip(img_patch, axis=2) # Vertical
                mask_patch = np.flip(mask_patch, axis=1)
            
            k = random.randint(0, 3)
            if k > 0:
                img_patch = np.rot90(img_patch, k, axes=(1,2))
                mask_patch = np.rot90(mask_patch, k, axes=(0,1))
                
        else:
            # Validation mode: deterministic grid
            patch_info = self.val_patches[idx]
            row = patch_info["row"]
            y, x = patch_info["y"], patch_info["x"]
            
            img_path = self.root_dir / row['image_path']
            mask_path = self.root_dir / row['mask_path']
            
            img_full = tifffile.imread(img_path)
            mask_full = tifffile.imread(mask_path)
            
            if img_full.ndim == 3 and img_full.shape[-1] == 2:
                img_full = np.moveaxis(img_full, -1, 0)
                
            img_patch = img_full[:, y:y+self.patch_size, x:x+self.patch_size]
            mask_patch = mask_full[y:y+self.patch_size, x:x+self.patch_size]
            
        # Normalize
        img_patch = self.normalizer(img_patch)
        
        # Convert to torch tensors
        img_tensor = torch.from_numpy(img_patch.copy()).float()
        
        # Mask shape: (1, H, W)
        mask_tensor = torch.from_numpy(mask_patch.copy()).long().unsqueeze(0)
        
        return img_tensor, mask_tensor
