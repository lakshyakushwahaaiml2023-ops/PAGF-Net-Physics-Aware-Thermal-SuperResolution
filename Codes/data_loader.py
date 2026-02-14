import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import cv2
import hashlib

class RigorousThermalDataset(Dataset):
    def __init__(self, root_dir, mode='train', split_ratio=0.8, patch_size=128):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.mode = mode
        self.file_paths = []
        
        # 1. FIXED SPLIT BY FOLDER (Scene-level split)
        # We verify scenes to prevent data leakage
        print(f"[INFO] Scanning {root_dir} for {mode} data...")
        
        all_scenes = []
        for root, dirs, files in os.walk(root_dir):
            if "all_bands.tif" in files:
                all_scenes.append(root)
        
        # Sort to ensure reproducibility
        all_scenes.sort()
        
        # Deterministic Split
        split_idx = int(len(all_scenes) * split_ratio)
        if mode == 'train':
            self.scenes = all_scenes[:split_idx]
        else:
            self.scenes = all_scenes[split_idx:]
            
        # Collect all file paths
        for scene in self.scenes:
            self.file_paths.append(os.path.join(scene, "all_bands.tif"))
            
        print(f"[SUCCESS] {mode.upper()} Set: {len(self.file_paths)} scenes.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        tif_path = self.file_paths[idx]
        
        try:
            with rasterio.open(tif_path) as src:
                # Load raw data (likely uint16)
                try:
                    optical = src.read(4) 
                    thermal = src.read(10)
                except:
                    optical = src.read(1)
                    thermal = src.read(1)
        except:
             # Retry random valid index if corrupt
            return self.__getitem__((idx + 1) % len(self))

        # 2. GLOBAL NORMALIZATION (No per-image min/max!)
        # Assuming uint16 data (0-65535). 
        # If your data is float (0-1), change 65535.0 to 1.0
        scale_factor = 65535.0 
        if optical.dtype == np.float32: scale_factor = 1.0 # Safety check
            
        optical = optical.astype(np.float32) / scale_factor
        thermal = thermal.astype(np.float32) / scale_factor

        # Clip just in case
        optical = np.clip(optical, 0, 1)
        thermal = np.clip(thermal, 0, 1)

        # 3. RANDOM CROP
        h, w = thermal.shape
        if h > self.patch_size and w > self.patch_size:
            # For testing, we might want center crop, but random is okay for now
            if self.mode == 'train':
                y = np.random.randint(0, h - self.patch_size)
                x = np.random.randint(0, w - self.patch_size)
            else:
                # Center crop for validation (deterministic)
                y = (h - self.patch_size) // 2
                x = (w - self.patch_size) // 2
            
            optical = optical[y:y+self.patch_size, x:x+self.patch_size]
            thermal = thermal[y:y+self.patch_size, x:x+self.patch_size]
        else:
            optical = cv2.resize(optical, (self.patch_size, self.patch_size))
            thermal = cv2.resize(thermal, (self.patch_size, self.patch_size))

        # 4. DEGRADATION MODEL (Downsample + Noise)
        h_lr, w_lr = self.patch_size // 4, self.patch_size // 4
        
        # A. Downsample (Blur)
        thermal_down = cv2.resize(thermal, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
        
        # B. Add Gaussian Noise (Simulate Sensor Noise)
        noise = np.random.normal(0, 0.01, thermal_down.shape).astype(np.float32) # Sigma = 0.01
        thermal_noisy = thermal_down + noise
        thermal_noisy = np.clip(thermal_noisy, 0, 1)

        # C. Upsample (Bicubic Baseline Input)
        thermal_input = cv2.resize(thermal_noisy, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)

        # 5. Tensors
        input_stack = np.stack([thermal_input, optical], axis=0)
        target_stack = thermal[np.newaxis, ...]

        return torch.from_numpy(input_stack).float(), torch.from_numpy(target_stack).float()