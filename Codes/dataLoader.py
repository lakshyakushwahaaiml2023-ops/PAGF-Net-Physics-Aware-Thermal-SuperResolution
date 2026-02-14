import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import cv2

class SatelliteThermalDataset(Dataset):
    def __init__(self, root_dir, max_samples=5000, patch_size=128):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.image_paths = []
        
        print(f"[INFO] Scanning {root_dir} for data...")
        
        count = 0
        for root, dirs, files in os.walk(root_dir):
            if "all_bands.tif" in files:
                self.image_paths.append(os.path.join(root, "all_bands.tif"))
                count += 1
                if max_samples is not None and count >= max_samples:
                    break
        
        print(f"[SUCCESS] Ready: Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. LOAD FILE
        tif_path = self.image_paths[idx]
        
        try:
            with rasterio.open(tif_path) as src:
                try:
                    optical = src.read(4)  # Red band (Structure)
                    thermal = src.read(10) # Thermal band
                except IndexError:
                    optical = src.read(1)
                    thermal = src.read(src.count) 
        except:
            return self.__getitem__((idx + 1) % len(self))

        # 2. NORMALIZE (Min-Max)
        optical = optical.astype(np.float32)
        thermal = thermal.astype(np.float32)
        
        # Safety check for flat images (avoid divide by zero)
        if optical.max() > optical.min():
            optical = (optical - optical.min()) / (optical.max() - optical.min())
        else:
            optical = np.zeros_like(optical)
            
        if thermal.max() > thermal.min():
            thermal = (thermal - thermal.min()) / (thermal.max() - thermal.min())
        else:
            thermal = np.zeros_like(thermal)

        # 3. RANDOM CROP
        h, w = thermal.shape
        if h > self.patch_size and w > self.patch_size:
            y = np.random.randint(0, h - self.patch_size)
            x = np.random.randint(0, w - self.patch_size)
            optical = optical[y:y+self.patch_size, x:x+self.patch_size]
            thermal = thermal[y:y+self.patch_size, x:x+self.patch_size]
        else:
            optical = cv2.resize(optical, (self.patch_size, self.patch_size))
            thermal = cv2.resize(thermal, (self.patch_size, self.patch_size))

        # 4. REALISTIC DEGRADATION MODEL (The Update)
        h_lr, w_lr = self.patch_size // 4, self.patch_size // 4
        
        # A. Blur (Simulate Sensor Point Spread Function)
        thermal_blur = cv2.GaussianBlur(thermal, (7, 7), 1.5)
        
        # B. Downsample (Simulate Low Resolution Sensor)
        thermal_lr = cv2.resize(thermal_blur, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
        
        # C. Add Noise (Simulate Thermal Detector Noise)
        noise = np.random.normal(0, 0.01, thermal_lr.shape).astype(np.float32)
        thermal_lr_noisy = thermal_lr + noise
        
        # Clip to ensure valid range [0, 1]
        thermal_lr_noisy = np.clip(thermal_lr_noisy, 0, 1)

        # D. Upsample (The "Bicubic Baseline" Input)
        thermal_input = cv2.resize(thermal_lr_noisy, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)

        # 5. TENSORS
        input_stack = np.stack([thermal_input, optical], axis=0)
        target_stack = thermal[np.newaxis, ...] # The clean, sharp original is the target

        return torch.from_numpy(input_stack).float(), torch.from_numpy(target_stack).float()