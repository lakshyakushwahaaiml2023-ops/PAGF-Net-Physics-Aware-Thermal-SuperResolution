import rasterio
import numpy as np
import cv2
import os

# --- 1. SETUP PATHS ---
# Update this path if you changed folders. 
# Based on your logs, this is where your data lives:
base_path = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark\ssl4eo_l_oli_tirs_toa_benchmark\0000000\LC08_045030_20190814"
tif_path = os.path.join(base_path, "all_bands.tif")

# --- 2. LOAD DATA ---
if not os.path.exists(tif_path):
    print(f" Error: File not found at {tif_path}")
    exit()

print(f" Loading file: {tif_path}")

with rasterio.open(tif_path) as src:
    # Read Optical (Band 4)
    try:
        optical_img = src.read(4) 
        print(" Optical (Band 4) loaded.")
    except IndexError:
        print(" Warning: Band 4 not found. Loading Band 1.")
        optical_img = src.read(1)

    # Read Thermal (Band 10)
    try:
        thermal_img = src.read(10)
        print(" Thermal (Band 10) loaded.")
    except IndexError:
        print(" Warning: Band 10 not found. Loading last band.")
        thermal_img = src.read(src.count)

# --- 3. ROBUST NORMALIZATION ---
def normalize(img):
    # Converts to Float32 and scales to [0, 1]
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-6)

print("Normalizing data...")
optical = normalize(optical_img)
thermal = normalize(thermal_img)

# --- 4. SIMULATE LOW-RES INPUT (The "Physics" Part) ---
scale = 4
h, w = thermal.shape

# Downsample (Simulate sensor blur)
thermal_lr = cv2.resize(thermal, (w//scale, h//scale), interpolation=cv2.INTER_AREA)

# Upsample (The "Blurry Input" for the network)
thermal_lr_up = cv2.resize(thermal_lr, (w, h), interpolation=cv2.INTER_CUBIC)

# --- 5. STACK FOR AI MODEL ---
# Stack them: Channel 0 = Blurry Thermal, Channel 1 = Sharp Optical
input_stack = np.stack([thermal_lr_up, optical], axis=0)

print("-" * 30)
print(f"SUCCESS! Data is ready for the Neural Network.")
print(f"Input Shape (Channels, H, W): {input_stack.shape}")
print(f"Data Type: {input_stack.dtype}")
print(f"Value Range: {input_stack.min():.4f} to {input_stack.max():.4f}")
print("-" * 30)