import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Point to the SINGLE stacked file
base_path = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark\ssl4eo_l_oli_tirs_toa_benchmark\0000000\LC08_045030_20190814"
tif_path = os.path.join(base_path, "all_bands.tif")

# Check if it exists
if not os.path.exists(tif_path):
    print(f"Error: File not found at {tif_path}")
else:
    print(f"Found file: {tif_path}")
    
    with rasterio.open(tif_path) as src:
        print(f"File Information: {src.count} bands, {src.width}x{src.height}")
        
        # --- CRITICAL: Landsat 8 Band Mapping ---
        # In this dataset, bands usually follow the standard order:
        # Band 4 = Red (Optical Structure) -> Index 4
        # Band 10 = Thermal Infrared (Temperature) -> Index 10
        
        # 1. Read Optical (Band 4)
        # Note: Rasterio uses 1-based indexing
        try:
            optical_img = src.read(4) 
            print("Optical (Band 4) loaded.")
        except IndexError:
            print("Warning: Band 4 not found. Loading Band 1 instead.")
            optical_img = src.read(1)

        # 2. Read Thermal (Band 10)
        try:
            thermal_img = src.read(10)
            print("Thermal (Band 10) loaded.")
        except IndexError:
            # Fallback if the dataset only has fewer bands (e.g. some subsets only have RGB)
            print(" Warning: Band 10 (Thermal) not found. Checking total bands...")
            thermal_img = src.read(src.count) # Read the last available band as a guess

        import numpy as np

# Assuming 'optical_img' and 'thermal_img' are your loaded arrays from rasterio
        print(f"Optical Type: {optical_img.dtype}")
        print(f"Thermal Type: {thermal_img.dtype}")

        # You can also check the min/max to see the range
        print(f"Optical Range: {optical_img.min()} - {optical_img.max()}")
        print(f"Thermal Range: {thermal_img.min()} - {thermal_img.max()}")

        # --- VISUALIZATION CHECK ---
        plt.figure(figsize=(10, 5))
        
        # Display Optical (Normalized for viewing)
        plt.subplot(1, 2, 1)
        plt.imshow(optical_img, cmap='gray')
        plt.title(f"Optical Input (Band 4)\nMin:{optical_img.min()} Max:{optical_img.max()}")
        plt.axis('off')

        # Display Thermal
        plt.subplot(1, 2, 2)
        plt.imshow(thermal_img, cmap='inferno')
        plt.title(f"Thermal Input (Band 10)\nMin:{thermal_img.min()} Max:{thermal_img.max()}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

        print("\nSUCCESS: Data is ready for the Physics-Guided Model.")