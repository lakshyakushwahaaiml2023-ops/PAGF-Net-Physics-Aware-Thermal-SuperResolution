import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# IMPORTS
from model import OpticalGuidedSR
from dataLoader import SatelliteThermalDataset
from torch.utils.data import DataLoader

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/final_model.pth"
DATASET_PATH = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"

# CRITICAL: The number of images you used for training
TRAIN_SIZE = 3000 

def test_on_unseen():
    print(f"Preparing UNSEEN data test...")
    
    # 1. Load a LARGER dataset than before
    # We ask for 3100 images, so we can throw away the first 3000
    full_dataset = SatelliteThermalDataset(DATASET_PATH, max_samples=TRAIN_SIZE + 50)
    
    # 2. THE HACK: Manually slice the dataset to keep only the last 50 images
    # This guarantees these images were NOT in the training loop
    if len(full_dataset.image_paths) > TRAIN_SIZE:
        full_dataset.image_paths = full_dataset.image_paths[TRAIN_SIZE:]
        print(f"SUCCESS: Skipped first {TRAIN_SIZE} images.")
        print(f"Testing on {len(full_dataset.image_paths)} NEW, UNSEEN images.")
    else:
        print("WARNING: Not enough data found to skip training set!")
        print("   Testing on whatever we found (results might be biased).")

    # 3. Create Loader for this unseen subset
    test_loader = DataLoader(full_dataset, batch_size=1, shuffle=True)
    
    # 4. Load Model
    model = OpticalGuidedSR().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    print("running inference...")
    
    # 5. Visual Validation Loop
    # We will save the first 3 examples we find
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= 3: break # Only show 3 examples
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Predict
            prediction = model(inputs)
            
            # Prepare for plotting (Move to CPU, Numpy)
            # [0] = Blurry Thermal, [1] = Sharp Optical
            low_res = inputs[0, 0].cpu().numpy()
            optical = inputs[0, 1].cpu().numpy()
            pred_img = prediction[0, 0].cpu().numpy()
            gt_img = targets[0, 0].cpu().numpy()
            
            # Calculate Error Map
            error = np.abs(gt_img - pred_img)
            
            # --- PLOT ---
            fig, axes = plt.subplots(1, 5, figsize=(20, 5))
            
            # 1. Input (What the satellite actually sees)
            axes[0].imshow(low_res, cmap='inferno')
            axes[0].set_title("Input (LR Thermal)\nBlurry & Blocky")
            axes[0].axis('off')

            # 2. Guidance (The Helper)
            axes[1].imshow(optical, cmap='gray')
            axes[1].set_title("Guidance (Optical)\nSharp Structure")
            axes[1].axis('off')

            # 3. Your Model (The Result)
            axes[2].imshow(pred_img, cmap='inferno')
            axes[2].set_title("PAGF-Net Output\n(Physics-Enhanced)")
            axes[2].axis('off')

            # 4. Ground Truth (The Goal)
            axes[3].imshow(gt_img, cmap='inferno')
            axes[3].set_title("Ground Truth\n(Real High-Res)")
            axes[3].axis('off')
            
            # 5. Error (The Proof)
            im = axes[4].imshow(error, cmap='jet', vmin=0, vmax=0.15)
            axes[4].set_title("Error Map\n(Blue = Perfect)")
            axes[4].axis('off')
            
            # Add colorbar to the error map
            plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
            
            plt.suptitle(f"Unseen Test Sample #{i+1}", fontsize=16)
            plt.tight_layout()
            plt.show()
            
            # Save for slides
            plt.savefig(f"unseen_result_{i}.png")
            print(f" Saved unseen_result_{i}.png")

if __name__ == "__main__":
    test_on_unseen()