import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# --- NEW IMPORTS ---
from model_advanced import AdvancedOpticalSR 
from dataLoader import SatelliteThermalDataset
from torch.utils.data import DataLoader

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Point to your NEW Advanced Model
MODEL_PATH = "checkpoints/final_advanced_model.pth" 
DATASET_PATH = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"

# Skip training data to ensure we test on UNSEEN images
TRAIN_SIZE = 4000 

def test_on_unseen():
    print(f"Preparing ADVANCED Visual Test...")
    
    # 1. Load Unseen Data
    full_dataset = SatelliteThermalDataset(DATASET_PATH, max_samples=TRAIN_SIZE + 50)
    
    if len(full_dataset.image_paths) > TRAIN_SIZE:
        full_dataset.image_paths = full_dataset.image_paths[TRAIN_SIZE:]
        print(f"Testing on {len(full_dataset.image_paths)} NEW, UNSEEN images.")
    else:
        print("Warning: Not enough data to skip. Testing on what we found.")

    test_loader = DataLoader(full_dataset, batch_size=1, shuffle=True)
    
    # 2. Load ADVANCED Model
    model = AdvancedOpticalSR().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Advanced Model loaded.")
    else:
        print(f"Error: Checkpoint not found at {MODEL_PATH}")
        print("   Did you run train_advanced.py?")
        return

    model.eval()
    
    print("running inference...")
    
    # 3. Visual Validation Loop
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= 3: break # Save 3 examples
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # --- CRITICAL CHANGE: UNPACK TUPLE ---
            # The advanced model returns (Final, Mid). We only want Final.
            prediction, _ = model(inputs)
            
            # Prepare for plotting
            # Inputs[0,0] is the Bicubic Baseline (from loader)
            # Inputs[0,1] is Optical Guidance
            bicubic = inputs[0, 0].cpu().numpy()
            optical = inputs[0, 1].cpu().numpy()
            pred_img = prediction[0, 0].cpu().numpy()
            gt_img = targets[0, 0].cpu().numpy()
            
            # Calculate Error Map
            error = np.abs(gt_img - pred_img)
            
            # --- PLOT ---
            fig, axes = plt.subplots(1, 5, figsize=(20, 5))
            
            # 1. Bicubic (Baseline)
            axes[0].imshow(bicubic, cmap='inferno')
            axes[0].set_title("Bicubic Baseline\n(Blurry)")
            axes[0].axis('off')

            # 2. Optical (Guidance)
            axes[1].imshow(optical, cmap='gray')
            axes[1].set_title("Optical Guidance\n(Structural Detail)")
            axes[1].axis('off')

            # 3. Advanced Output
            axes[2].imshow(pred_img, cmap='inferno')
            axes[2].set_title("Advanced SFT Output\n(Sharper Edges)")
            axes[2].axis('off')

            # 4. Ground Truth
            axes[3].imshow(gt_img, cmap='inferno')
            axes[3].set_title("Ground Truth\n(Real Data)")
            axes[3].axis('off')
            
            # 5. Error Map
            im = axes[4].imshow(error, cmap='jet', vmin=0, vmax=0.1)
            axes[4].set_title("Error Map\n(Dark Blue = Perfect)")
            axes[4].axis('off')
            
            plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
            plt.suptitle(f"Advanced Model Test #{i+1}", fontsize=16)
            plt.tight_layout()
            
            save_name = f"advanced_result_{i}.png"
            plt.savefig(save_name)
            print(f"Saved {save_name}")
            plt.show()

if __name__ == "__main__":
    test_on_unseen()