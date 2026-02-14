import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader

# IMPORTS
from model_v2 import PAGFNetV2
from dataLoader import SatelliteThermalDataset

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints_final/best_model_v2.pth"
DATASET_PATH = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"
TRAIN_SIZE = 4000

def generate_visuals():
    print(f"GENERATING FINAL VISUAL PROOFS")
    
    # Load Model
    model = PAGFNetV2().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    else:
        print(f"Checkpoint not found: {MODEL_PATH}")
        return

    # Load Data (Randomly shuffle to find good examples)
    dataset = SatelliteThermalDataset(DATASET_PATH, max_samples=TRAIN_SIZE + 100)
    if len(dataset.image_paths) > TRAIN_SIZE:
        dataset.image_paths = dataset.image_paths[TRAIN_SIZE:]
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Save 3 Best Examples
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            if count >= 3: break
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Inference
            pred, _ = model(inputs)
            
            # Extract Images
            bicubic = inputs[0, 0].cpu().numpy()
            optical = inputs[0, 1].cpu().numpy()
            ours = pred[0, 0].cpu().numpy()
            gt = targets[0, 0].cpu().numpy()
            
            # --- PLOTTING ---
            fig, axes = plt.subplots(1, 4, figsize=(20, 6))
            
            # 1. Bicubic
            axes[0].imshow(bicubic, cmap='inferno')
            axes[0].set_title(f"Bicubic (Baseline)\nBlurry", fontsize=14)
            axes[0].axis('off')
            
            # 2. Optical
            axes[1].imshow(optical, cmap='gray')
            axes[1].set_title(f"Optical Guidance\n(High Frequency Detail)", fontsize=14)
            axes[1].axis('off')
            
            # 3. Our Model
            axes[2].imshow(ours, cmap='inferno')
            axes[2].set_title(f"PAGF-Net v2\n(Restored)", fontsize=14, fontweight='bold', color='darkblue')
            axes[2].axis('off')
            
            # 4. Ground Truth
            axes[3].imshow(gt, cmap='inferno')
            axes[3].set_title(f"Ground Truth\n(Target)", fontsize=14)
            axes[3].axis('off')
            
            plt.suptitle(f"Final Model Evaluation - Example {count+1}", fontsize=20)
            plt.tight_layout()
            
            filename = f"FINAL_RESULT_{count+1}.png"
            plt.savefig(filename, dpi=300)
            print(f"Saved high-res image: {filename}")
            
            count += 1
            plt.close()

if __name__ == "__main__":
    generate_visuals()