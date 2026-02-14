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
MODEL_PATH = "checkpoints/final_pagf_v2.pth"
DATASET_PATH = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"
TRAIN_SIZE = 4000

def visualize_v2():
    print("Generating Visual Proofs for PAGF-Net v2...")
    
    # Load Model
    model = PAGFNetV2().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Load Data (Skip training set)
    dataset = SatelliteThermalDataset(DATASET_PATH, max_samples=TRAIN_SIZE + 50)
    if len(dataset.image_paths) > TRAIN_SIZE:
        dataset.image_paths = dataset.image_paths[TRAIN_SIZE:]
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Visualization Loop (Save 3 examples)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            if i >= 3: break
            
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Inference
            pred_final, _ = model(inputs)
            
            # Prepare Images
            bicubic = inputs[0, 0].cpu().numpy()
            optical = inputs[0, 1].cpu().numpy()
            prediction = pred_final[0, 0].cpu().numpy()
            ground_truth = targets[0, 0].cpu().numpy()
            
            # Calculate Error Maps (Difference from Ground Truth)
            err_bic = np.abs(ground_truth - bicubic)
            err_net = np.abs(ground_truth - prediction)
            
            # PLOTTING
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            
            # Row 1: The Process
            axes[0,0].imshow(bicubic, cmap='inferno')
            axes[0,0].set_title("Input (Bicubic)\nBlurry & Low Res")
            
            axes[0,1].imshow(optical, cmap='gray')
            axes[0,1].set_title("Optical Guidance\n(Structural Prior)")
            
            axes[0,2].imshow(prediction, cmap='inferno')
            axes[0,2].set_title("PAGF-Net v2 Output\n(Sharpened & Restored)")
            
            # Row 2: The Proof (Error Maps)
            # Use same scale (vmax) for fair comparison
            max_err = max(err_bic.max(), err_net.max())
            
            axes[1,0].imshow(ground_truth, cmap='inferno')
            axes[1,0].set_title("Ground Truth\n(Target)")
            
            im1 = axes[1,1].imshow(err_bic, cmap='jet', vmin=0, vmax=max_err)
            axes[1,1].set_title("Bicubic Error\n(More Red = Bad)")
            plt.colorbar(im1, ax=axes[1,1], fraction=0.046)
            
            im2 = axes[1,2].imshow(err_net, cmap='jet', vmin=0, vmax=max_err)
            axes[1,2].set_title("PAGF-Net Error\n(Blue = Perfect)")
            plt.colorbar(im2, ax=axes[1,2], fraction=0.046)
            
            for ax in axes.flatten(): ax.axis('off')
            
            plt.suptitle(f"PAGF-Net v2 Result #{i+1}", fontsize=16)
            plt.tight_layout()
            plt.savefig(f"v2_result_{i}.png", dpi=300)
            print(f"Saved v2_result_{i}.png")
            plt.close()

if __name__ == "__main__":
    visualize_v2()