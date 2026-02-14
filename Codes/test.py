import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# IMPORTS
from model import OpticalGuidedSR
from dataLoader import SatelliteThermalDataset
from torch.utils.data import DataLoader

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/final_model.pth" # Or "checkpoints/model_epoch_20.pth"
DATASET_PATH = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"

def visualize_results():
    print(f" Loading Model from {MODEL_PATH}...")
    
    # 1. Load Model
    model = OpticalGuidedSR().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set to evaluation mode (freezes BatchNorm/Dropout)
    
    # 2. Load One Batch of Data
    # We use the same loader but only take 1 batch
    dataset = SatelliteThermalDataset(DATASET_PATH, max_samples=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    
    # 3. Run Inference
    with torch.no_grad():
        predictions = model(inputs)
    
    # 4. Calculate Physics Metrics (RMSE)
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse).item()
    print(f" Root Mean Squared Error (RMSE) on this batch: {rmse:.4f}")
    
    # 5. VISUALIZATION LOOP (Show top 2 examples)
    for i in range(2):
        # Convert to Numpy for plotting
        # [0] = Thermal Input (Blurry), [1] = Optical
        low_res_in = inputs[i, 0].cpu().numpy() 
        optical_in = inputs[i, 1].cpu().numpy()
        prediction = predictions[i, 0].cpu().numpy()
        ground_truth = targets[i, 0].cpu().numpy()
        
        # Calculate Error Map (Difference)
        error_map = np.abs(ground_truth - prediction)
        
        # --- PLOTTING ---
        plt.figure(figsize=(15, 4))
        
        # 1. Low Res Input
        plt.subplot(1, 5, 1)
        plt.imshow(low_res_in, cmap='inferno')
        plt.title("Input (Low Res)\nStandard Camera")
        plt.axis('off')
        
        # 2. Optical Guidance
        plt.subplot(1, 5, 2)
        plt.imshow(optical_in, cmap='gray')
        plt.title("Optical Guidance\n(Structural Prior)")
        plt.axis('off')
        
        # 3. YOUR RESULT
        plt.subplot(1, 5, 3)
        plt.imshow(prediction, cmap='inferno')
        plt.title(f"PAGF-Net Output\n(Physics Enhanced)")
        plt.axis('off')

        # 4. Ground Truth
        plt.subplot(1, 5, 4)
        plt.imshow(ground_truth, cmap='inferno')
        plt.title("Ground Truth\n(Target)")
        plt.axis('off')
        
        # 5. Error Map (The Scientific Proof)
        plt.subplot(1, 5, 5)
        plt.imshow(error_map, cmap='jet', vmin=0, vmax=0.1) # 'jet' highlights errors
        plt.title("Error Map\n(Dark Blue = Perfect)")
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        
        # Save it for your slides
        save_name = f"result_slide_{i}.png"
        plt.savefig(save_name)
        print(f"Saved visualization to {save_name}")

if __name__ == "__main__":
    visualize_results()