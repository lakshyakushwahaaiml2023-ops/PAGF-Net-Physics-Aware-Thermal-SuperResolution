import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from model_advanced import AdvancedOpticalSR 
from dataLoader import SatelliteThermalDataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/final_advanced_model.pth"
DATASET_PATH = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"

def visualize_zoom():
    # Load Model
    model = AdvancedOpticalSR().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Load Data
    dataset = SatelliteThermalDataset(DATASET_PATH, max_samples=4050)
    # Get a sample from deep in the test set
    if len(dataset) > 4000:
        dataset.image_paths = dataset.image_paths[4000:]
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    inputs, targets = next(iter(loader))
    inputs = inputs.to(DEVICE)
    
    # Inference
    with torch.no_grad():
        pred, _ = model(inputs)
    
    # Prepare Images
    bicubic = inputs[0, 0].cpu().numpy()
    optical = inputs[0, 1].cpu().numpy()
    ours = pred[0, 0].cpu().numpy()
    gt = targets[0, 0].cpu().numpy()
    
    # --- ZOOM LOGIC ---
    # Crop center 50x50
    h, w = bicubic.shape
    cy, cx = h//2, w//2
    size = 25
    
    y1, y2 = cy-size, cy+size
    x1, x2 = cx-size, cx+size
    
    # Extract Crops
    crop_bic = bicubic[y1:y2, x1:x2]
    crop_opt = optical[y1:y2, x1:x2]
    crop_ours = ours[y1:y2, x1:x2]
    crop_gt = gt[y1:y2, x1:x2]
    
    # --- PLOT ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Full Image
    axes[0,0].imshow(bicubic, cmap='inferno'); axes[0,0].set_title("Full Bicubic")
    axes[0,1].imshow(optical, cmap='gray');    axes[0,1].set_title("Full Optical")
    axes[0,2].imshow(ours, cmap='inferno');    axes[0,2].set_title("Full Advanced SR")
    axes[0,3].imshow(gt, cmap='inferno');      axes[0,3].set_title("Full Ground Truth")
    
    # Row 2: Zoomed Crops (The Proof)
    axes[1,0].imshow(crop_bic, cmap='inferno'); axes[1,0].set_title("Zoom: Blurry Edges")
    axes[1,1].imshow(crop_opt, cmap='gray');    axes[1,1].set_title("Zoom: Structural Guidance")
    axes[1,2].imshow(crop_ours, cmap='inferno'); axes[1,2].set_title("Zoom: Recovered Edges")
    axes[1,3].imshow(crop_gt, cmap='inferno');   axes[1,3].set_title("Zoom: Ground Truth")
    
    # Remove axis ticks for cleanliness
    for ax in axes.flatten(): ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("zoom_comparison.png", dpi=300)
    print("Saved high-res comparison: zoom_comparison.png")
    plt.show()

if __name__ == "__main__":
    visualize_zoom()