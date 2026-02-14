import torch
import numpy as np
import cv2
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# IMPORTS
from model import OpticalGuidedSR
from dataLoader import SatelliteThermalDataset
from torch.utils.data import DataLoader

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/final_model.pth"
DATASET_PATH = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"

def evaluate_model():
    print(f"evaluating model on {DEVICE}...")
    
    # 1. Load Model
    model = OpticalGuidedSR().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    model.eval()
    
    # 2. Load Test Data (Use a larger sample for valid stats, e.g., 200 images)
    dataset = SatelliteThermalDataset(DATASET_PATH, max_samples=200)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    print("Calculating Metrics...")
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Run Inference
            prediction = model(inputs)
            
            # Convert to Numpy for calculation (Range 0-1)
            # We squeeze dimensions: (1, 1, H, W) -> (H, W)
            pred_np = prediction.squeeze().cpu().numpy()
            target_np = targets.squeeze().cpu().numpy()
            
            # Clip values to [0, 1] to avoid math errors
            pred_np = np.clip(pred_np, 0, 1)
            target_np = np.clip(target_np, 0, 1)
            
            # Calculate PSNR
            # data_range=1.0 because our images are normalized 0-1
            current_psnr = psnr(target_np, pred_np, data_range=1.0)
            
            # Calculate SSIM
            current_ssim = ssim(target_np, pred_np, data_range=1.0)
            
            total_psnr += current_psnr
            total_ssim += current_ssim
            count += 1
            
    # 3. Final Results
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    
    print("-" * 30)
    print(f"FINAL RESULTS (n={count})")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("-" * 30)
    
    # Interpretation for you
    if avg_psnr > 30:
        print("Verdict: Excellent Reconstruction quality.")
    elif avg_psnr > 25:
        print("Verdict: Good, but could be sharper.")
    else:
        print("Verdict: Model needs more training.")

if __name__ == "__main__":
    evaluate_model()