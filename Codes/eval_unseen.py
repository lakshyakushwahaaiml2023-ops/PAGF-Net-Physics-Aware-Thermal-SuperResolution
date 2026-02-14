import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# IMPORTS
from model import OpticalGuidedSR
from dataLoader import SatelliteThermalDataset

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/final_model.pth"
DATASET_PATH = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"

# CRITICAL: Must match your training run
TRAIN_SIZE = 3000 
TEST_SIZE = 200  # How many unseen images to test

def evaluate_unseen():
    print(f" Starting Evaluation on UNSEEN Data...")
    
    # 1. Load Model
    model = OpticalGuidedSR().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    else:
        print(f" Error: Model not found at {MODEL_PATH}")
        return

    # 2. Prepare UNSEEN Data
    # We load enough data to cover Train + Test, then slice it
    total_needed = TRAIN_SIZE + TEST_SIZE
    full_dataset = SatelliteThermalDataset(DATASET_PATH, max_samples=total_needed)
    
    # SLICE: Keep only the data AFTER the training set
    if len(full_dataset.image_paths) > TRAIN_SIZE:
        full_dataset.image_paths = full_dataset.image_paths[TRAIN_SIZE:]
        print(f" Verified: Testing on {len(full_dataset.image_paths)} images (Indices {TRAIN_SIZE} to {TRAIN_SIZE+TEST_SIZE})")
    else:
        print(" Warning: Not enough data for a clean split. Testing on available data.")

    dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False)
    
    # 3. Metrics Loop
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    print("running metrics...")
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Inference
            prediction = model(inputs)
            
            # Post-process for Metrics (Convert to Numpy 0-1)
            pred_np = prediction.squeeze().cpu().numpy()
            target_np = targets.squeeze().cpu().numpy()
            
            # Clip to valid range [0, 1]
            pred_np = np.clip(pred_np, 0, 1)
            target_np = np.clip(target_np, 0, 1)
            
            # Calculate
            current_psnr = psnr(target_np, pred_np, data_range=1.0)
            current_ssim = ssim(target_np, pred_np, data_range=1.0)
            
            total_psnr += current_psnr
            total_ssim += current_ssim
            count += 1
            
    # 4. Final Report
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    
    print("\n" + "="*40)
    print(f"FINAL UNSEEN TEST RESULTS (n={count})")
    print(f"PSNR: {avg_psnr:.4f} dB  (Target: >30)")
    print(f"SSIM: {avg_ssim:.4f}     (Target: >0.85)")
    print("="*40)

    # Automatic Judge Interpretation
    if avg_psnr > 32:
        print("Verdict: STATE OF THE ART performance.")
    elif avg_psnr > 28:
        print("Verdict: COMPETITIVE performance.")
    else:
        print("Verdict: Proof of Concept working.")

if __name__ == "__main__":
    evaluate_unseen()