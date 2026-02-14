import torch
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- NEW IMPORTS ---
# We now import the ADVANCED model
from model_advanced import AdvancedOpticalSR 
from dataLoader import SatelliteThermalDataset

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Make sure this points to your NEW checkpoint
MODEL_PATH = "checkpoints/final_advanced_model.pth" 
DATASET_PATH = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"

# Validation split
TRAIN_SIZE = 3000
TEST_SIZE = 500

def compare_fair():
    print(f"Starting FAIR Baseline Comparison (Advanced Model)...")
    
    # 1. Load ADVANCED Model
    model = AdvancedOpticalSR().to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"Loaded checkpoint: {MODEL_PATH}")
    else:
        print(f"Checkpoint not found at {MODEL_PATH}")
        print("   Did you run train_advanced.py yet?")
        return

    # 2. Load Unseen Data
    full_dataset = SatelliteThermalDataset(DATASET_PATH, max_samples=TRAIN_SIZE + TEST_SIZE)
    
    # Slice to keep only unseen
    if len(full_dataset.image_paths) > TRAIN_SIZE:
        full_dataset.image_paths = full_dataset.image_paths[TRAIN_SIZE:]
        print(f"Verified: Testing on {len(full_dataset.image_paths)} unseen images.")
    
    dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False)
    
    # Metrics
    metrics = {
        'bicubic_psnr': [], 'bicubic_ssim': [],
        'model_psnr': [], 'model_ssim': []
    }
    
    print("running metrics...")
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # --- 1. Generate Bicubic Baseline ---
            # Input[0] is the Low-Res Thermal (already upsampled by Bicubic in the loader)
            bicubic_baseline = inputs[:, 0:1, :, :]
            
            # --- 2. Generate Model Prediction ---
            # IMPORTANT: The advanced model returns (final, mid)
            # We only care about 'final' for evaluation
            model_output, _ = model(inputs) 
            
            # --- 3. Convert to Numpy ---
            gt_np = targets.squeeze().cpu().numpy()
            bicubic_np = bicubic_baseline.squeeze().cpu().numpy()
            model_np = model_output.squeeze().cpu().numpy()
            
            # Clip [0, 1]
            gt_np = np.clip(gt_np, 0, 1)
            bicubic_np = np.clip(bicubic_np, 0, 1)
            model_np = np.clip(model_np, 0, 1)
            
            # --- 4. Calculate Scores ---
            # Bicubic
            metrics['bicubic_psnr'].append(psnr(gt_np, bicubic_np, data_range=1.0))
            metrics['bicubic_ssim'].append(ssim(gt_np, bicubic_np, data_range=1.0))
            
            # Model
            metrics['model_psnr'].append(psnr(gt_np, model_np, data_range=1.0))
            metrics['model_ssim'].append(ssim(gt_np, model_np, data_range=1.0))

    # --- FINAL REPORT ---
    avg_bicubic_psnr = np.mean(metrics['bicubic_psnr'])
    avg_model_psnr = np.mean(metrics['model_psnr'])
    avg_bicubic_ssim = np.mean(metrics['bicubic_ssim'])
    avg_model_ssim = np.mean(metrics['model_ssim'])
    
    imp_psnr = avg_model_psnr - avg_bicubic_psnr
    imp_ssim = ((avg_model_ssim - avg_bicubic_ssim) / avg_bicubic_ssim) * 100
    
    print("\n" + "="*60)
    print("FAIR SCIENTIFIC EVALUATION (Advanced SFT Model)")
    print("="*60)
    print(f"{'Metric':<15} | {'Bicubic':<20} | {'Advanced SR':<20} | {'Improvement'}")
    print("-" * 75)
    print(f"{'PSNR (dB)':<15} | {avg_bicubic_psnr:<20.4f} | {avg_model_psnr:<20.4f} | {imp_psnr:+.2f} dB")
    print(f"{'SSIM':<15} | {avg_bicubic_ssim:<20.4f} | {avg_model_ssim:<20.4f} | {imp_ssim:+.2f} %")
    print("="*60)

if __name__ == "__main__":
    compare_fair()