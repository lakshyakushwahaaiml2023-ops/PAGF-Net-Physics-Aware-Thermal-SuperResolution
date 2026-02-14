import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# IMPORTS
from model_v2 import PAGFNetV2  # <--- Using V2 Model
from dataLoader import SatelliteThermalDataset

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/final_pagf_v2.pth"
DATASET_PATH = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"

# Test Split (Using the last 1000 images as unseen test set)
TRAIN_SIZE = 4000 
TEST_SIZE = 1000

def evaluate_v2():
    print(f"Starting SCIENTIFIC EVALUATION for PAGF-Net v2...")
    
    # 1. Load Model
    model = PAGFNetV2().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"Loaded Checkpoint: {MODEL_PATH}")
    else:
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # 2. Load Data
    full_dataset = SatelliteThermalDataset(DATASET_PATH, max_samples=TRAIN_SIZE + TEST_SIZE)
    if len(full_dataset.image_paths) > TRAIN_SIZE:
        full_dataset.image_paths = full_dataset.image_paths[TRAIN_SIZE:]
        print(f"Testing on {len(full_dataset.image_paths)} unseen images.")
    
    dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False)
    
    # Metrics Storage
    metrics = {
        'bic_psnr': [], 'bic_ssim': [],
        'net_psnr': [], 'net_ssim': []
    }
    
    print("Running Inference Loop...")
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # --- BASELINE ---
            # Input channel 0 is the bicubic upsampled version
            bicubic = inputs[:, 0:1, :, :]
            
            # --- MODEL INFERENCE ---
            # V2 returns (Final, Mid). We only want Final.
            pred_final, _ = model(inputs)
            
            # --- CONVERT TO NUMPY ---
            gt_np = np.clip(targets.squeeze().cpu().numpy(), 0, 1)
            bic_np = np.clip(bicubic.squeeze().cpu().numpy(), 0, 1)
            pred_np = np.clip(pred_final.squeeze().cpu().numpy(), 0, 1)
            
            # --- CALCULATE SCORES ---
            # Bicubic Scores
            metrics['bic_psnr'].append(psnr(gt_np, bic_np, data_range=1.0))
            metrics['bic_ssim'].append(ssim(gt_np, bic_np, data_range=1.0))
            
            # Network Scores
            metrics['net_psnr'].append(psnr(gt_np, pred_np, data_range=1.0))
            metrics['net_ssim'].append(ssim(gt_np, pred_np, data_range=1.0))

    # --- FINAL TABLE ---
    avg_bic_psnr = np.mean(metrics['bic_psnr'])
    avg_net_psnr = np.mean(metrics['net_psnr'])
    avg_bic_ssim = np.mean(metrics['bic_ssim'])
    avg_net_ssim = np.mean(metrics['net_ssim'])
    
    imp_psnr = avg_net_psnr - avg_bic_psnr
    imp_ssim = ((avg_net_ssim - avg_bic_ssim) / avg_bic_ssim) * 100
    
    print("\n" + "="*80)
    print(" FINAL RESULTS: PAGF-Net v2 (Physics-Aware)")
    print("="*80)
    print(f"{'Metric':<15} | {'Bicubic (Baseline)':<20} | {'PAGF-Net v2':<20} | {'Improvement'}")
    print("-" * 80)
    print(f"{'PSNR (dB)':<15} | {avg_bic_psnr:<20.4f} | {avg_net_psnr:<20.4f} | {imp_psnr:+.2f} dB")
    print(f"{'SSIM':<15} | {avg_bic_ssim:<20.4f} | {avg_net_ssim:<20.4f} | {imp_ssim:+.2f} %")
    print("="*80)

if __name__ == "__main__":
    evaluate_v2()