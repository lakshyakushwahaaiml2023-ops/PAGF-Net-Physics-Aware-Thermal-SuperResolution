import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# Local Project Imports
from model_v2 import PAGFNetV2
from dataLoader import SatelliteThermalDataset

# --- SETTINGS ---
EVAL_CONFIG = {
    "ckpt": "checkpoints/final_pagf_v2.pth",
    "data_path": r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "split_idx": 4000,
    "test_n": 1000
}

def scientific_evaluation():
    print(f"[*] Starting Phase 2 Benchmark: PAGF-Net v2")
    
    # 1. Load Model
    net = PAGFNetV2().to(EVAL_CONFIG["device"])
    if os.path.exists(EVAL_CONFIG["ckpt"]):
        net.load_state_dict(torch.load(EVAL_CONFIG["ckpt"], map_location=EVAL_CONFIG["device"]))
        net.eval()
        print(f"[->] Successfully loaded weights from {EVAL_CONFIG['ckpt']}")
    else:
        print(f"[X] CRITICAL: Checkpoint not found at {EVAL_CONFIG['ckpt']}")
        return

    # 2. Dataset Slicing (Unseen split)
    full_ds = SatelliteThermalDataset(EVAL_CONFIG["data_path"], 
                                       max_samples=EVAL_CONFIG["split_idx"] + EVAL_CONFIG["test_n"])
    
    # Manually override the paths to only include the test subset
    full_ds.image_paths = full_ds.image_paths[EVAL_CONFIG["split_idx"]:]
    loader = DataLoader(full_ds, batch_size=1, shuffle=False)
    print(f"[->] Evaluated on {len(full_ds)} distinct unseen samples.")

    # 3. Metric Containers (Using lists for statistical significance)
    results = {
        'bic_p': [], 'bic_s': [],
        'net_p': [], 'net_s': []
    }

    # 4. Inference Loop
    print("\n[Running Benchmark]")
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs, targets = inputs.to(EVAL_CONFIG["device"]), targets.to(EVAL_CONFIG["device"])
            
            # Baseline: Bicubic channel
            img_bic = inputs[:, 0:1, :, :]
            
            # Prediction: Final SR output
            img_net, _ = net(inputs)
            
            # Move to CPU and Clip
            gt_np  = targets.squeeze().cpu().numpy().clip(0, 1)
            bic_np = img_bic.squeeze().cpu().numpy().clip(0, 1)
            net_np = img_net.squeeze().cpu().numpy().clip(0, 1)

            # Record PSNR
            results['bic_p'].append(compute_psnr(gt_np, bic_np, data_range=1.0))
            results['net_p'].append(compute_psnr(gt_np, net_np, data_range=1.0))
            
            # Record SSIM
            results['bic_s'].append(compute_ssim(gt_np, bic_np, data_range=1.0))
            results['net_s'].append(compute_ssim(gt_np, net_np, data_range=1.0))

    # 5. Reporting
    b_psnr_mu, b_psnr_std = np.mean(results['bic_p']), np.std(results['bic_p'])
    n_psnr_mu, n_psnr_std = np.mean(results['net_p']), np.std(results['net_p'])
    
    b_ssim_mu = np.mean(results['bic_s'])
    n_ssim_mu = np.mean(results['net_s'])

    gain_p = n_psnr_mu - b_psnr_mu
    gain_s = ((n_ssim_mu - b_ssim_mu) / b_ssim_mu) * 100

    print("\n" + "="*85)
    print(f"{'METRIC':<18} | {'BICUBIC BASELINE':<25} | {'PAGF-NET V2 (SFT)':<25} | {'GAIN'}")
    print("-" * 85)
    print(f"{'Avg PSNR (dB)':<18} | {b_psnr_mu:>8.4f} (±{b_psnr_std:.2f}) | {n_psnr_mu:>8.4f} (±{n_psnr_std:.2f}) | {gain_p:+.3f} dB")
    print(f"{'Avg SSIM':<18} | {b_ssim_mu:>15.4f} | {n_ssim_mu:>15.4f} | {gain_s:+.2f} %")
    print("="*85)

if __name__ == "__main__":
    scientific_evaluation()