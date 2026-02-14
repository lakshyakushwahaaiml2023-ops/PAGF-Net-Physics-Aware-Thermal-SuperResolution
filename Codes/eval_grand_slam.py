import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# IMPORTS
from model import OpticalGuidedSR          # Old Model
from model_advanced import AdvancedOpticalSR # New Model
from dataLoader import SatelliteThermalDataset

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLD_MODEL_PATH = "checkpoints/final_model.pth"
NEW_MODEL_PATH = "checkpoints/final_advanced_model.pth"
DATASET_PATH = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"

# FULL SCALE EVALUATION
TRAIN_SIZE = 4000
TEST_SIZE = 1000 # Test on 1000 images for statistical significance

def grand_slam_eval():
    print(f"Starting GRAND SLAM Benchmark (Bicubic vs Basic vs Advanced)...")
    
    # 1. Load Models
    model_old = OpticalGuidedSR().to(DEVICE)
    model_new = AdvancedOpticalSR().to(DEVICE)
    
    # Load Weights
    if os.path.exists(OLD_MODEL_PATH):
        model_old.load_state_dict(torch.load(OLD_MODEL_PATH, map_location=DEVICE))
        print("Basic Model loaded.")
    else:
        print("Basic Model checkpoint missing. Skipping.")
        model_old = None

    if os.path.exists(NEW_MODEL_PATH):
        model_new.load_state_dict(torch.load(NEW_MODEL_PATH, map_location=DEVICE))
        print("Advanced Model loaded.")
    else:
        print("Advanced Model checkpoint missing. Skipping.")
        model_new = None
        
    if model_old: model_old.eval()
    if model_new: model_new.eval()

    # 2. Data Loader (Full Scale)
    full_dataset = SatelliteThermalDataset(DATASET_PATH, max_samples=TRAIN_SIZE + TEST_SIZE)
    if len(full_dataset.image_paths) > TRAIN_SIZE:
        full_dataset.image_paths = full_dataset.image_paths[TRAIN_SIZE:]
        print(f"Evaluating on {len(full_dataset.image_paths)} unseen scenes.")
    
    dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False)
    
    # Metrics Dictionary
    results = {'bicubic': {'psnr': [], 'ssim': []},
               'basic':   {'psnr': [], 'ssim': []},
               'advanced':{'psnr': [], 'ssim': []}}
    
    print("Running inference marathon...")
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # --- BASELINE: Bicubic ---
            # Input channel 0 is already bicubic upsampled
            bicubic = inputs[:, 0:1, :, :]
            
            # --- INFERENCE ---
            # Basic Model
            if model_old:
                pred_basic = model_old(inputs)
            
            # Advanced Model (Returns tuple, take first)
            if model_new:
                pred_advanced, _ = model_new(inputs)
            
            # --- CONVERT TO NUMPY ---
            gt_np = np.clip(targets.squeeze().cpu().numpy(), 0, 1)
            bic_np = np.clip(bicubic.squeeze().cpu().numpy(), 0, 1)
            
            # Record Bicubic
            results['bicubic']['psnr'].append(psnr(gt_np, bic_np, data_range=1.0))
            results['bicubic']['ssim'].append(ssim(gt_np, bic_np, data_range=1.0))
            
            # Record Basic
            if model_old:
                bas_np = np.clip(pred_basic.squeeze().cpu().numpy(), 0, 1)
                results['basic']['psnr'].append(psnr(gt_np, bas_np, data_range=1.0))
                results['basic']['ssim'].append(ssim(gt_np, bas_np, data_range=1.0))
                
            # Record Advanced
            if model_new:
                adv_np = np.clip(pred_advanced.squeeze().cpu().numpy(), 0, 1)
                results['advanced']['psnr'].append(psnr(gt_np, adv_np, data_range=1.0))
                results['advanced']['ssim'].append(ssim(gt_np, adv_np, data_range=1.0))

    # --- FINAL REPORT ---
    print("\n" + "="*80)
    print(f"{'MODEL ARCHITECTURE':<25} | {'PSNR (dB)':<15} | {'SSIM':<15} | {'STATUS'}")
    print("-" * 80)
    
    # Report Bicubic
    b_psnr = np.mean(results['bicubic']['psnr'])
    b_ssim = np.mean(results['bicubic']['ssim'])
    print(f"{'1. Bicubic Interpolation':<25} | {b_psnr:<15.4f} | {b_ssim:<15.4f} | {'Baseline'}")
    
    # Report Basic
    if model_old:
        o_psnr = np.mean(results['basic']['psnr'])
        o_ssim = np.mean(results['basic']['ssim'])
        gain = o_psnr - b_psnr
        print(f"{'2. Basic CNN':<25} | {o_psnr:<15.4f} | {o_ssim:<15.4f} | +{gain:.2f} dB")
        
    # Report Advanced
    if model_new:
        n_psnr = np.mean(results['advanced']['psnr'])
        n_ssim = np.mean(results['advanced']['ssim'])
        gain = n_psnr - b_psnr
        gain_vs_old = n_psnr - o_psnr if model_old else 0
        print(f"{'3. Advanced SFT-Net':<25} | {n_psnr:<15.4f} | {n_ssim:<15.4f} | +{gain:.2f} dB (Best)")

    print("="*80)
    
    if model_new and model_old:
        if n_psnr > o_psnr:
            print(f"SUCCESS: Architectural improvements added +{n_psnr - o_psnr:.2f} dB over your previous model.")
        else:
            print("NOTE: Advanced model did not outperform basic. Check training epochs.")

if __name__ == "__main__":
    grand_slam_eval()