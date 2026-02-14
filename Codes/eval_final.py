import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- IMPORTS ---
from model_v2 import PAGFNetV2
from dataLoader import SatelliteThermalDataset

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# THIS IS THE KEY: Point to your overnight best model
MODEL_PATH = "checkpoints_final/best_model_v2.pth" 
DATASET_PATH = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"

# We test on the LAST 500 images to ensure they are 100% unseen
TRAIN_SIZE = 4000 

def final_evaluation():
    print(f" STARTING FINAL SCIENTIFIC EVALUATION")
    print(f"   • Model: {MODEL_PATH}")
    print("-" * 60)
    
    # 1. Load Model
    model = PAGFNetV2().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.eval()
        print(" Model weights loaded successfully.")
    else:
        print(f" CRITICAL ERROR: Could not find {MODEL_PATH}")
        print("   Did the overnight training finish?")
        return

    # 2. Load Test Data
    dataset = SatelliteThermalDataset(DATASET_PATH, max_samples=TRAIN_SIZE + 500)
    if len(dataset.image_paths) > TRAIN_SIZE:
        dataset.image_paths = dataset.image_paths[TRAIN_SIZE:]
        print(f"Evaluating on {len(dataset.image_paths)} unseen test images.")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 3. Metrics
    bic_psnr_list, bic_ssim_list = [], []
    our_psnr_list, our_ssim_list = [], []
    
    print("Running Inference...")
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # A. Baseline (Bicubic)
            # Input channel 0 is already the bicubic upsampled version
            bicubic = inputs[:, 0:1, :, :]
            
            # B. Our Model (PAGF-Net v2)
            pred, _ = model(inputs)
            
            # C. Convert to Numpy & Clip
            gt_np = np.clip(targets.squeeze().cpu().numpy(), 0, 1)
            bic_np = np.clip(bicubic.squeeze().cpu().numpy(), 0, 1)
            pred_np = np.clip(pred.squeeze().cpu().numpy(), 0, 1)
            
            # D. Score
            # Bicubic
            bic_psnr_list.append(psnr(gt_np, bic_np, data_range=1.0))
            bic_ssim_list.append(ssim(gt_np, bic_np, data_range=1.0))
            
            # Ours
            our_psnr_list.append(psnr(gt_np, pred_np, data_range=1.0))
            our_ssim_list.append(ssim(gt_np, pred_np, data_range=1.0))

    # 4. Final Calculation
    avg_bic_psnr = np.mean(bic_psnr_list)
    avg_our_psnr = np.mean(our_psnr_list)
    
    avg_bic_ssim = np.mean(bic_ssim_list)
    avg_our_ssim = np.mean(our_ssim_list)
    
    imp_psnr = avg_our_psnr - avg_bic_psnr
    imp_ssim = ((avg_our_ssim - avg_bic_ssim) / avg_bic_ssim) * 100

    # 5. Print The "Winning Table"
    print("\n" + "="*85)
    print(" FINAL RESULTS: PAGF-Net v2 (Overnight Run)")
    print("="*85)
    print(f"{'METRIC':<20} | {'BASELINE (Bicubic)':<20} | {'OUR MODEL':<20} | {'IMPROVEMENT'}")
    print("-" * 85)
    print(f"{'PSNR (dB)':<20} | {avg_bic_psnr:<20.4f} | {avg_our_psnr:<20.4f} | +{imp_psnr:.2f} dB")
    print(f"{'SSIM (Structure)':<20} | {avg_bic_ssim:<20.4f} | {avg_our_ssim:<20.4f} | +{imp_ssim:.2f} %")
    print("="*85)
    
    if imp_psnr > 1.0:
        print(" CONCLUSION: Statistically Significant Improvement Verified.")
    else:
        print(" CONCLUSION: Marginal Improvement. Check data quality.")

if __name__ == "__main__":
    final_evaluation()