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
MODEL_PATH = "checkpoints_final/best_model_v2.pth"
DATASET_PATH = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"
TRAIN_SIZE = 4000

def predict_tta_full(model, inputs):
    """
    Runs inference on the Full Dihedral Group (D4): 8 Geometric Variations.
    This provides the maximum theoretical reduction in variance.
    """
    model.eval()
    preds = []
    
    # 1. Identity (Original)
    p1, _ = model(inputs)
    preds.append(p1)
    
    # 2. Horizontal Flip
    in_h = torch.flip(inputs, [3])
    p2, _ = model(in_h)
    preds.append(torch.flip(p2, [3]))
    
    # 3. Vertical Flip
    in_v = torch.flip(inputs, [2])
    p3, _ = model(in_v)
    preds.append(torch.flip(p3, [2]))
    
    # 4. Rotate 90
    in_r90 = torch.rot90(inputs, 1, [2, 3])
    p4, _ = model(in_r90)
    preds.append(torch.rot90(p4, 3, [2, 3]))
    
    # 5. Rotate 180
    in_r180 = torch.rot90(inputs, 2, [2, 3])
    p5, _ = model(in_r180)
    preds.append(torch.rot90(p5, 2, [2, 3]))
    
    # 6. Rotate 270
    in_r270 = torch.rot90(inputs, 3, [2, 3])
    p6, _ = model(in_r270)
    preds.append(torch.rot90(p6, 1, [2, 3]))

    # --- NEW ADDITIONS (Diagonal Symmetries) ---
    
    # 7. Horizontal Flip + Rotate 90 (Diagonal 1)
    in_hr90 = torch.rot90(torch.flip(inputs, [3]), 1, [2,3])
    p7, _ = model(in_hr90)
    preds.append(torch.flip(torch.rot90(p7, 3, [2,3]), [3]))

    # 8. Vertical Flip + Rotate 90 (Diagonal 2)
    in_vr90 = torch.rot90(torch.flip(inputs, [2]), 1, [2,3])
    p8, _ = model(in_vr90)
    preds.append(torch.flip(torch.rot90(p8, 3, [2,3]), [2]))
    
    # --- ENSEMBLE AVERAGE ---
    stack = torch.stack(preds, dim=0)
    final_pred = torch.mean(stack, dim=0)
    
    return final_pred

def run_full_evaluation():
    print(f" LAUNCHING FINAL D4 ENSEMBLE EVALUATION (8-Point TTA)")
    print(f"   • Model: {MODEL_PATH}")
    print("-" * 60)
    
    # Load Model
    model = PAGFNetV2().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(" Model loaded.")
    else:
        print(f" Error: {MODEL_PATH} not found.")
        return

    # Load Data
    dataset = SatelliteThermalDataset(DATASET_PATH, max_samples=TRAIN_SIZE + 500)
    if len(dataset.image_paths) > TRAIN_SIZE:
        dataset.image_paths = dataset.image_paths[TRAIN_SIZE:]
        print(f" Analyzing {len(dataset.image_paths)} unseen images.")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    bic_psnr, bic_ssim = [], []
    tta_psnr, tta_ssim = [], []
    
    print("Running 8-Point Inference (This is the slow, precise part)...")
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Baseline
            bicubic = inputs[:, 0:1, :, :]
            
            # Full 8-Point TTA
            pred = predict_tta_full(model, inputs)
            
            # Metrics
            gt_np = np.clip(targets.squeeze().cpu().numpy(), 0, 1)
            bic_np = np.clip(bicubic.squeeze().cpu().numpy(), 0, 1)
            pred_np = np.clip(pred.squeeze().cpu().numpy(), 0, 1)
            
            bic_psnr.append(psnr(gt_np, bic_np, data_range=1.0))
            bic_ssim.append(ssim(gt_np, bic_np, data_range=1.0))
            
            tta_psnr.append(psnr(gt_np, pred_np, data_range=1.0))
            tta_ssim.append(ssim(gt_np, pred_np, data_range=1.0))

    # --- FINAL REPORT ---
    b_p = np.mean(bic_psnr); b_s = np.mean(bic_ssim)
    t_p = np.mean(tta_psnr); t_s = np.mean(tta_ssim)
    
    print("\n" + "="*85)
    print(" FINAL SYMPOSIUM BENCHMARK (Full D4 Ensemble)")
    print("="*85)
    print(f"{'Method':<25} | {'PSNR (dB)':<20} | {'SSIM':<20} | {'Improvement'}")
    print("-" * 85)
    print(f"{'Bicubic Baseline':<25} | {b_p:<20.4f} | {b_s:<20.4f} | -")
    print(f"{'PAGF-Net v2 (8x TTA)':<25} | {t_p:<20.4f} | {t_s:<20.4f} | +{t_p - b_p:.2f} dB")
    print("="*85)

if __name__ == "__main__":
    run_full_evaluation()