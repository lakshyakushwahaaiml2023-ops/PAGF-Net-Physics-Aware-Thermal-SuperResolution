import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# Project-specific imports
from model import OpticalGuidedSR
from dataLoader import SatelliteThermalDataset

# --- SETTINGS ---
RUN_CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "ckpt_path": "checkpoints/final_model.pth",
    "data_root": r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark",
    "train_offset": 3000,  # Skip these as they were seen during training
    "test_count": 200      # Number of unseen samples to evaluate
}

def run_evaluation():
    print(f"--- Initializing Unseen Data Evaluation ---")
    
    # 1. Setup Model
    model = OpticalGuidedSR().to(RUN_CONFIG["device"])
    
    if not os.path.exists(RUN_CONFIG["ckpt_path"]):
        print(f"[!] Critical Error: Weights not found at {RUN_CONFIG['ckpt_path']}")
        return

    # Load weights with map_location to handle CPU/GPU swaps gracefully
    state_dict = torch.load(RUN_CONFIG["ckpt_path"], map_location=RUN_CONFIG["device"])
    model.load_state_dict(state_dict)
    model.eval()
    print(f"-> Model weights loaded successfully.")

    # 2. Data Preparation (Slicing for Unseen Split)
    total_samples = RUN_CONFIG["train_offset"] + RUN_CONFIG["test_count"]
    ds = SatelliteThermalDataset(RUN_CONFIG["data_root"], max_samples=total_samples)
    
    # Manual split to ensure zero overlap with training data
    if len(ds.image_paths) > RUN_CONFIG["train_offset"]:
        ds.image_paths = ds.image_paths[RUN_CONFIG["train_offset"]:]
        print(f"-> Testing on slice: [{RUN_CONFIG['train_offset']}:{total_samples}]")
        print(f"-> Sample count: {len(ds)}")
    else:
        print("[!] Warning: Dataset size smaller than offset. Split may be invalid.")

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    
    # 3. Evaluation Loop
    psnr_scores = []
    ssim_scores = []
    
    print("\nStarting inference...")
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating", unit="img"):
            inputs = inputs.to(RUN_CONFIG["device"])
            
            # Forward pass
            out = model(inputs)
            
            # Prep tensors for metrics (Squeeze and move to CPU)
            # Clamping is essential for floating point precision issues
            pred_img = out.squeeze().cpu().numpy().clip(0, 1)
            gt_img = targets.squeeze().cpu().numpy().clip(0, 1)
            
            # Metrics Calculation
            psnr_val = compute_psnr(gt_img, pred_img, data_range=1.0)
            ssim_val = compute_ssim(gt_img, pred_img, data_range=1.0, channel_axis=None)
            
            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)
            
    # 4. Results Reporting
    if not psnr_scores:
        print("No data was processed.")
        return

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    
    print("\n" + "#" * 30)
    print(f" TEST REPORT (n={len(psnr_scores)})")
    print(f" Mean PSNR: {avg_psnr:.3f} dB")
    print(f" Mean SSIM: {avg_ssim:.4f}")
    print("#" * 30)

    # Performance Grade
    if avg_psnr >= 32.0:
        verdict = "SOTA / Excellent"
    elif avg_psnr >= 28.0:
        verdict = "Strong / Competitive"
    else:
        verdict = "Baseline / Functional"
        
    print(f"Performance Grade: {verdict}\n")

if __name__ == "__main__":
    run_evaluation()