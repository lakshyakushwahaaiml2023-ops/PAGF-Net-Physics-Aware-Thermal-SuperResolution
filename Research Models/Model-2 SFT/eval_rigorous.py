import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# Research Modules
from model_advanced import AdvancedOpticalSR 
from dataLoader import SatelliteThermalDataset

# --- EVALUATION SETTINGS ---
EVAL_PARAMS = {
    "ckpt_path": "checkpoints/final_advanced_model.pth",
    "data_root": r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "train_offset": 3000,
    "test_samples": 500
}

def run_scientific_comparison():
    print(f"[*] Initializing Fair Baseline Comparison")
    
    # 1. Model Loading
    net = AdvancedOpticalSR().to(EVAL_PARAMS["device"])
    
    if os.path.exists(EVAL_PARAMS["ckpt_path"]):
        net.load_state_dict(torch.load(EVAL_PARAMS["ckpt_path"], map_location=EVAL_PARAMS["device"]))
        net.eval()
        print(f"[*] Advanced Model loaded successfully from {EVAL_PARAMS['ckpt_path']}")
    else:
        print(f"[!] Error: Weights not found. Run training first.")
        return

    # 2. Data Preparation (Slicing for Unseen split)
    full_ds = SatelliteThermalDataset(EVAL_PARAMS["data_root"], 
                                       max_samples=EVAL_PARAMS["train_offset"] + EVAL_PARAMS["test_samples"])
    
    # Isolate the validation/test slice
    full_ds.image_paths = full_ds.image_paths[EVAL_PARAMS["train_offset"]:]
    loader = DataLoader(full_ds, batch_size=1, shuffle=False)
    print(f"[*] Validating on {len(full_ds)} unseen image pairs.")

    # 3. Metric Containers
    # Using lists to calculate standard deviation later
    stats = {
        'bicubic_p': [], 'bicubic_s': [],
        'sft_p': [], 'sft_s': []
    }

    # 4. Evaluation Loop
    print("\n[Running Metric Computation]")
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs, targets = inputs.to(EVAL_PARAMS["device"]), targets.to(EVAL_PARAMS["device"])
            
            # --- BASELINE (Bicubic) ---
            # Channel 0 of the input is already bicubic-upsampled thermal
            img_bicubic = inputs[:, 0:1, :, :]
            
            # --- MODEL (Advanced SFT) ---
            # Advanced model returns (Final, Mid) -> we take Final
            img_sft, _ = net(inputs)
            
            # Conversion to Numpy for skimage metrics
            # Detach, move to CPU, squeeze batch/channel, and clip to valid range
            gt_np = targets.squeeze().cpu().numpy().clip(0, 1)
            bic_np = img_bicubic.squeeze().cpu().numpy().clip(0, 1)
            sft_np = img_sft.squeeze().cpu().numpy().clip(0, 1)

            # Record Scores
            stats['bicubic_p'].append(compute_psnr(gt_np, bic_np, data_range=1.0))
            stats['bicubic_s'].append(compute_ssim(gt_np, bic_np, data_range=1.0))
            
            stats['sft_p'].append(compute_psnr(gt_np, sft_np, data_range=1.0))
            stats['sft_s'].append(compute_ssim(gt_np, sft_np, data_range=1.0))

    # 5. Scientific Reporting
    def get_summary(key_p, key_s):
        return np.mean(stats[key_p]), np.std(stats[key_p]), np.mean(stats[key_s])

    b_mu_p, b_std_p, b_mu_s = get_summary('bicubic_p', 'bicubic_s')
    s_mu_p, s_std_p, s_mu_s = get_summary('sft_p', 'sft_s')

    print("\n" + "="*70)
    print(f"{'METRIC':<18} | {'BICUBIC BASELINE':<20} | {'ADVANCED SFT-SR':<20}")
    print("-" * 70)
    print(f"{'PSNR (dB) ↑':<18} | {b_mu_p:>7.3f} (±{b_std_p:.2f})  | {s_mu_p:>7.3f} (±{s_std_p:.2f})")
    print(f"{'SSIM (0-1) ↑':<18} | {b_mu_s:>17.4f} | {s_mu_s:>17.4f}")
    print("-" * 70)
    
    # Delta Calculation
    print(f"PSNR GAIN:  {s_mu_p - b_mu_p:+.3f} dB")
    print(f"SSIM GAIN:  {((s_mu_s - b_mu_s)/b_mu_s)*100:+.2f}% improvement")
    print("="*70)

if __name__ == "__main__":
    run_scientific_comparison()