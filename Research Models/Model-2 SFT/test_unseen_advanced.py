import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Project Modules
from model_advanced import AdvancedOpticalSR 
from dataLoader import SatelliteThermalDataset

# --- SETTINGS ---
EVAL_CONFIG = {
    "ckpt": "checkpoints/final_advanced_model.pth",
    "data_path": r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "train_split_idx": 4000,
    "num_visuals": 3
}

def run_advanced_inference():
    print(f"[*] Initializing Advanced Model Inference...")

    # 1. Dataset Preparation
    # Load slightly more than the train split to get a fresh batch of unseen data
    ds = SatelliteThermalDataset(EVAL_CONFIG["data_path"], 
                                  max_samples=EVAL_CONFIG["train_split_idx"] + 50)
    
    # Filter for truly unseen data
    unseen_paths = ds.image_paths[EVAL_CONFIG["train_split_idx"]:]
    if not unseen_paths:
        print("[!] No unseen data found. Check your dataset size and split index.")
        return
    
    ds.image_paths = unseen_paths
    print(f"[->] Evaluated on {len(ds)} fresh samples (Index {EVAL_CONFIG['train_split_idx']}+)")

    loader = DataLoader(ds, batch_size=1, shuffle=True)

    # 2. Model Loading
    net = AdvancedOpticalSR().to(EVAL_CONFIG["device"])
    
    if os.path.exists(EVAL_CONFIG["ckpt"]):
        state = torch.load(EVAL_CONFIG["ckpt"], map_location=EVAL_CONFIG["device"])
        net.load_state_dict(state)
        net.eval()
        print(f"[->] Loaded Advanced Weights from {EVAL_CONFIG['ckpt']}")
    else:
        print(f"[X] Weights not found at {EVAL_CONFIG['ckpt']}. Check path.")
        return

    # 3. Inference & Plotting
    print("Inference in progress...")
    
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(loader):
            if idx >= EVAL_CONFIG["num_visuals"]:
                break
            
            inputs, targets = inputs.to(EVAL_CONFIG["device"]), targets.to(EVAL_CONFIG["device"])
            
            # SFT Model returns (Final_SR, Mid_SR)
            sr_final, _ = net(inputs)
            
            # Convert tensors to numpy for Matplotlib
            # [0,0] extracts first batch, first channel
            img_bicubic = inputs[0, 0].cpu().numpy()
            img_optical = inputs[0, 1].cpu().numpy()
            img_sr      = sr_final[0, 0].cpu().numpy()
            img_gt      = targets[0, 0].cpu().numpy()
            img_error   = np.abs(img_gt - img_sr)

            # --- VISUALIZATION BLOCK ---
            fig, ax = plt.subplots(1, 5, figsize=(22, 5), dpi=100)
            
            data = [img_bicubic, img_optical, img_sr, img_gt, img_error]
            labels = ["Bicubic (Baseline)", "Optical (Prior)", "SFT-SR (Advanced)", "Ground Truth", "Error (GT-SR)"]
            cmaps = ["inferno", "gray", "inferno", "inferno", "jet"]

            for i in range(5):
                curr_im = ax[i].imshow(data[i], cmap=cmaps[i])
                ax[i].set_title(labels[i], fontsize=11, pad=10)
                ax[i].axis("off")
                
                if i == 4: # Error Map colorbar
                    plt.colorbar(curr_im, ax=ax[i], fraction=0.046, pad=0.04)

            plt.suptitle(f"Qualitative Analysis: Sample {idx+1}", fontsize=15, y=1.02)
            plt.tight_layout()
            
            fn = f"val_result_{idx}.png"
            plt.savefig(fn, bbox_inches='tight')
            print(f"[+] Saved comparison: {fn}")
            plt.show()

if __name__ == "__main__":
    run_advanced_inference()