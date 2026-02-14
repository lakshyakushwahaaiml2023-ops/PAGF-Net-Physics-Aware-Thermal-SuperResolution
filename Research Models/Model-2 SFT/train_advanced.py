import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Research Modules
from dataLoader import SatelliteThermalDataset
from model_advanced import AdvancedOpticalSR
from loss_advanced_laplacian import TotalLoss

# --- RUN SETTINGS ---
HYPERPARAMS = {
    "lr": 1e-4,
    "batch_size": 8,  # SFT layers are memory intensive
    "epochs": 15,
    "aux_weight": 0.5, # Weight for mid-stage supervision
    "ckpt_dir": "checkpoints/advanced",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

os.makedirs(HYPERPARAMS["ckpt_dir"], exist_ok=True)

def train_phase_advanced():
    print(f"[*] Starting Advanced SFT Training Mode")
    print(f"[*] Target Device: {HYPERPARAMS['device']}")

    # 1. Data Setup
    data_root = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"
    ds = SatelliteThermalDataset(data_root, max_samples=4000)
    loader = DataLoader(ds, batch_size=HYPERPARAMS["batch_size"], shuffle=True, num_workers=0)

    # 2. Model & Optimization
    model = AdvancedOpticalSR().to(HYPERPARAMS["device"])
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS["lr"])
    
    # Custom Laplacian/Physics Loss
    criterion_main = TotalLoss().to(HYPERPARAMS["device"])
    criterion_aux = nn.L1Loss() # Simple L1 for the intermediate stage

    # 3. Training Loop
    for epoch in range(HYPERPARAMS["epochs"]):
        model.train()
        total_epoch_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{HYPERPARAMS['epochs']}")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(HYPERPARAMS["device"]), targets.to(HYPERPARAMS["device"])
            
            # The model takes Bicubic-upsampled Thermal + HR Optical
            # For energy conservation, we reference the upsampled thermal input
            thermal_ref = inputs[:, 0:1, :, :]
            
            # Forward pass (Returns dual outputs for deep supervision)
            final_pred, mid_pred = model(inputs)
            
            # Loss 1: Primary Advanced Loss (Physics + Laplacian)
            loss_final = criterion_main(final_pred, targets, thermal_ref)
            
            # Loss 2: Auxiliary supervision to stabilize middle layers
            loss_mid = criterion_aux(mid_pred, targets)
            
            # Weighted combination
            combined_loss = loss_final + (HYPERPARAMS["aux_weight"] * loss_mid)
            
            # Backprop
            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()
            
            total_epoch_loss += combined_loss.item()
            pbar.set_postfix({
                "total": f"{combined_loss.item():.4f}", 
                "final": f"{loss_final.item():.4f}"
            })

        # --- Statistics & Saving ---
        avg_loss = total_epoch_loss / len(loader)
        print(f"==> Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.6f}")
        
        if (epoch + 1) % 5 == 0:
            save_name = f"advanced_sft_ep{epoch+1}.pth"
            save_path = os.path.join(HYPERPARAMS["ckpt_dir"], save_name)
            torch.save(model.state_dict(), save_path)
            print(f"[!] Checkpoint: {save_name} saved.")

    # Final Export
    final_out = os.path.join(HYPERPARAMS["ckpt_dir"], "sft_final_model.pth")
    torch.save(model.state_dict(), final_out)
    print(f"\n[✔] Advanced SFT Training Finished. Model stored at {final_out}")

if __name__ == "__main__":
    try:
        train_phase_advanced()
    except KeyboardInterrupt:
        print("\n[!] User aborted training.")
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()