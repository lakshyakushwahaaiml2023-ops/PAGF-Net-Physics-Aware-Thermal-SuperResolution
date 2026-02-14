import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

# Local Project Imports
from model_v2 import PAGFNetV2
from loss_v2 import PhysicsAwareLoss
from dataLoader import SatelliteThermalDataset

# --- SETTINGS ---
RUN_CONFIG = {
    "lr": 1e-4,
    "batch_size": 8,
    "epochs": 20,
    "aux_weight": 0.5,
    "val_interval": 2,      # Run validation every 2 epochs
    "ckpt_dir": "checkpoints/pagf_v2",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "data_path": r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"
}

os.makedirs(RUN_CONFIG["ckpt_dir"], exist_ok=True)

def validate(model, loader):
    """Computes average PSNR on a small validation subset."""
    model.eval()
    psnr_values = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(RUN_CONFIG["device"]), targets.to(RUN_CONFIG["device"])
            output, _ = model(inputs)
            
            # Convert to numpy and clip for metric calculation
            out_np = output.detach().cpu().numpy().clip(0, 1)
            gt_np = targets.detach().cpu().numpy().clip(0, 1)
            
            for b in range(out_np.shape[0]):
                psnr_values.append(compute_psnr(gt_np[b, 0], out_np[b, 0], data_range=1.0))
    
    return np.mean(psnr_values)

def train_v2():
    print(f"[*] Starting PAGF-Net v2 Pipeline on {RUN_CONFIG['device']}")

    # 1. Dataset & Split (Training + Small Validation set)
    full_dataset = SatelliteThermalDataset(RUN_CONFIG["data_path"], max_samples=4200)
    train_ds = Subset(full_dataset, range(0, 4000))
    val_ds   = Subset(full_dataset, range(4000, 4200)) # 200 samples for val
    
    train_loader = DataLoader(train_ds, batch_size=RUN_CONFIG["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=RUN_CONFIG["batch_size"], shuffle=False)

    # 2. Initialization
    model = PAGFNetV2().to(RUN_CONFIG["device"])
    optimizer = optim.Adam(model.parameters(), lr=RUN_CONFIG["lr"], betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    criterion = PhysicsAwareLoss().to(RUN_CONFIG["device"])

    best_psnr = 0.0

    # 3. Training Loop
    for epoch in range(RUN_CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{RUN_CONFIG['epochs']}")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(RUN_CONFIG["device"]), targets.to(RUN_CONFIG["device"])
            bicubic_ref = inputs[:, 0:1, :, :] # Physics anchor
            
            # Forward & Deep Supervision Loss
            out_final, out_mid = model(inputs)
            loss_final = criterion(out_final, targets, bicubic_ref)
            loss_aux   = criterion(out_mid, targets, bicubic_ref)
            
            total_loss = loss_final + (RUN_CONFIG["aux_weight"] * loss_aux)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        scheduler.step()

        # 4. Validation & Checkpointing
        if (epoch + 1) % RUN_CONFIG["val_interval"] == 0:
            avg_psnr = validate(model, val_loader)
            print(f"Epoch {epoch+1} Val PSNR: {avg_psnr:.2f} dB")
            
            # Save "Best" model based on PSNR
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(model.state_dict(), os.path.join(RUN_CONFIG["ckpt_dir"], "best_model.pth"))
                print(f"New Best PSNR achieved. Model saved.")

    # Final Save
    torch.save(model.state_dict(), os.path.join(RUN_CONFIG["ckpt_dir"], "final_pagf_v2.pth"))
    print(f"\nTraining Complete. Best Val PSNR: {best_psnr:.2f} dB")

if __name__ == "__main__":
    train_v2()