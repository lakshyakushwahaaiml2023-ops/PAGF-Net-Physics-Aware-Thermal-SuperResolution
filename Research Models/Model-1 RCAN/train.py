import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local module imports
from dataLoader import SatelliteThermalDataset
from model import OpticalGuidedSR

# --- HYPERPARAMETERS & SETUP ---
CONFIG = {
    "lr": 1e-4,
    "batch_size": 16,
    "epochs": 20,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "ckpt_dir": "checkpoints",
    "data_path": r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"
}

os.makedirs(CONFIG["ckpt_dir"], exist_ok=True)

class PhysicsInformedLoss(nn.Module):
    """
    Combines standard pixel-wise reconstruction with an 
    energy conservation constraint to maintain physical validity.
    """
    def __init__(self, energy_weight=0.5):
        super().__init__()
        self.criterion = nn.L1Loss()
        self.w_energy = energy_weight
        
    def forward(self, pred, target, lr_thermal):
        # Primary reconstruction accuracy
        loss_pixel = self.criterion(pred, target)
        
        # Energy conservation: downsampled SR should match LR input
        # Simulates the sensor's integration over the pixel area
        pred_blurred = torch.nn.functional.avg_pool2d(pred, kernel_size=4, stride=4)
        lr_blurred = torch.nn.functional.avg_pool2d(lr_thermal, kernel_size=4, stride=4)
        
        loss_energy = self.criterion(pred_blurred, lr_blurred)
        
        return loss_pixel + (self.w_energy * loss_energy)

def run_trainer():
    print(f"[*] Initializing trainer on device: {CONFIG['device']}")

    # 1. Dataset & Loader
    # Using num_workers=0 to avoid multiprocessing overhead/errors on Windows
    train_ds = SatelliteThermalDataset(CONFIG["data_path"], max_samples=3000)
    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=0
    )

    # 2. Model, Opts, and Scheduler
    model = OpticalGuidedSR().to(CONFIG["device"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], betas=(0.9, 0.999))
    
    # Adding a scheduler - typical for getting better convergence
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    criterion = PhysicsInformedLoss()

    # 3. Training Loop
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(CONFIG["device"]), targets.to(CONFIG["device"])
            
            # Slice out the thermal channel (Channel 0) for the physics constraint
            lr_thermal = inputs[:, 0:1, :, :]
            
            # Optimization step
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, lr_thermal)
            
            loss.backward()
            optimizer.step()
            
            # Track and display
            running_loss.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        # Epoch cleanup
        scheduler.step()
        avg_epoch_loss = sum(running_loss) / len(running_loss)
        print(f"-> Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.6f}")

        # Periodic Save
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(CONFIG["ckpt_dir"], f"model_ep{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[!] Checkpoint saved: {ckpt_path}")

    # Final Export
    final_path = os.path.join(CONFIG["ckpt_dir"], "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n[STARE] Training complete. Model saved to {final_path}")

if __name__ == "__main__":
    try:
        run_trainer()
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")
    except Exception as e:
        print(f"\n[X] Fatal Error during training: {e}")
        import traceback
        traceback.print_exc()