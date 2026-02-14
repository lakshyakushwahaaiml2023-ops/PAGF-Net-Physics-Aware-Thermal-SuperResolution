import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

# --- IMPORTS ---
from model_v2 import PAGFNetV2
from loss_v2 import PhysicsAwareLoss
from dataLoader import SatelliteThermalDataset

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"
SAVE_DIR = "checkpoints_final"

# OVERNIGHT HYPERPARAMETERS
EPOCHS = 50           # Deep training
BATCH_SIZE = 8        # Keep small for stability
START_LR = 2e-4       # Start slightly higher
MIN_LR = 1e-6         # Floor for decay
WEIGHT_DECAY = 1e-5   # Regularization to prevent overfitting

def train_overnight():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f"INITIALIZING OVERNIGHT TRAINING SESSION")
    print(f"   • Device:      {DEVICE}")
    print(f"   • Epochs:      {EPOCHS}")
    print(f"   • Model:       PAGF-Net v2 (Physics-Aware)")
    print(f"   • Saving to:   {SAVE_DIR}/")
    print("-" * 60)

    # 1. DATA LOADER
    train_dataset = SatelliteThermalDataset(DATASET_PATH, max_samples=4500)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 2. MODEL & OPTIMIZER
    model = PAGFNetV2().to(DEVICE)
    criterion = PhysicsAwareLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=START_LR, weight_decay=WEIGHT_DECAY)
    
    # 3. SCHEDULER (FIXED: Removed verbose=True)
    # Reduces LR if loss stops improving for 3 epochs.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Tracking
    best_loss = float('inf')
    start_time = time.time()

    # 4. TRAINING LOOP
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Extract LR approximation for Energy Loss
            bicubic_in = inputs[:, 0:1, :, :]
            
            # Forward Pass (Returns Final, Mid)
            out_final, out_mid = model(inputs)
            
            # Loss Calculation (Deep Supervision)
            loss_final = criterion(out_final, targets, bicubic_in)
            loss_mid = criterion(out_mid, targets, bicubic_in)
            
            # Weighted Sum
            total_loss = loss_final + (0.5 * loss_mid)
            
            # Backward Pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())
        
        # End of Epoch Stats
        avg_loss = epoch_loss / len(train_loader)
        
        # Step the Scheduler
        scheduler.step(avg_loss)
        
        # Manual LR Print (Replaces verbose=True)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"   Done. Avg Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

        # 5. SMART SAVING
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model_v2.pth")
            print(f"   NEW BEST MODEL SAVED! (Loss: {best_loss:.6f})")
        
        torch.save(model.state_dict(), f"{SAVE_DIR}/latest_checkpoint.pth")

    # Final wrap up
    total_time = (time.time() - start_time) / 3600
    print("-" * 60)
    print(f"TRAINING COMPLETE in {total_time:.2f} hours.")
    print(f"   Best Loss achieved: {best_loss:.6f}")
    print(f"   Model saved at: {SAVE_DIR}/best_model_v2.pth")

if __name__ == "__main__":
    try:
        train_overnight()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saved latest checkpoint.")
    except Exception as e:
        print(f"\nError occurred: {e}")