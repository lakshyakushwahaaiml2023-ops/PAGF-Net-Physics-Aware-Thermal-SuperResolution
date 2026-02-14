import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
from tqdm import tqdm  # For progress bars

# IMPORTS (Make sure your files are named correctly)
from dataLoader import SatelliteThermalDataset  # Your data loader file
from model import OpticalGuidedSR               # Your model file

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16       # Lower this if you run out of GPU memory
LEARNING_RATE = 1e-4
EPOCHS = 20           # For Hackathon demo. Increase if you have time.
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- PHYSICS LOSS FUNCTIONS ---
class PhysicsLoss(nn.Module):
    def __init__(self):
        super(PhysicsLoss, self).__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, pred, target, input_lr):
        # 1. Reconstruction Loss (Standard Accuracy)
        # Matches the high-res ground truth
        loss_rec = self.l1(pred, target)
        
        # 2. Energy Conservation Loss (The "Physics" Part)
        # Downsample the prediction to match the LR input scale (4x down)
        # We use AvgPool to simulate "Area Integration" of sensor energy
        pred_down = torch.nn.functional.avg_pool2d(pred, kernel_size=4, stride=4)
        input_down = torch.nn.functional.avg_pool2d(input_lr, kernel_size=4, stride=4)
        
        loss_energy = self.l1(pred_down, input_down)
        
        # 3. Total Loss
        # We weigh reconstruction higher (1.0) and energy as a constraint (0.5)
        return loss_rec + 0.5 * loss_energy

# --- TRAINING FUNCTION ---
def train():
    print(f"Training on {DEVICE}...")
    
    # 1. Data
    # UPDATE PATH HERE
    dataset_path = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"
    dataset = SatelliteThermalDataset(dataset_path, max_samples=3000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for Windows safety
    
    # 2. Model
    model = OpticalGuidedSR().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = PhysicsLoss()
    
    # 3. Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        # Progress Bar
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Forward
            # Input channel 0 is the "Blurry Thermal" (LR Input)
            lr_input = inputs[:, 0:1, :, :] 
            
            predictions = model(inputs)
            
            # Loss Calculation
            # We pass lr_input to enforce Energy Conservation
            loss = criterion(predictions, targets, lr_input)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update Progress Bar
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())
        
        # Save Checkpoint
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.6f}")
        
        # Save every 5 epochs or if it's the best one
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    # Final Save
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
    print("🏆 Training Complete!")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()