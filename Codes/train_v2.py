import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# IMPORTS
from model_v2 import PAGFNetV2
from loss_v2 import PhysicsAwareLoss
from dataLoader import SatelliteThermalDataset # Ensure this uses the 'Blur+Noise' version

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20  # More epochs for the advanced model
BATCH_SIZE = 8
LR = 1e-4

def train_v2():
    print(f"Initializing PAGF-Net v2 on {DEVICE}...")
    
    # Dataset
    data_path = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"
    dataset = SatelliteThermalDataset(data_path, max_samples=4000)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model & Loss
    model = PAGFNetV2().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = PhysicsAwareLoss().to(DEVICE)
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        loop = tqdm(loader, leave=True)
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            bicubic_in = inputs[:, 0:1, :, :] # Extract for Energy Loss
            
            # Forward (Returns Final and Mid)
            out_final, out_mid = model(inputs)
            
            # Loss Calculation (Deep Supervision)
            loss_final = criterion(out_final, targets, bicubic_in)
            loss_mid = criterion(out_mid, targets, bicubic_in) # Aux loss
            
            total_loss = loss_final + (0.5 * loss_mid)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())
            
        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {epoch_loss/len(loader):.5f}")
        
        # Save checkpoints
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/pagf_v2_epoch_{epoch+1}.pth")

    # Final Save
    torch.save(model.state_dict(), "checkpoints/final_pagf_v2.pth")
    print("Training Complete: PAGF-Net v2 is ready.")

if __name__ == "__main__":
    train_v2()