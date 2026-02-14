import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# NEW IMPORTS
from dataLoader import SatelliteThermalDataset
from model_advanced import AdvancedOpticalSR  # <--- NEW MODEL
from loss_advanced_laplacian import TotalLoss           # <--- NEW LOSS

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # SFT is heavier, might need smaller batch
EPOCHS = 15     # Advanced models converge faster but need time
LR = 1e-4

def train_advanced():
    print(f"Launching ADVANCED Training on {DEVICE}...")
    
    # Data (Uses your new degradation loader)
    dataset_path = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"
    dataset = SatelliteThermalDataset(dataset_path, max_samples=4000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model & Loss
    model = AdvancedOpticalSR().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = TotalLoss().to(DEVICE)
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        loop = tqdm(dataloader, leave=True)
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Extract LR input for Physics Loss (Channel 0)
            # Note: Your new loader returns (Bicubic_Input, Optical)
            # We need the true LR for energy loss. Since we don't return it directly,
            # we can approximate it by downsampling the Bicubic Input again.
            bicubic_in = inputs[:, 0:1, :, :]
            lr_approx = torch.nn.functional.avg_pool2d(bicubic_in, 4, 4)
            
            # Forward (Multi-Scale!)
            out_final, out_mid = model(inputs)
            
            # Loss Calculation (Deep Supervision)
            # Loss on Final Output
            loss_final = criterion(out_final, targets, bicubic_in) # Pass bicubic as 'input_lr' reference
            
            # Loss on Intermediate Output (Auxiliary Loss)
            # We don't apply energy loss to mid, just pixel loss for stability
            loss_mid = torch.nn.L1Loss()(out_mid, targets)
            
            # Total Loss = Final + 0.5 * Mid
            total_loss = loss_final + (0.5 * loss_mid)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())

        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.5f}")
        
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/advanced_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "checkpoints/final_advanced_model.pth")
    print("Advanced Training Complete!")

if __name__ == "__main__":
    train_advanced()