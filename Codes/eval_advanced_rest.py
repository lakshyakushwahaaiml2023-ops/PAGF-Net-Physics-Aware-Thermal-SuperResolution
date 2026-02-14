import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# --- IMPORTS ---
# CHANGED: Importing the Advanced SFT Model
from model_advanced import AdvancedOpticalSR 
from dataLoader import SatelliteThermalDataset

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CHANGED: Point to your Phase 2 checkpoint
MODEL_PATH = "checkpoints/final_advanced_model.pth" 
DATASET_PATH = r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark"

# EVAL SETTINGS
TRAINED_ON = 4500           # Training set size (to skip)
NUM_TEST_SAMPLES = 1000     # Number of random images to test
BATCH_SIZE = 8              # Batch size

# OUTPUT
SAVE_VISUALS = True
RESULTS_DIR = "results_phase2_sft"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 1. VECTORIZED BACK-PROJECTION (Physics Refiner) ---
def back_projection(sr_tensor, lr_target_tensor, iterations=5, lr_rate=0.05):
    refined_sr = sr_tensor.clone().detach()
    target_h, target_w = refined_sr.shape[2], refined_sr.shape[3]
    
    for _ in range(iterations):
        # 1. Simulate the sensor (Downsample the current SR guess)
        sr_down = F.avg_pool2d(refined_sr, kernel_size=4, stride=4)
        
        # 2. Calculate the "Physics Error"
        if sr_down.shape != lr_target_tensor.shape:
            # Handle if lr_target is already upsampled
            lr_small = F.avg_pool2d(lr_target_tensor, kernel_size=4, stride=4) 
            diff = lr_small - sr_down
        else:
            diff = lr_target_tensor - sr_down

        # 3. Upsample the error to match SR size
        diff_sr = F.interpolate(diff, size=(target_h, target_w), mode='bicubic', align_corners=False)
        
        # 4. Correct the image
        refined_sr = refined_sr + (diff_sr * lr_rate)
        
    return refined_sr

# --- 2. 8-POINT ENSEMBLE (TTA) ---
def predict_tta_batch(model, inputs):
    model.eval()
    preds = []
    
    # 8-Point Dihedral Transformations
    transforms = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.rot90(x, 1, [2, 3]),
        lambda x: torch.rot90(x, 2, [2, 3]),
        lambda x: torch.rot90(x, 3, [2, 3]),
        lambda x: torch.rot90(torch.flip(x, [3]), 1, [2,3]),
        lambda x: torch.rot90(torch.flip(x, [2]), 1, [2,3]),
    ]
    
    inv_transforms = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.rot90(x, 3, [2, 3]),
        lambda x: torch.rot90(x, 2, [2, 3]),
        lambda x: torch.rot90(x, 1, [2, 3]),
        lambda x: torch.flip(torch.rot90(x, 3, [2,3]), [3]), 
        lambda x: torch.flip(torch.rot90(x, 3, [2,3]), [2]),
    ]
    
    for t, inv_t in zip(transforms, inv_transforms):
        aug_input = t(inputs)
        with torch.no_grad():
            # CHANGED: Handle the tuple return (final, mid)
            aug_pred_final, _ = model(aug_input) 
        
        preds.append(inv_t(aug_pred_final))
    
    return torch.mean(torch.stack(preds), dim=0)

# --- 3. MAIN EXECUTION ---
def run_evaluation():
    print(f"LAUNCHING PHASE 2 EVALUATION (SFT Model)")
    print(f"   • Model Path:  {MODEL_PATH}")
    print(f"   • Test Size:   {NUM_TEST_SAMPLES} Random Samples")
    print("-" * 60)
    
    # 1. Load Advanced Model
    model = AdvancedOpticalSR().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Phase 2 Model Loaded Successfully")
    else:
        print(f"Error: Checkpoint not found at {MODEL_PATH}")
        return

    # 2. Dataset Slicing
    full_dataset = SatelliteThermalDataset(DATASET_PATH, max_samples=None)
    total_avail = len(full_dataset)
    
    if total_avail < TRAINED_ON:
        print("Dataset too small.")
        return

    unseen_indices = list(range(TRAINED_ON, total_avail))
    
    if len(unseen_indices) > NUM_TEST_SAMPLES:
        test_indices = random.sample(unseen_indices, NUM_TEST_SAMPLES)
    else:
        test_indices = unseen_indices
        
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"   • Testing on indices: {test_indices[:5]} ... (Total {len(test_dataset)})")

    # 3. Metrics Loop
    psnr_scores = []
    
    loop = tqdm(test_loader, desc="Evaluating Phase 2", unit="batch")
    
    for batch_idx, (inputs, targets) in enumerate(loop):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        with torch.no_grad():
            # A. Geometric Ensemble
            tta_pred = predict_tta_batch(model, inputs)
            
            # B. Physics Back-Projection
            final_pred = back_projection(tta_pred, inputs[:, 0:1, :, :], iterations=5)
            
        # Metrics
        gt_np = np.clip(targets.cpu().numpy(), 0, 1)
        pred_np = np.clip(final_pred.cpu().numpy(), 0, 1)
        
        for i in range(gt_np.shape[0]):
            val = psnr(gt_np[i,0], pred_np[i,0], data_range=1.0)
            psnr_scores.append(val)
        
        loop.set_postfix(avg_psnr=f"{np.mean(psnr_scores):.2f}")
        
        # Save Visual
        if SAVE_VISUALS and batch_idx == 0:
            save_visual_batch(inputs, final_pred, targets)

    # 4. Final Stats
    avg = np.mean(psnr_scores)
    std = np.std(psnr_scores)
    
    print("\n" + "="*60)
    print(f"FINAL PHASE 2 RESULTS (Advanced SFT)")
    print("-" * 60)
    print(f"   • Average PSNR: {avg:.4f} dB")
    print(f"   • Consistency:  ±{std:.4f}")
    print("="*60)

def save_visual_batch(inputs, preds, targets):
    inputs = inputs.cpu().numpy()
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    for i in range(min(4, inputs.shape[0])):
        plt.subplot(4, 3, i*3 + 1)
        plt.imshow(inputs[i, 0], cmap='inferno')
        plt.title("Input (LR)", fontsize=8); plt.axis('off')
        
        plt.subplot(4, 3, i*3 + 2)
        plt.imshow(preds[i, 0], cmap='inferno')
        plt.title("Phase 2 SFT Output", fontsize=8); plt.axis('off')
        
        plt.subplot(4, 3, i*3 + 3)
        plt.imshow(targets[i, 0], cmap='inferno')
        plt.title("Ground Truth", fontsize=8); plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/visual_phase2.png", dpi=300)
    print(f"\nVisual proof saved to {RESULTS_DIR}/visual_phase2.png")

if __name__ == "__main__":
    run_evaluation()