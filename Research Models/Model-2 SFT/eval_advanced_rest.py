import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

# Local research modules
from model_advanced import AdvancedOpticalSR 
from dataLoader import SatelliteThermalDataset

# --- SETTINGS ---
RUN_CONFIG = {
    "ckpt": "checkpoints/final_advanced_model.pth",
    "data_path": r"D:\Lakshya\Symposium\Dataset\ssl4eo_l_oli_tirs_toa_benchmark",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "offset": 4500,        # Skip samples used for training
    "test_n": 1000,        # Number of samples for a robust evaluation
    "batch_size": 8,
    "out_dir": "eval_results_v2"
}

os.makedirs(RUN_CONFIG["out_dir"], exist_ok=True)

# --- REFINEMENT TOOLS ---

def physics_back_projection(sr, lr_ref, steps=5, step_size=0.05):
    """
    Iterative Back-Projection (IBP) to ensure the SR result 
    is physically consistent with the original LR observation.
    """
    refined = sr.clone().detach()
    _, _, h, w = refined.shape
    
    for _ in range(steps):
        # Downsample the current estimate back to LR space
        # Using Area-based pooling to simulate sensor integration
        downsampled = torch.nn.functional.avg_pool2d(refined, 4, 4)
        
        # Calculate residuals in LR space
        # If lr_ref is already upsampled, we downsample it for comparison
        target_lr = lr_ref if lr_ref.shape == downsampled.shape else \
                    torch.nn.functional.avg_pool2d(lr_ref, 4, 4)
        
        error = target_lr - downsampled
        
        # Project the error back to HR space
        upsampled_error = torch.nn.functional.interpolate(
            error, size=(h, w), mode='bicubic', align_corners=False
        )
        
        # Update the estimate
        refined += (step_size * upsampled_error)
        
    return refined

def ensemble_inference(model, x):
    """
    8-Point Geometric Ensemble (Test-Time Augmentation).
    Averages predictions across flips and rotations for better stability.
    """
    # Define the 8 dihedral group transforms
    # (identity, h-flip, v-flip, rotations, etc.)
    transforms = [
        (lambda t: t, lambda t: t),
        (lambda t: t.flip(3), lambda t: t.flip(3)),
        (lambda t: t.flip(2), lambda t: t.flip(2)),
        (lambda t: t.rot90(1, [2, 3]), lambda t: t.rot90(3, [2, 3])),
        (lambda t: t.rot90(2, [2, 3]), lambda t: t.rot90(2, [2, 3])),
        (lambda t: t.rot90(3, [2, 3]), lambda t: t.rot90(1, [2, 3])),
        (lambda t: t.flip(3).rot90(1, [2, 3]), lambda t: t.rot90(3, [2, 3]).flip(3)),
        (lambda t: t.flip(2).rot90(1, [2, 3]), lambda t: t.rot90(3, [2, 3]).flip(2))
    ]
    
    acc_preds = []
    
    for aug, de_aug in transforms:
        with torch.no_grad():
            # Get the final SR output (index 0 of the returned tuple)
            out_sr, _ = model(aug(x))
            acc_preds.append(de_aug(out_sr))
            
    return torch.stack(acc_preds).mean(dim=0)

# --- EVALUATION LOGIC ---

def run_phase2_eval():
    print(f"[*] Initializing Phase 2 Evaluation (SFT-Model + TTA + IBP)")
    
    # 1. Model Initialization
    net = AdvancedOpticalSR().to(RUN_CONFIG["device"])
    if not os.path.exists(RUN_CONFIG["ckpt"]):
        print(f"[!] Error: Checkpoint missing at {RUN_CONFIG['ckpt']}")
        return
    
    net.load_state_dict(torch.load(RUN_CONFIG["ckpt"], map_location=RUN_CONFIG["device"]))
    net.eval()

    # 2. Data Slicing (Unseen Random Subset)
    full_ds = SatelliteThermalDataset(RUN_CONFIG["data_path"], max_samples=None)
    
    # Ensure we aren't overlapping with training data
    valid_indices = list(range(RUN_CONFIG["offset"], len(full_ds)))
    if len(valid_indices) < RUN_CONFIG["test_n"]:
        print("[!] Warning: Dataset too small for requested test size.")
        selected_indices = valid_indices
    else:
        selected_indices = random.sample(valid_indices, RUN_CONFIG["test_n"])
    
    loader = DataLoader(Subset(full_ds, selected_indices), 
                        batch_size=RUN_CONFIG["batch_size"], 
                        shuffle=False)

    # 3. Execution
    results = []
    pbar = tqdm(loader, desc="Running Phase 2 Inference")

    for i, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(RUN_CONFIG["device"]), targets.to(RUN_CONFIG["device"])
        
        # Step A: Geometric Ensemble (TTA)
        sr_ensemble = ensemble_inference(net, inputs)
        
        # Step B: Physics Refinement (Back-Projection)
        # Using the LR Thermal channel as the physical anchor
        final_sr = physics_back_projection(sr_ensemble, inputs[:, 0:1, :, :])
        
        # Metric calculation
        gt_np = targets.cpu().numpy().clip(0, 1)
        sr_np = final_sr.cpu().numpy().clip(0, 1)
        
        for b in range(gt_np.shape[0]):
            score = compute_psnr(gt_np[b, 0], sr_np[b, 0], data_range=1.0)
            results.append(score)
            
        pbar.set_postfix(psnr=f"{np.mean(results):.2f}")

        # Save visualization for the first batch
        if i == 0:
            generate_report_visual(inputs, final_sr, targets)

    # 4. Final Report
    print("\n" + "#"*40)
    print(f" FINAL EVALUATION REPORT")
    print(f" Mean PSNR: {np.mean(results):.4f} dB")
    print(f" Std Dev:   {np.std(results):.4f}")
    print("#"*40)

def generate_report_visual(inputs, preds, targets):
    """Produces a side-by-side comparison for visual confirmation."""
    inputs, preds, targets = inputs.cpu().numpy(), preds.cpu().numpy(), targets.cpu().numpy()
    
    n_rows = min(4, inputs.shape[0])
    fig, axes = plt.subplots(n_rows, 3, figsize=(10, 12))
    
    for r in range(n_rows):
        axes[r, 0].imshow(inputs[r, 0], cmap='inferno')
        axes[r, 0].set_title("LR Input", fontsize=9); axes[r, 0].axis('off')
        
        axes[r, 1].imshow(preds[r, 0], cmap='inferno')
        axes[r, 1].set_title("Refined SFT-SR", fontsize=9); axes[r, 1].axis('off')
        
        axes[r, 2].imshow(targets[r, 0], cmap='inferno')
        axes[r, 2].set_title("Ground Truth", fontsize=9); axes[r, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RUN_CONFIG["out_dir"], "qualitative_check.png"), dpi=300)
    print(f"[*] Visualization saved to {RUN_CONFIG['out_dir']}/qualitative_check.png")

if __name__ == "__main__":
    run_phase2_eval()