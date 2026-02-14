import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. NEW: HEAT DIFFUSION LOSS (Laplacian) ---
class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()
        # The Laplacian Kernel (Finite Difference Approximation for ∇²)
        # This measures the "smoothness" or "flow" of heat at every pixel.
        k = torch.tensor([[0,  1, 0], 
                          [1, -4, 1], 
                          [0,  1, 0]]).float().view(1, 1, 3, 3)
        
        self.register_buffer('laplacian_kernel', k)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # Convolve both images with the Laplacian kernel
        # We use padding=1 to keep the size the same
        pred_lap = F.conv2d(pred, self.laplacian_kernel, padding=1)
        target_lap = F.conv2d(target, self.laplacian_kernel, padding=1)
        
        # Minimize the difference in their "Heat Flow" properties
        return self.l1(pred_lap, target_lap)

# --- 2. EXISTING: EDGE LOSS ---
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
        k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3)
        self.register_buffer('k_x', k_x)
        self.register_buffer('k_y', k_y)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred_x = F.conv2d(pred, self.k_x, padding=1)
        pred_y = F.conv2d(pred, self.k_y, padding=1)
        target_x = F.conv2d(target, self.k_x, padding=1)
        target_y = F.conv2d(target, self.k_y, padding=1)
        return self.l1(torch.abs(pred_x)+torch.abs(pred_y), torch.abs(target_x)+torch.abs(target_y))

# --- 3. UPDATED: TOTAL PHYSICS LOSS ---
class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.edge_loss = EdgeLoss()
        self.laplacian_loss = LaplacianLoss() # <--- NEW

    def forward(self, pred, target, input_lr):
        # A. Radiometric Fidelity (Pixel Values)
        l_pix = self.l1(pred, target)
        
        # B. Geometric Structural Consistency (Edges)
        l_edge = self.edge_loss(pred, target)
        
        # C. Thermodynamic Conservation (Energy)
        pred_down = F.avg_pool2d(pred, 4, 4)
        input_down = F.avg_pool2d(input_lr, 4, 4)
        l_energy = self.l1(pred_down, input_down)
        
        # D. Thermodynamic Consistency (Heat Diffusion) <--- NEW
        l_diff = self.laplacian_loss(pred, target)
        
        # WEIGHTS:
        # Pixel: 1.0 (Base)
        # Edge: 0.5 (Structure)
        # Energy: 0.5 (Conservation)
        # Diffusion: 0.2 (Smoothness - keep small to avoid over-blurring)
        return l_pix + (0.5 * l_edge) + (0.5 * l_energy) + (0.2 * l_diff)