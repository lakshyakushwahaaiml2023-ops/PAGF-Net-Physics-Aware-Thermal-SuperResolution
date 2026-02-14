import torch
import torch.nn as nn
import torch.nn.functional as F

class LaplacianLoss(nn.Module):
    """
    Penalizes differences in the second-order derivatives.
    In thermal imaging, this helps preserve heat diffusion patterns.
    """
    def __init__(self):
        super().__init__()
        # Standard 3x3 Laplacian kernel (4-connectivity)
        kernel = torch.tensor([[0, 1, 0], 
                               [1, -4, 1], 
                               [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian_k', kernel)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        p_lap = F.conv2d(pred, self.laplacian_k, padding=1)
        t_lap = F.conv2d(target, self.laplacian_k, padding=1)
        return self.l1(p_lap, t_lap)

class GradientLoss(nn.Module):
    """
    Sobel-based edge loss to preserve structural boundaries.
    """
    def __init__(self):
        super().__init__()
        # Sobel-X
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        # Sobel-Y
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        px, py = F.conv2d(pred, self.kx, padding=1), F.conv2d(pred, self.ky, padding=1)
        tx, ty = F.conv2d(target, self.kx, padding=1), F.conv2d(target, self.ky, padding=1)
        
        pred_grad = torch.abs(px) + torch.abs(py)
        target_grad = torch.abs(tx) + torch.abs(ty)
        return self.l1(pred_grad, target_grad)

class PhysicsAwareLoss(nn.Module):
    """
    Hybrid Loss combining:
    1. Pixel Fidelity (L1)
    2. Structural Integrity (Gradient/Edge)
    3. Energy Conservation (Downsampling Constraint)
    4. Diffusion Consistency (Laplacian)
    """
    def __init__(self, weights=(1.0, 0.5, 0.5, 0.2)):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.gradient_loss = GradientLoss()
        self.laplacian_loss = LaplacianLoss()
        self.w = weights

    def forward(self, pred, target, input_lr_bicubic):
        # 1. Pixel Loss
        l_pix = self.l1(pred, target)
        
        # 2. Structural/Edge Loss
        l_edge = self.gradient_loss(pred, target)
        
        # 3. Energy Conservation (Physics I)
        # Ensuring the downsampled SR image matches the LR observation
        pred_down = F.avg_pool2d(pred, 4, 4)
        input_down = F.avg_pool2d(input_lr_bicubic, 4, 4) 
        l_energy = self.l1(pred_down, input_down)
        
        # 4. Heat Diffusion Consistency (Physics II)
        l_diff = self.laplacian_loss(pred, target)
        
        # Weighted Total
        return (self.w[0] * l_pix + 
                self.w[1] * l_edge + 
                self.w[2] * l_energy + 
                self.w[3] * l_diff)