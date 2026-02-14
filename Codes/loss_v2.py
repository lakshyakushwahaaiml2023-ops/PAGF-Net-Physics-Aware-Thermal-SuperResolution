import torch
import torch.nn as nn
import torch.nn.functional as F

class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).float().view(1, 1, 3, 3)
        self.register_buffer('k', k)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.l1(F.conv2d(pred, self.k, padding=1), F.conv2d(target, self.k, padding=1))

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
        self.register_buffer('k_x', k)
        self.register_buffer('k_y', k.transpose(2, 3))
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        px = F.conv2d(pred, self.k_x, padding=1)
        py = F.conv2d(pred, self.k_y, padding=1)
        tx = F.conv2d(target, self.k_x, padding=1)
        ty = F.conv2d(target, self.k_y, padding=1)
        return self.l1(torch.abs(px)+torch.abs(py), torch.abs(tx)+torch.abs(ty))

class PhysicsAwareLoss(nn.Module):
    def __init__(self):
        super(PhysicsAwareLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.edge = EdgeLoss()
        self.laplacian = LaplacianLoss()

    def forward(self, pred, target, input_lr_bicubic):
        # 1. Fidelity
        loss_pix = self.l1(pred, target)
        
        # 2. Structure (Edges)
        loss_edge = self.edge(pred, target)
        
        # 3. Physics I: Energy Conservation
        # Downsample prediction and compare to original low-res input
        # Note: We approximate 'input_lr' by downsampling the bicubic input
        pred_down = F.avg_pool2d(pred, 4, 4)
        input_down = F.avg_pool2d(input_lr_bicubic, 4, 4)
        loss_energy = self.l1(pred_down, input_down)
        
        # 4. Physics II: Heat Diffusion (Laplacian)
        loss_diff = self.laplacian(pred, target)
        
        return loss_pix + (0.5 * loss_edge) + (0.5 * loss_energy) + (0.2 * loss_diff)