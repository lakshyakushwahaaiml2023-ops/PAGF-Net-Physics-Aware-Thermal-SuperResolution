import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # Sobel Kernels
        k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
        k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3)
        
        self.register_buffer('k_x', k_x)
        self.register_buffer('k_y', k_y)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # Calculate gradients
        pred_x = F.conv2d(pred, self.k_x, padding=1)
        pred_y = F.conv2d(pred, self.k_y, padding=1)
        
        target_x = F.conv2d(target, self.k_x, padding=1)
        target_y = F.conv2d(target, self.k_y, padding=1)
        
        # Calculate gradient magnitude
        pred_grad = torch.abs(pred_x) + torch.abs(pred_y)
        target_grad = torch.abs(target_x) + torch.abs(target_y)
        
        # Loss is the difference in edge strength
        return self.l1(pred_grad, target_grad)

class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.edge_loss = EdgeLoss()

    def forward(self, pred, target, input_lr):
        # 1. Pixel Loss (L1) - Fidelity
        l_pix = self.l1(pred, target)
        
        # 2. Edge Loss - Sharpness
        l_edge = self.edge_loss(pred, target)
        
        # 3. Energy Loss - Physics
        pred_down = F.avg_pool2d(pred, 4, 4)
        input_down = F.avg_pool2d(input_lr, 4, 4) # Assuming 4x downsample
        l_energy = self.l1(pred_down, input_down)
        
        # Weighted Sum
        # Pixel is base (1.0). Edge is critical (0.5). Energy is constraint (0.5)
        return l_pix + (0.5 * l_edge) + (0.5 * l_energy)