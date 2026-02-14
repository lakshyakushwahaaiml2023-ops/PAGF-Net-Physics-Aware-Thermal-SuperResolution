import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeExtractionLoss(nn.Module):
    """
    Encourages the model to recover sharp boundaries by penalizing 
    discrepancies in the image gradients (Sobel).
    """
    def __init__(self):
        super().__init__()
        # Define Sobel filters for edge detection
        # Note: Humans often define these as list then convert to tensor for readability
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        
        # Reshape to (out_channels, in_channels, H, W)
        k_x = torch.FloatTensor(sobel_x).view(1, 1, 3, 3)
        k_y = torch.FloatTensor(sobel_y).view(1, 1, 3, 3)
        
        # register_buffer ensures these move to GPU with the model but aren't 'parameters'
        self.register_buffer('filter_x', k_x)
        self.register_buffer('filter_y', k_y)
        
        self.l1 = nn.L1Loss()

    def forward(self, pred, gt):
        # Extract horizontal and vertical edges
        grad_p_x = F.conv2d(pred, self.filter_x, padding=1)
        grad_p_y = F.conv2d(pred, self.filter_y, padding=1)
        
        grad_g_x = F.conv2d(gt, self.filter_x, padding=1)
        grad_g_y = F.conv2d(gt, self.filter_y, padding=1)
        
        # Approximate gradient magnitude (L1-norm style for faster convergence)
        mag_pred = torch.abs(grad_p_x) + torch.abs(grad_p_y)
        mag_gt = torch.abs(grad_g_x) + torch.abs(grad_g_y)
        
        return self.l1(mag_pred, mag_gt)

class HybridPhysicsLoss(nn.Module):
    def __init__(self, w_edge=0.5, w_energy=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.edge_loss = EdgeExtractionLoss()
        
        # Weights as parameters (allows for easier experimentation)
        self.w_edge = w_edge
        self.w_energy = w_energy

    def forward(self, pred, target, lr_ref):
        # 1. Pixel-wise reconstruction (Fidelity)
        loss_pix = self.l1(pred, target)
        
        # 2. Structural Edge loss (Sharpness)
        loss_edge = self.edge_loss(pred, target)
        
        # 3. Energy Conservation (Physics Constraint)
        # We ensure the downsampled SR image matches the original LR input
        pred_down = F.avg_pool2d(pred, kernel_size=4, stride=4)
        lr_down = F.avg_pool2d(lr_ref, kernel_size=4, stride=4)
        loss_energy = self.l1(pred_down, lr_down)
        
        # Combine with weighted importance
        total_loss = loss_pix + (self.w_edge * loss_edge) + (self.w_energy * loss_energy)
        
        return total_loss