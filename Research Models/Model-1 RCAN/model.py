import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Standard SE-block for recalibrating channel weights.
    Identical logic to original, just cleaner grouping.
    """
    def __init__(self, in_dims, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Bottleneck architecture for attention mask
        self.mlp = nn.Sequential(
            nn.Conv2d(in_dims, in_dims // reduction, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_dims // reduction, in_dims, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.mlp(self.avg_pool(x))

class ResidualBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        # Grouped the core operations into a 'body' sequence
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            ChannelAttention(n_feats)
        )

    def forward(self, x):
        # Local skip connection
        return x + self.body(x)

class OpticalGuidedSR(nn.Module):
    def __init__(self, base_c=64, n_res=6):
        super().__init__()

        # Input processing for Thermal and Optical streams
        self.t_head = nn.Conv2d(1, base_c, 3, padding=1)
        self.o_head = nn.Conv2d(1, base_c, 3, padding=1)

        # Merge the two 64-channel features back to 64
        self.fusion = nn.Conv2d(base_c * 2, base_c, 3, padding=1)

        # Main feature extraction trunk
        self.res_trunk = nn.Sequential(
            *[ResidualBlock(base_c) for _ in range(n_res)]
        )

        # Final map to generate the residual image
        self.reconstruct = nn.Conv2d(base_c, 1, 3, padding=1)

    def forward(self, x):
        # Slice the input tensor [B, 2, H, W] into Thermal and Optical
        # Logic remains identical to your original slice operations
        t_img = x[:, 0:1, :, :]
        o_img = x[:, 1:2, :, :]

        # Feature extraction
        t_feat = self.t_head(t_img)
        o_feat = self.o_head(o_img)

        # Concatenate and apply fusion + activation
        fused = torch.cat([t_feat, o_feat], dim=1)
        fused = F.relu(self.fusion(fused))

        # Deep refinement
        refined = self.res_trunk(fused)

        # Predict the high-frequency residual
        res = self.reconstruct(refined)

        # Global residual learning: Add predicted detail back to thermal input
        return t_img + res