import torch
import torch.nn as nn
import torch.nn.functional as F

class SFTLayer(nn.Module):
    """
    Spatial Feature Transform: Adjusts thermal features based on optical priors.
    Learns spatial-wise affine transformation (scale & shift).
    """
    def __init__(self, n_feats):
        super().__init__()
        self.cond_map = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats, n_feats * 2, 1) 
        )

    def forward(self, x, cond):
        # Generate affine parameters: gamma (scale) and beta (shift)
        params = self.cond_map(cond)
        gamma, beta = torch.chunk(params, 2, dim=1)
        
        # Apply the transform: (1 + gamma) provides a stable identity start
        return x * (gamma + 1) + beta

class SFTResBlock(nn.Module):
    """Residual block integrated with an SFT layer for guided feature modulation."""
    def __init__(self, n_feats):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.sft = SFTLayer(n_feats)
        self.relu = nn.ReLU(True)

    def forward(self, x, cond):
        res = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.sft(out, cond)
        return res + out

class AdvancedOpticalSR(nn.Module):
    def __init__(self, n_feats=64, n_blocks=8):
        super().__init__()

        # --- Front End ---
        self.thermal_head = nn.Conv2d(1, n_feats, 3, padding=1)
        self.optical_head = nn.Conv2d(1, n_feats, 3, padding=1)

        # Condition generator (latent space for optical prior)
        self.conditioner = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats, n_feats, 1)
        )

        # --- Feature Trunk ---
        # Split into two stages to allow for intermediate supervision/auxiliary loss
        m_point = n_blocks // 2
        self.body_stage1 = nn.ModuleList([SFTResBlock(n_feats) for _ in range(m_point)])
        self.body_stage2 = nn.ModuleList([SFTResBlock(n_feats) for _ in range(n_blocks - m_point)])

        # --- Reconstruction Heads ---
        self.to_mid_res = nn.Conv2d(n_feats, 1, 3, padding=1)
        self.to_final_res = nn.Conv2d(n_feats, 1, 3, padding=1)

    def forward(self, x):
        # x: [B, 2, H, W] -> [Thermal, Optical]
        t_img = x[:, 0:1, :, :]
        o_img = x[:, 1:2, :, :]

        # 1. Feature Extraction & Conditioning
        t_feat = self.thermal_head(t_img)
        cond = self.conditioner(self.optical_head(o_img))

        # 2. Stage 1 Processing
        feat = t_feat
        for block in self.body_stage1:
            feat = block(feat, cond)
        
        # Intermediate output (useful for progressive training/deep supervision)
        mid_out = t_img + self.to_mid_res(feat)

        # 3. Stage 2 Processing
        for block in self.body_stage2:
            feat = block(feat, cond)

        # Final output
        final_out = t_img + self.to_final_res(feat)

        return final_out, mid_out