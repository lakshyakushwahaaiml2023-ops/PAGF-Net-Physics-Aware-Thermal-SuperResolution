import torch
import torch.nn as nn
import torch.nn.functional as F

class SFTLayer(nn.Module):
    """
    Spatial Feature Transform (SFT) Layer.
    Learns spatial-wise affine transformation (gamma/beta) to modulate 
    thermal features using optical guidance.
    """
    def __init__(self, channels):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels * 2, 1)
        )

    def forward(self, x, condition):
        # Generate scale (gamma) and shift (beta) maps from the condition
        params = self.mapping(condition)
        gamma, beta = torch.chunk(params, 2, dim=1)
        
        # Apply affine transformation: (1 + gamma) ensures identity start
        return x * (gamma + 1) + beta

class SFTResidualBlock(nn.Module):
    """Residual Block integrated with SFT modulation."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.sft   = SFTLayer(channels)

    def forward(self, x, cond):
        res = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.sft(out, cond)
        return res + out

class PAGFNetV2(nn.Module):
    """
    Physics-Aware Guided Fusion Network (V2).
    Utilizes SFT for structural guidance and multi-scale supervision.
    """
    def __init__(self, n_feats=64, n_blocks=8):
        super().__init__()

        # --- 1. Shallow Feature Extraction ---
        self.thermal_head = nn.Conv2d(1, n_feats, 3, padding=1)
        self.optical_head = nn.Conv2d(1, n_feats, 3, padding=1)

        # --- 2. Condition Generator ---
        self.conditioner = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, n_feats, 1)
        )

        # --- 3. Deep SFT Backbone ---
        # Splitting into two stages for Progressive/Multi-Scale Supervision
        m_point = n_blocks // 2
        self.stage1 = nn.ModuleList([SFTResidualBlock(n_feats) for _ in range(m_point)])
        self.stage2 = nn.ModuleList([SFTResidualBlock(n_feats) for _ in range(n_blocks - m_point)])

        # --- 4. Reconstruction Heads ---
        self.upsample_mid   = nn.Conv2d(n_feats, 1, 3, padding=1)
        self.upsample_final = nn.Conv2d(n_feats, 1, 3, padding=1)

    def forward(self, x):
        # x shape: (B, 2, H, W) -> [Thermal, Optical]
        t_in = x[:, 0:1, :, :]
        o_in = x[:, 1:2, :, :]

        # Extract features and generate the prior condition
        f_t = self.thermal_head(t_in)
        cond = self.conditioner(self.optical_head(o_in))

        # Stage 1: Initial Modulation
        feat = f_t
        for block in self.stage1:
            feat = block(feat, cond)
        
        # Intermediate Coarse Result (Deep Supervision)
        out_mid = t_in + self.upsample_mid(feat)

        # Stage 2: Fine Modulation
        for block in self.stage2:
            feat = block(feat, cond)

        # Final Refined Result
        out_final = t_in + self.upsample_final(feat)

        return out_final, out_mid