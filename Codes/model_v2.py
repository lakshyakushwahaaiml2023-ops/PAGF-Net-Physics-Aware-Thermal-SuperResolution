import torch
import torch.nn as nn
import torch.nn.functional as F

class SFTLayer(nn.Module):
    """
    Spatial Feature Transform (SFT) Layer.
    Learns to modulate thermal features based on optical structure.
    """
    def __init__(self, channels):
        super(SFTLayer, self).__init__()
        self.sft_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels * 2, 1) # Output: Scale (gamma) & Shift (beta)
        )

    def forward(self, x, condition):
        # x: Thermal Features
        # condition: Optical Features
        scale, shift = torch.chunk(self.sft_net(condition), 2, dim=1)
        return x * (scale + 1) + shift

class SFTResidualBlock(nn.Module):
    def __init__(self, channels):
        super(SFTResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.sft = SFTLayer(channels)

    def forward(self, x, cond):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.sft(out, cond) # <--- Fusion happens here
        return residual + out

class PAGFNetV2(nn.Module):
    def __init__(self, channels=64, num_blocks=8):
        super(PAGFNetV2, self).__init__()
        
        # 1. Shallow Feature Extraction
        self.feat_thermal = nn.Conv2d(1, channels, 3, padding=1)
        self.feat_optical = nn.Conv2d(1, channels, 3, padding=1)
        
        # 2. Condition Generator (Optical Processing)
        self.cond_net = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1)
        )
        
        # 3. Deep Feature Modulation (The Backbone)
        # We split into two stages for Multi-Scale Supervision
        self.stage1 = nn.ModuleList([SFTResidualBlock(channels) for _ in range(num_blocks // 2)])
        self.stage2 = nn.ModuleList([SFTResidualBlock(channels) for _ in range(num_blocks // 2)])
        
        # 4. Reconstruction Heads
        self.tail_mid = nn.Conv2d(channels, 1, 3, padding=1)
        self.tail_final = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x):
        # Input x is (B, 2, H, W) -> Thermal, Optical
        thermal = x[:, 0:1, :, :]
        optical = x[:, 1:2, :, :]
        
        # Extract
        f_t = self.feat_thermal(thermal)
        f_o = self.feat_optical(optical)
        
        # Generate Condition
        cond = self.cond_net(f_o)
        
        # Stage 1
        curr = f_t
        for block in self.stage1:
            curr = block(curr, cond)
        
        # Output 1 (Coarse/Mid-level)
        res_mid = self.tail_mid(curr)
        out_mid = thermal + res_mid
        
        # Stage 2
        for block in self.stage2:
            curr = block(curr, cond)
            
        # Output 2 (Final Refined)
        res_final = self.tail_final(curr)
        out_final = thermal + res_final
        
        return out_final, out_mid