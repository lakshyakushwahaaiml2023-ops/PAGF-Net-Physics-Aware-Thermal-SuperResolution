import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# Channel Attention Block
# -------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


# -------------------------------
# Residual Block
# -------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out)
        return residual + out


# -------------------------------
# Main Network
# -------------------------------
class OpticalGuidedSR(nn.Module):
    def __init__(self, base_channels=64, num_res_blocks=6):
        super(OpticalGuidedSR, self).__init__()

        # Thermal branch
        self.thermal_head = nn.Conv2d(1, base_channels, 3, padding=1)

        # Optical branch
        self.optical_head = nn.Conv2d(1, base_channels, 3, padding=1)

        # Fusion
        self.fusion_conv = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_res_blocks)]
        )

        # Output residual predictor
        self.output_conv = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x):
        thermal = x[:, 0:1, :, :]
        optical = x[:, 1:2, :, :]

        t_feat = self.thermal_head(thermal)
        o_feat = self.optical_head(optical)

        fused = torch.cat([t_feat, o_feat], dim=1)
        fused = F.relu(self.fusion_conv(fused))

        refined = self.res_blocks(fused)

        residual = self.output_conv(refined)

        # Residual learning (VERY IMPORTANT)
        output = thermal + residual

        return output
