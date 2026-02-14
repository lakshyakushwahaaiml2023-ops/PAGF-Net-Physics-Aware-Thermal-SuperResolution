import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. OPTICAL-GUIDED ATTENTION (SFT Layer) ---
# "Spatial Feature Transform" - The Gold Standard for Guided SR
class SFTLayer(nn.Module):
    def __init__(self, channels):
        super(SFTLayer, self).__init__()
        # The SFT network learns to predict scale (gamma) and shift (beta)
        # from the condition (Optical) for the features (Thermal)
        self.sft_predict = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels * 2, 1) # Output gamma and beta
        )

    def forward(self, x, condition):
        # x: Thermal Features
        # condition: Optical Features (must match x shape)
        scale, shift = torch.chunk(self.sft_predict(condition), 2, dim=1)
        return x * (scale + 1) + shift

# --- 2. RESIDUAL BLOCK WITH SFT ---
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
        out = self.sft(out, cond) # Inject Optical Structure here
        return residual + out

# --- 3. MAIN ADVANCED NETWORK ---
class AdvancedOpticalSR(nn.Module):
    def __init__(self, base_channels=64, num_res_blocks=8):
        super(AdvancedOpticalSR, self).__init__()

        # Feature Extraction
        self.thermal_head = nn.Conv2d(1, base_channels, 3, padding=1)
        self.optical_head = nn.Conv2d(1, base_channels, 3, padding=1)

        # Condition Generator (Processes Optical separately)
        self.cond_net = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, base_channels, 1)
        )

        # Backbone (Split into two stages for Multi-Scale Supervision)
        mid_point = num_res_blocks // 2
        self.stage1 = nn.ModuleList([SFTResidualBlock(base_channels) for _ in range(mid_point)])
        self.stage2 = nn.ModuleList([SFTResidualBlock(base_channels) for _ in range(mid_point)])

        # Output Heads
        self.mid_output = nn.Conv2d(base_channels, 1, 3, padding=1)
        self.final_output = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x):
        # Unpack inputs
        thermal = x[:, 0:1, :, :] # (B, 1, H, W)
        optical = x[:, 1:2, :, :] # (B, 1, H, W)

        # 1. Extract Features
        t_feat = self.thermal_head(thermal)
        o_feat = self.optical_head(optical)
        
        # 2. Generate Condition
        cond = self.cond_net(o_feat)

        # 3. Stage 1 Processing
        curr_feat = t_feat
        for block in self.stage1:
            curr_feat = block(curr_feat, cond)
        
        # ---> Supervision Point 1 (Intermediate Result)
        mid_res = self.mid_output(curr_feat)
        out_mid = thermal + mid_res # Residual Learning

        # 4. Stage 2 Processing
        for block in self.stage2:
            curr_feat = block(curr_feat, cond)

        # ---> Supervision Point 2 (Final Result)
        final_res = self.final_output(curr_feat)
        out_final = thermal + final_res # Residual Learning

        return out_final, out_mid