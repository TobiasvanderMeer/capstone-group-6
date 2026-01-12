# unet.py
# U-Net for 60x60 -> 60x60 regression (logK -> normalized head)

import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    Standard double conv block.
    Keeps height/width same.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UNet60(nn.Module):
    """
    U-Net specifically for 60x60 grid.
    Includes the 'Global Brain' fix in the bottleneck to catch that linear gradient.
    """

    def __init__(self, base_ch: int = 32, enforce_dirichlet_row0: bool = False):
        super().__init__()

        self.enforce_dirichlet_row0 = enforce_dirichlet_row0

        # Encoder path
        # -----------------------
        # Input is 1 channel (logK map)
        in_ch = 1

        # Level 1: 60x60
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)  # drops to 30x30

        # Level 2: 30x30
        self.enc2 = ConvBlock(base_ch, 2 * base_ch)
        self.pool2 = nn.MaxPool2d(2)  # drops to 15x15

        # Bottleneck: 15x15
        self.center = ConvBlock(2 * base_ch, 4 * base_ch)

        # -----------------------
        # THE FIX: Global Information Injection
        # Problem: Standard U-Net sucks at learning the global linear gradient (gravity).
        # Solution: Squeeze everything to 1x1, learn the bias, add it back.
        # -----------------------

        # 1. Squash 15x15 spatial dims to 1x1.
        # Prevents parameter explosion (keeps params ~65k instead of 14M).
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        feature_ch = 4 * base_ch  # e.g. 256 channels

        # 2. MLP to figure out the global slope/offset
        self.global_dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_ch, feature_ch),
            nn.ReLU(inplace=True),
            nn.Linear(feature_ch, feature_ch),
            nn.Unflatten(1, (feature_ch, 1, 1))  # Reshape to (N, C, 1, 1) for broadcasting
        )

        # Decoder path
        # -----------------------

        # Level 2 Up: 15 -> 30
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(4 * base_ch + 2 * base_ch, 2 * base_ch)

        # Level 1 Up: 30 -> 60
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(2 * base_ch + base_ch, base_ch)

        # Output projection
        self.out = nn.Conv2d(base_ch, 1, kernel_size=1)

        # Hardcoded boundary value for top row (physically h=100m, normalized)
        # y = (h - 146) / 37
        self.dirichlet_row0_value = (100.0 - 145.3243) / 35.5957

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, 1, 60, 60)

        # --- Encoder ---
        x1 = self.enc1(x)  # 60x60
        x2 = self.enc2(self.pool1(x1))  # 30x30

        # --- Bottleneck ---
        x_center = self.pool2(x2)  # 15x15
        x_center = self.center(x_center)

        # --- Global Injection ---
        # 1. Get average state of the system
        global_feat = self.global_pool(x_center)
        # 2. Process through dense layer
        global_feat = self.global_dense(global_feat)
        # 3. Add back to feature map (Residual connection)
        # Broadcasting happens here: (N, C, 15, 15) + (N, C, 1, 1)
        x_center = x_center + global_feat

        # --- Decoder ---
        d2 = self.up2(x_center)
        d2 = torch.cat([d2, x2], dim=1)  # Skip connection
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)  # Skip connection
        d1 = self.dec1(d1)

        out = self.out(d1)

        # Force physics on the top row if flag is set
        if self.enforce_dirichlet_row0:
            out[:, :, 0, :] = self.dirichlet_row0_value

        return out