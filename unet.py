# unet.py
# U-Net for 64x64 -> 64x64 regression (logK -> normalized head)

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


class UNet64(nn.Module):
    """
    U-Net specifically for 64x64 grid.
    Includes the 'Global Brain' fix in the bottleneck
    to capture the global hydraulic gradient.
    """

    def __init__(self, base_ch: int = 32, enforce_dirichlet_row0: bool = False):
        super().__init__()

        self.enforce_dirichlet_row0 = enforce_dirichlet_row0

        # ====================================================
        # Encoder
        # ====================================================

        in_ch = 1  # log(K)

        # Level 1: 64x64
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)  # 64 -> 32

        # Level 2: 32x32
        self.enc2 = ConvBlock(base_ch, 2 * base_ch)
        self.pool2 = nn.MaxPool2d(2)  # 32 -> 16

        # Bottleneck: 16x16
        self.center = ConvBlock(2 * base_ch, 4 * base_ch)

        # ====================================================
        # Global information injection (critical!)
        # ====================================================

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        feature_ch = 4 * base_ch

        self.global_dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_ch, feature_ch),
            nn.ReLU(inplace=True),
            nn.Linear(feature_ch, feature_ch),
            nn.Unflatten(1, (feature_ch, 1, 1))
        )

        # ====================================================
        # Decoder
        # ====================================================

        # Level 2 up: 16 -> 32
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(4 * base_ch + 2 * base_ch, 2 * base_ch)

        # Level 1 up: 32 -> 64
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(2 * base_ch + base_ch, base_ch)

        # Output
        self.out = nn.Conv2d(base_ch, 1, kernel_size=1)

        # Normalized Dirichlet value at y = 0 (optional)
        # Make sure these numbers match your dataset normalization!
        self.dirichlet_row0_value = (100.0 - 145.3243) / 35.5957

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, 1, 64, 64)

        # ================= Encoder =================
        x1 = self.enc1(x)                # 64x64
        x2 = self.enc2(self.pool1(x1))   # 32x32

        # ================= Bottleneck =================
        x_center = self.pool2(x2)        # 16x16
        x_center = self.center(x_center)

        # ---------- Global Brain ----------
        global_feat = self.global_pool(x_center)
        global_feat = self.global_dense(global_feat)
        x_center = x_center + global_feat  # broadcast add

        # ================= Decoder =================
        d2 = self.up2(x_center)          # 32x32
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                # 64x64
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        out = self.out(d1)

        # Enforce physics if requested
        if self.enforce_dirichlet_row0:
            out[:, :, 0, :] = self.dirichlet_row0_value

        return out
