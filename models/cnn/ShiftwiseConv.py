import torch
import torch.nn as nn
import torch.nn.functional as F

class Shift2D(nn.Module):
    """
    Channel-wise spatial shift (zero-parameter).
    """
    def __init__(self, shift):
        """
        shift: (dy, dx)
        """
        super().__init__()
        self.dy, self.dx = shift

    def forward(self, x):
        B, C, H, W = x.shape
        out = torch.zeros_like(x)

        y1 = max(0, self.dy)
        y2 = min(H, H + self.dy)
        x1 = max(0, self.dx)
        x2 = min(W, W + self.dx)

        out[:, :, y1:y2, x1:x2] = x[:, :, y1 - self.dy:y2 - self.dy, x1 - self.dx:x2 - self.dx]
        return out

class ShiftwiseConv2D(nn.Module):
    """
    2D Shiftwise Convolution (simplified, research-faithful).

    Replaces large kernels using:
      - small conv
      - multi-directional shifts
      - feature fusion
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, shift_groups=5):
        super().__init__()

        assert kernel_size == 3, "ShiftwiseConv assumes 3x3 base conv"

        # Base feature extraction (small kernel)
        self.base_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Define spatial shifts (like retina-inspired multi-paths)
        self.shifts = nn.ModuleList([
            Shift2D((0, 0)),    # identity
            Shift2D((-1, 0)),   # up
            Shift2D((1, 0)),    # down
            Shift2D((0, -1)),   # left
            Shift2D((0, 1)),    # right
        ])

        # Fuse shifted features
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * len(self.shifts), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.base_conv(x)

        shifted = [shift(x) for shift in self.shifts]
        x = torch.cat(shifted, dim=1)

        return self.fuse(x)

class ShiftwiseResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = ShiftwiseConv2D(channels, channels)

    def forward(self, x):
        return x + self.conv(x)

class ShiftwiseCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(
            ShiftwiseResidualBlock(64),
            ShiftwiseResidualBlock(64)
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 2, stride=2),
            ShiftwiseResidualBlock(128),
            ShiftwiseResidualBlock(128)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return self.head(x)
