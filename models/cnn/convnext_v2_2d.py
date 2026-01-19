import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = x.permute(0, 3, 1, 2)
        return x + residual


class ConvNeXtV2_2D(nn.Module):
    def __init__(self, in_chans=1, num_classes=4, dim=96, depth=6):
        super().__init__()

        self.stem = nn.Conv2d(in_chans, dim, kernel_size=4, stride=4)
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(dim) for _ in range(depth)
        ])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
