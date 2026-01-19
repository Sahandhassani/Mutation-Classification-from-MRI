import torch
import torch.nn as nn


class HybridCNNGlobal(nn.Module):
    def __init__(
        self,
        in_chans=1,
        cnn_dim=128,
        global_dim=256,
        num_classes=4
    ):
        super().__init__()

        # ---- CNN Encoder ----
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, cnn_dim, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_dim, cnn_dim, 3, padding=1),
            nn.ReLU()
        )

        # ---- Global Transformer ----
        self.proj = nn.Linear(cnn_dim, global_dim)

        self.global_block = nn.TransformerEncoderLayer(
            d_model=global_dim,
            nhead=8,
            batch_first=True
        )

        self.head = nn.Linear(global_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)                  # (B, C, H, W)
        x = x.flatten(2).transpose(1, 2)     # (B, N, C)
        x = self.proj(x)
        x = self.global_block(x)
        x = x.mean(dim=1)
        return self.head(x)
