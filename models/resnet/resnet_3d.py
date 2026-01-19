import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=4, in_channels=1):
        super().__init__()

        self.in_planes = 32   # 🔻 smaller width

        self.conv1 = nn.Conv3d(
            in_channels, 32,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)   # logits


def resnet3d_10(num_classes=4, in_channels=1):
    return ResNet3D(
        block=BasicBlock3D,
        layers=[1, 1, 1, 1],
        num_classes=num_classes,
        in_channels=in_channels
    )



def resnet3d_18(num_classes=4, in_channels=1):
    return ResNet3D(
        block=BasicBlock3D,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        in_channels=in_channels
    )

def resnet3d_34(num_classes=4, in_channels=1):
    return ResNet3D(
        block=BasicBlock3D,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_channels=in_channels
    )
