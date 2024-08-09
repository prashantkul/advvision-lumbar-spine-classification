import torch
import torch.nn as nn
from torch.nn import functional as F

class SE(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        se_channels = int(in_channels * se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, se_channels, 1)
        self.fc2 = nn.Conv2d(se_channels, in_channels, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        y = F.relu(self.fc1(y))
        y = F.sigmoid(self.fc2(y))
        return x * y

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, group_width=1, se_ratio=0.25):
        super().__init__()
        groups = out_channels // group_width

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = SE(out_channels, se_ratio)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)

class RegNetBase(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.stage1 = self._make_stage(32, 232, 2, 1, 232)
        self.stage2 = self._make_stage(232, 696, 7, 2, 232)
        self.stage3 = self._make_stage(696, 1392, 13, 2, 232)
        self.stage4 = self._make_stage(1392, 3712, 1, 2, 232)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3712, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, group_width):
        layers = [Bottleneck(in_channels, out_channels, stride, group_width)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels, out_channels, 1, group_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x