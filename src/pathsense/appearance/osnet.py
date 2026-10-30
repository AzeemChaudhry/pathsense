"""OSNet x0.25, a small person re-identification network: a clothing vector for a box.

The network follows OSNet (Zhou et al., ICCV 2019) as published in torchreid (MIT); the
layer names match its checkpoints so they load as they are. The weights are torchreid's
model zoo file, trained on MSMT17, whose agreement allows academic use only: do not ship
them. They are read with `weights_only=True`, so the file can only hold tensors, not code.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

CHANNELS = (16, 64, 96, 128)  # x0.25 width
LAYERS = (2, 2, 2)
FEATURES = 512
SIZE = (256, 128)  # height, width


class ConvBN(nn.Module):
    def __init__(self, cin, cout, kernel=1, stride=1, padding=0, relu=True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = relu

    def forward(self, x):
        x = self.bn(self.conv(x))
        return F.relu(x) if self.relu else x


class LightConv3x3(nn.Module):
    """1x1 then depthwise 3x3."""

    def __init__(self, cin, cout) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False, groups=cout)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)

    def forward(self, x):
        gate = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(gate))))


class OSBlock(nn.Module):
    """Four streams of 1 to 4 stacked 3x3 convolutions, mixed by a shared gate."""

    def __init__(self, cin, cout) -> None:
        super().__init__()
        mid = cout // 4
        self.conv1 = ConvBN(cin, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(*(LightConv3x3(mid, mid) for _ in range(2)))
        self.conv2c = nn.Sequential(*(LightConv3x3(mid, mid) for _ in range(3)))
        self.conv2d = nn.Sequential(*(LightConv3x3(mid, mid) for _ in range(4)))
        self.gate = ChannelGate(mid)
        self.conv3 = ConvBN(mid, cout, relu=False)
        self.downsample = ConvBN(cin, cout, relu=False) if cin != cout else None

    def forward(self, x):
        x1 = self.conv1(x)
        streams = (self.conv2a, self.conv2b, self.conv2c, self.conv2d)
        x2 = sum(self.gate(stream(x1)) for stream in streams)
        identity = self.downsample(x) if self.downsample is not None else x
        return F.relu(self.conv3(x2) + identity)


def stage(cin, cout, blocks, reduce):
    layers = [OSBlock(cin, cout), *(OSBlock(cout, cout) for _ in range(blocks - 1))]
    if reduce:
        layers.append(nn.Sequential(ConvBN(cout, cout), nn.AvgPool2d(2, 2)))
    return nn.Sequential(*layers)


class OSNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        c = CHANNELS
        self.conv1 = ConvBN(3, c[0], 7, 2, 3)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.conv2 = stage(c[0], c[1], LAYERS[0], reduce=True)
        self.conv3 = stage(c[1], c[2], LAYERS[1], reduce=True)
        self.conv4 = stage(c[2], c[3], LAYERS[2], reduce=False)
        self.conv5 = ConvBN(c[3], c[3])
        self.fc = nn.Sequential(nn.Linear(c[3], FEATURES), nn.BatchNorm1d(FEATURES), nn.ReLU())

    def forward(self, x):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.maxpool(self.conv1(x))))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def load(weights: Path) -> OSNet:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model = OSNet()
    own = model.state_dict()
    unused = sorted(set(state) - set(own))
    if [k for k in unused if not k.startswith("classifier.")]:
        raise ValueError(f"{weights}: layers OSNet x0.25 does not have: {unused[:5]}")
    model.load_state_dict({k: v for k, v in state.items() if k in own})  # strict: all present
    return model.eval()
