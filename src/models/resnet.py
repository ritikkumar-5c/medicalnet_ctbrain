"""3D ResNet backbones from MedicalNet / Med3D (Tencent).

Ported to match the published pretrained checkpoints
(``resnet_{10,18,34,50}_23dataset.pth``) so their weights load cleanly.

Differences vs the original repo:
  * The segmentation decoder (``conv_seg``) is dropped — this is a backbone
    only. Pretrained backbone weights still load; the decoder keys are ignored.
  * For classification we keep standard strided downsampling in layer3/layer4
    instead of the dilated (stride-1) segmentation config. Conv *weight shapes*
    are identical either way, so pretrained weights load regardless; this just
    keeps the feature map small enough to global-pool cheaply.

The official shortcut convention is followed: type 'A' (zero-pad) for
resnet10/18/34, type 'B' (1x1 conv) for resnet50+.
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, dilation=dilation, bias=False,
    )


def downsample_basic_block(x, planes, stride):
    """Type 'A' shortcut: avg-pool then zero-pad channels (no params)."""
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4),
        dtype=out.dtype, device=out.device,
    )
    return torch.cat([out, zero_pads], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet3D(nn.Module):
    """Med3D ResNet backbone returning a feature map (no classification head)."""

    def __init__(self, block, layers, shortcut_type="B", in_channels=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(
            in_channels, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        self.out_channels = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block, planes=planes * block.expansion, stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )
        layers = [block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


_CFG = {
    10: (BasicBlock, [1, 1, 1, 1], "B"),
    18: (BasicBlock, [2, 2, 2, 2], "A"),
    34: (BasicBlock, [3, 4, 6, 3], "A"),
    50: (Bottleneck, [3, 4, 6, 3], "B"),
    101: (Bottleneck, [3, 4, 23, 3], "B"),
    152: (Bottleneck, [3, 8, 36, 3], "B"),
    200: (Bottleneck, [3, 24, 36, 3], "B"),
}


def build_resnet3d(depth=34, in_channels=1):
    if depth not in _CFG:
        raise ValueError(f"Unsupported depth {depth}; choose from {sorted(_CFG)}")
    block, layers, shortcut = _CFG[depth]
    return ResNet3D(block, layers, shortcut_type=shortcut, in_channels=in_channels)
