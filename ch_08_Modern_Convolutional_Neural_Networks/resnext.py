"""
Implementation of ResNext50 (Chapter 8.6)
"""

import torch
import torch.nn as nn

from typing import cast
from ch_08_Modern_Convolutional_Neural_Networks.utils import get_detail_model


class BlockNaive(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, inter_channels: int, groups: int, strides: int, use_11_conv: bool
    ):
        super().__init__()
        branches = []

        if inter_channels % groups != 0:
            raise ValueError(
                f"Error: inter_channels should be dividible by the number of groups (Got {inter_channels} and {groups})"
            )

        channels_per_g = int(inter_channels / groups)

        for i in range(groups):
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, channels_per_g, 1, bias=False),
                    nn.BatchNorm2d(channels_per_g),
                    nn.ReLU(),
                    nn.Conv2d(channels_per_g, channels_per_g, 3, strides, 1, bias=False),
                    nn.BatchNorm2d(channels_per_g),
                    nn.ReLU(),
                )
            )

        self.branches = nn.ModuleList(branches)

        self.conv1 = nn.Conv2d(inter_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.proj: nn.Conv2d | None
        self.proj_bn: nn.BatchNorm2d | None
        if use_11_conv:
            self.proj = nn.Conv2d(in_channels, out_channels, 1, strides, bias=False)
            self.proj_bn = nn.BatchNorm2d(out_channels)
        else:
            self.proj = None
            self.proj_bn = None

        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        outputs = [branch(X) for branch in self.branches]
        output = torch.cat(outputs, dim=1)
        output = self.conv1(output)
        output = self.bn1(output)
        if self.proj is not None and self.proj_bn is not None:
            X = self.proj(X)
            X = self.proj_bn(X)
        return cast(torch.Tensor, self.relu(output + X))


class Block(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, inter_channels: int, groups: int, strides: int, use_11_conv: bool
    ):
        super().__init__()

        if inter_channels % groups != 0:
            raise ValueError(
                f"Error: inter_channels should be dividible by the number of groups (Got {inter_channels} and {groups})"
            )

        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, 3, strides, 1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv3 = nn.Conv2d(inter_channels, out_channels, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.proj: nn.Conv2d | None
        self.proj_bn: nn.BatchNorm2d | None

        if use_11_conv:
            self.proj = nn.Conv2d(in_channels, out_channels, 1, strides, bias=False)
            self.proj_bn = nn.BatchNorm2d(out_channels)
        else:
            self.proj = None
            self.proj_bn = None

        self.relu = nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        output = self.conv1(X)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)
        output = self.conv3(output)
        output = self.bn3(output)
        if self.proj is not None and self.proj_bn is not None:
            X = self.proj(X)
            X = self.proj_bn(X)
        return cast(torch.Tensor, self.relu(output + X))


class ResNext50_32x4d(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, 1)

        self.section1 = self._make_section(64, 256, 128, 32, 1, 3)
        self.section2 = self._make_section(256, 512, 256, 32, 2, 4)
        self.section3 = self._make_section(512, 1024, 512, 32, 2, 6)
        self.section4 = self._make_section(1024, 2048, 1024, 32, 2, 3)
        self.lin = nn.Linear(2048, num_classes)
        self.relu = nn.ReLU()

    def _make_section(
        self,
        in_channels: int,
        out_channels: int,
        inter_channels: int,
        groups: int,
        strides: int,
        nb_blocks: int,
    ) -> nn.Module:
        section = []
        section.append(Block(in_channels, out_channels, inter_channels, groups, strides, True))
        for i in range(nb_blocks - 1):
            section.append(Block(out_channels, out_channels, inter_channels, groups, 1, False))

        return nn.Sequential(*section)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.pool(out)
        out = self.relu(out)
        out = self.section1(out)
        out = self.section2(out)
        out = self.section3(out)
        out = self.section4(out)
        out = torch.mean(out, dim=(2, 3))
        out = self.lin(out)
        return cast(torch.Tensor, out)


if __name__ == "__main__":
    dummy_tensor = torch.rand(1, 3, 224, 224)

    get_detail_model(ResNext50_32x4d(), dummy_tensor)
