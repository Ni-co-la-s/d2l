"""
Implementation of VGG (Chapter 8.2).
"""

import torch
import torch.nn as nn

from utils import get_detail_model


class Block(nn.Module):
    def __init__(self, num_conv: int, in_channels: int, out_channels: int):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, 3, 1, 1)]
            + [nn.Conv2d(out_channels, out_channels, 3, 1, 1) for _ in range(num_conv - 1)]
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, X):
        out = X
        for conv in self.convs:
            out = conv(out)
            out = self.relu(out)
        out = self.pool(out)
        return out


class VGGBase(nn.Module):
    def __init__(self, arch: tuple):
        super().__init__()
        list_blocks = []
        input_channel = 3  # rgb images
        for block_cfg in arch:
            list_blocks.append(Block(block_cfg[0], input_channel, block_cfg[1]))
            input_channel = block_cfg[1]
        self.blocks = nn.Sequential(*list_blocks)

        self.lin1 = nn.LazyLinear(4096)
        self.lin2 = nn.Linear(4096, 4096)
        self.lin3 = nn.Linear(4096, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, X):
        out = self.blocks(X)
        out = torch.flatten(out, 1)
        out = self.lin1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.lin2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.lin3(out)
        return out


class VGGSmaller(VGGBase):
    def __init__(self):
        super().__init__(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)))


class VGG11(VGGBase):
    def __init__(self):
        super().__init__(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)))


class VGG16(VGGBase):
    def __init__(self):
        super().__init__(arch=((2, 64), (2, 128), (3, 256), (3, 512), (3, 512)))


class VGG19(VGGBase):
    def __init__(self):
        super().__init__(arch=((2, 64), (2, 128), (4, 256), (4, 512), (4, 512)))


if __name__ == "__main__":
    dummy_tensor = torch.rand(1, 3, 224, 224)

    get_detail_model(VGGSmaller(), dummy_tensor)
    get_detail_model(VGG11(), dummy_tensor)
    get_detail_model(VGG16(), dummy_tensor)
    get_detail_model(VGG19(), dummy_tensor)
