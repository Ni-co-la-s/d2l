"""
Implementation of NiN (Chapter 8.3)
"""

import torch
import torch.nn as nn
from ch_08_Modern_Convolutional_Neural_Networks.utils import get_detail_model


class Block(nn.Module):
    def __init__(self, conv: nn.Module, nb_1_1_conv: int = 2, add_pool: bool = True):
        super().__init__()

        self.out_channels = conv.out_channels

        self.conv = conv
        self.relu = nn.ReLU()
        self.convs1_1 = nn.ModuleList(
            [nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0) for _ in range(nb_1_1_conv)]
        )
        self.pool = nn.MaxPool2d(3, 2) if add_pool else None

    def forward(self, X):
        out = self.conv(X)
        out = self.relu(out)
        for conv1_1 in self.convs1_1:
            out = conv1_1(out)
            out = self.relu(out)
        if self.pool is not None:
            out = self.pool(out)
        return out


class NiNBase(nn.Module):
    def __init__(self, config: tuple):  # tuple of tuples contains (conv:nn.Module, nb_1_1_conv: int, add_pool:bool)
        super().__init__()

        self.relu = nn.ReLU()

        list_blocks = []
        for block_cfg in config:
            list_blocks.append(Block(block_cfg[0], block_cfg[1], block_cfg[2]))
        self.blocks = nn.Sequential(*list_blocks)

    def forward(self, X):
        out = self.blocks(X)
        out = torch.mean(out, (2, 3))
        return out


class NiN(NiNBase):
    def __init__(self, num_classes:int=10):
        config = (
            (nn.Conv2d(3, 96, 11, 4), 2, True),
            (nn.Conv2d(96, 256, 5, 1, 2), 2, True),
            (nn.Conv2d(256, 384, 3, 1, 1), 2, True),
            (nn.Conv2d(384, num_classes, 3, 1, 1), 2, False),
        )
        super().__init__(config=config)


if __name__ == "__main__":
    model = NiN()
    dummy_tensor = torch.rand(1, 3, 224, 224)
    get_detail_model(model, dummy_tensor)
