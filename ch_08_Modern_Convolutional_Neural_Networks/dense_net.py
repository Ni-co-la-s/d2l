"""
Implementation of DenseNet (Chapter 8.7)
I implemented it referencing the paper instead of the chapter (https://arxiv.org/pdf/1608.06993)
Three variants (DenseNet121, DenseNet169, DenseNet201)
"""

import torch
import torch.nn as nn

from ch_08_Modern_Convolutional_Neural_Networks.utils import get_detail_model

from collections import OrderedDict
from typing import cast


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, num_convs: int, growth_rate: int):
        super().__init__()
        self.block = nn.ModuleList()
        cur_in_channels = in_channels
        for i in range(num_convs):
            layers: list[nn.Module] = []
            layers.append(nn.BatchNorm2d(cur_in_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(cur_in_channels, 4 * growth_rate, 1, bias=False))
            layers.append(nn.BatchNorm2d(4 * growth_rate))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(4 * growth_rate, growth_rate, 3, 1, 1, bias=False))
            cur_in_channels += growth_rate
            self.block.append(nn.Sequential(*layers))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.block:
            out = layer(X)
            X = torch.cat((X, out), 1)  # Concatenate along the channel dimension
        return X


class TransitionLayer(nn.Module):
    def __init__(self, in_channels: int, compression_factor: float):
        super().__init__()
        out_channel = in_channels * compression_factor
        if not out_channel.is_integer():
            raise ValueError(
                f"in_channels * compression_factor should be an integer: got {in_channels} and {compression_factor}."
            )
        out_channel = int(out_channel)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv11 = nn.Conv2d(in_channels, out_channel, 1, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.bn(X)
        out = self.conv11(out)
        out = self.relu(out)
        out = self.pool(out)
        return cast(torch.Tensor, out)


class DenseNetBase(nn.Module):
    def __init__(self, blocks_list: list[tuple], compression_factor: float = 0.5, num_classes: int = 10):
        """
        Create a DenseNet model.
        Accepts a takes a list of blocks configs (in_channels,num_convs,growth_rate)
        Inserts a Transition layer in-between each block.
        Rest of the layers are common to all versions of the model
        """
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.relu = nn.ReLU()

        blocks_dict: OrderedDict[str, nn.Module] = OrderedDict()

        for i, (in_channels, num_convs, growth_rate) in enumerate(blocks_list):
            blocks_dict[f"dense_block{i}"] = DenseBlock(in_channels, num_convs, growth_rate)
            if i < len(blocks_list) - 1:  # Last dense block has no transition layer following it
                out_channels = in_channels + num_convs * growth_rate
                blocks_dict[f"transition_layer{i}"] = TransitionLayer(out_channels, compression_factor)

        last_block_out = blocks_list[-1][0] + blocks_list[-1][1] * blocks_list[-1][2]
        self.blocks = nn.Sequential(blocks_dict)
        self.bn2 = nn.BatchNorm2d(last_block_out)
        self.lin = nn.Linear(last_block_out, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.conv(X)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.blocks(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = torch.mean(out, dim=(2, 3))
        out = self.lin(out)
        return cast(torch.Tensor, out)


class DenseNet_121(DenseNetBase):
    def __init__(self, num_classes: int = 10):
        blocks_list = [
            (64, 6, 32),
            (128, 12, 32),
            (256, 24, 32),
            (512, 16, 32),
        ]

        super().__init__(blocks_list, 0.5, num_classes)


class DenseNet_169(DenseNetBase):
    def __init__(self, num_classes: int = 10):
        blocks_list = [
            (64, 6, 32),
            (128, 12, 32),
            (256, 32, 32),
            (640, 32, 32),
        ]

        super().__init__(blocks_list, 0.5, num_classes)


class DenseNet_201(DenseNetBase):
    def __init__(self, num_classes: int = 10):
        blocks_list = [
            (64, 6, 32),
            (128, 12, 32),
            (256, 48, 32),
            (896, 32, 32),
        ]

        super().__init__(blocks_list, 0.5, num_classes)


if __name__ == "__main__":
    dummy_tensor = torch.rand(1, 3, 224, 224)

    get_detail_model(DenseNet_121(), dummy_tensor)
