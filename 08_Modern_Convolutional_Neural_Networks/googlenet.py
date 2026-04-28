"""
Implementation of GoogLeNet (Chapter 8.4)
"""

import torch
import torch.nn as nn

from utils import get_detail_model


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        c1: int,
        c2: tuple,
        c3: tuple,
        c4: int,
    ):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, c1, 1), nn.ReLU())
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], 1), nn.ReLU(), nn.Conv2d(c2[0], c2[1], 3, 1, 1), nn.ReLU()
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], 1), nn.ReLU(), nn.Conv2d(c3[0], c3[1], 5, 1, 2), nn.ReLU()
        )
        self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(in_channels, c4, 1), nn.ReLU())

    def forward(self, X):
        out1 = self.b1(X)
        out2 = self.b2(X)
        out3 = self.b3(X)
        out4 = self.b4(X)
        return torch.cat((out1, out2, out3, out4), 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64, 192, 3, 1, 1)
        self.block1 = nn.Sequential(
            InceptionBlock(192, 64, (96, 128), (16, 32), 32), InceptionBlock(256, 128, (128, 192), (32, 96), 64)
        )
        self.block2 = nn.Sequential(
            InceptionBlock(480, 192, (96, 208), (16, 48), 64),
            InceptionBlock(512, 160, (112, 224), (24, 64), 64),
            InceptionBlock(512, 128, (128, 256), (24, 64), 64),
            InceptionBlock(512, 112, (144, 288), (32, 64), 64),
            InceptionBlock(528, 256, (160, 320), (32, 128), 128),
        )
        self.block3 = nn.Sequential(
            InceptionBlock(832, 256, (160, 320), (32, 128), 128), InceptionBlock(832, 384, (192, 384), (48, 128), 128)
        )

        self.lin = nn.Linear(1024, 10)

        self.pool = nn.MaxPool2d(3, 2, 1)
        self.relu = nn.ReLU()

    def forward(self, X):
        out = self.conv1(X)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.block1(out)
        out = self.pool(out)

        out = self.block2(out)
        out = self.pool(out)

        out = self.block3(out)
        out = self.pool(out)

        out = torch.mean(out, (2, 3))

        out = self.lin(out)

        return out


if __name__ == "__main__":
    dummy_tensor = torch.rand(1, 3, 224, 224)

    get_detail_model(GoogLeNet(), dummy_tensor)
