"""
Implementation of AlexNet (Chapter 8.1)
"""

import torch
import torch.nn as nn

from utils import get_detail_model


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, 4, 0)  # in_channels, out_channels, kernel_size, stride, padding
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.lin1 = nn.Linear(6400, 4096)
        self.lin2 = nn.Linear(4096, 4096)
        self.lin3 = nn.Linear(4096, 10)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, 2)  # kernel_size, padding
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        out = self.conv1(X)
        out = self.pool(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.pool(out)
        out = self.relu(out)
        out = out.reshape(out.shape[0], -1)
        out = self.lin1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.lin2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.lin3(out)
        return out


if __name__ == "__main__":
    dummy_tensor = torch.rand(1, 3, 224, 224)

    get_detail_model(AlexNet(), dummy_tensor)
