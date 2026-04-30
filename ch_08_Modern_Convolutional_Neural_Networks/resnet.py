import torch
import torch.nn as nn

from ch_08_Modern_Convolutional_Neural_Networks.utils import get_detail_model


class Residual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_11_conv: bool):
        super().__init__()
        if use_11_conv:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False)
            self.conv3 = nn.Conv2d(in_channels, out_channels, 1, 2, 0, bias=False)
            # In the book, there is no batch norm for that branch, but I add it to match the parameter count from https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
            self.conv3 = None
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, X: torch.tensor):
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.conv3 is not None:
            out = out + self.bn3(self.conv3(X))
        else:
            out = out + X
        return self.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.block1 = Residual(64, 64, False)
        self.block2 = Residual(64, 64, False)
        self.block3 = Residual(64, 128, True)
        self.block4 = Residual(128, 128, False)
        self.block5 = Residual(128, 256, True)
        self.block6 = Residual(256, 256, False)
        self.block7 = Residual(256, 512, True)
        self.block8 = Residual(512, 512, False)
        self.lin = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, X: torch.tensor):
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.pool(out)
        out = self.relu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = torch.mean(out, dim=(2, 3))
        out = self.lin(out)
        return out


if __name__ == "__main__":
    dummy_tensor = torch.rand(1, 3, 224, 224)

    get_detail_model(ResNet18(), dummy_tensor)
