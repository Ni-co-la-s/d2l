"""
Implementation of individual layers used in LeNet (for comparison with torch implementation)
"""

import torch
from torch import nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias: nn.Parameter | None = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = X @ self.weight.T
        if self.bias is not None:
            out += self.bias
        return out


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int | tuple, stride: int | tuple | None = None, padding: int = 0):
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            self.kernel_size = kernel_size
        else:
            raise ValueError(
                "kernel_size should be a tuple of length 2 or an integer (in which case, a square kernel is inferred)"
            )

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple) and len(stride) == 2:
            self.stride = stride
        else:
            raise ValueError(
                "stride should be a tuple of length 2 or an integer (in which case, a square stride is inferred)"
            )

        if isinstance(padding, int):
            self.padding = padding
        else:
            raise ValueError("padding should be an integer")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, C, H, W = X.shape
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride
        # Add padding
        X_pad = F.pad(X, (self.padding, self.padding, self.padding, self.padding), mode="constant", value=float("-inf"))

        # Define output
        Ho = (H + 2 * self.padding - Kh) // Sh + 1
        Wo = (W + 2 * self.padding - Kw) // Sw + 1

        out = X.new_zeros(B, C, Ho, Wo)

        for n in range(B):
            for d in range(C):
                for i in range(Ho):
                    for j in range(Wo):
                        h = i * Sh
                        w = j * Sw
                        out[n][d][i][j] = torch.max(X_pad[n, d, h : h + Kh, w : w + Kw])
        return out


class Conv2dNotOpti(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            self.kernel_size = kernel_size
        else:
            raise ValueError(
                "kernel_size should be a tuple of length 2 or an integer (in which case, a square kernel is inferred)"
            )

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple) and len(stride) == 2:
            self.stride = stride
        else:
            raise ValueError(
                "stride should be a tuple of length 2 or an integer (in which case, a square stride is inferred)"
            )

        if isinstance(padding, int):
            self.padding = padding
        else:
            raise ValueError("padding should be an integer")

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bias: nn.Parameter | None = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))  # bias is of shape (Co)

        self.weight = nn.Parameter(
            torch.zeros(out_channels, in_channels, *self.kernel_size)
        )  # Kernel is of shape (Co,Ci,Kh, Kw)

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # X is of shape (B,Ci,H,W)
        B, Ci, H, W = X.shape
        Co, Ci, Kh, Kw = self.weight.shape
        Sh, Sw = self.stride
        # Add padding
        X_pad = F.pad(X, (self.padding, self.padding, self.padding, self.padding), mode="constant", value=0)

        # Define output
        Ho = (H + 2 * self.padding - Kh) // Sh + 1
        Wo = (W + 2 * self.padding - Kw) // Sw + 1

        out = X.new_zeros(B, Co, Ho, Wo)

        # Main algo
        for n in range(B):
            for d in range(Co):
                for i in range(Ho):
                    for j in range(Wo):
                        acc = X.new_tensor(0.0)
                        for a in range(Kh):
                            for b in range(Kw):
                                for c in range(Ci):
                                    h = i * Sh + a
                                    w = j * Sw + b

                                    acc += X_pad[n, c, h, w] * self.weight[d, c, a, b]
                        if self.bias is not None:
                            acc += self.bias[d]
                        out[n, d, i, j] = acc

        return out


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            self.kernel_size = kernel_size
        else:
            raise ValueError(
                "kernel_size should be a tuple of length 2 or an integer (in which case, a square kernel is inferred)"
            )

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple) and len(stride) == 2:
            self.stride = stride
        else:
            raise ValueError(
                "stride should be a tuple of length 2 or an integer (in which case, a square stride is inferred)"
            )

        if isinstance(padding, int):
            self.padding = padding
        else:
            raise ValueError("padding should be an integer")

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bias: nn.Parameter | None = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))  # bias is of shape (Co)

        self.weight = nn.Parameter(
            torch.zeros(out_channels, in_channels, *self.kernel_size)
        )  # Kernel is of shape (Co,Ci,Kh, Kw)

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # X is of shape (B,Ci,H,W)
        B, Ci, H, W = X.shape
        Co, Ci, Kh, Kw = self.weight.shape
        Sh, Sw = self.stride
        # Add padding
        Ho = (H + 2 * self.padding - Kh) // Sh + 1
        Wo = (W + 2 * self.padding - Kw) // Sw + 1

        X_col = F.unfold(X, self.kernel_size, 1, self.padding, self.stride)  # X_col is of shape (B,Kh*Kw*Co,Ho*Wo)
        W_row = self.weight.reshape(Co, -1)  # W_row is of shape (Co, Kh*Kw*Ci)

        out = W_row @ X_col
        if self.bias is not None:
            out += self.bias.reshape(1, Co, 1)

        return out.reshape(B, Co, Ho, Wo)
