import pytest
import torch
import torch.nn as nn

import ch_07_Convolutional_Neural_Network.cnn as cnn
import ch_07_Convolutional_Neural_Network.modules as modules


class TestModules:
    # -------------------Linear-------------------

    @pytest.mark.parametrize("in_features", [1, 10, 100, 1000])
    @pytest.mark.parametrize("out_features", [1, 10, 100, 1000])
    @pytest.mark.parametrize("bias", [False, True])
    def test_linear_forward_pass(self, in_features: int, out_features: int, bias: bool) -> None:
        """
        Test output of forward pass for Linear layer compared to torch implementation
        """
        lin_base = nn.Linear(in_features, out_features, bias)
        lin_new = modules.Linear(in_features, out_features, bias)
        lin_new.weight = nn.Parameter(lin_base.weight)
        if bias:
            lin_new.bias = nn.Parameter(lin_base.bias)
        X = torch.randn(2, in_features)
        out_base = lin_base(X)
        out_new = lin_new(X)
        torch.testing.assert_close(out_base, out_new)

    # -------------------MaxPool2d-------------------

    @pytest.mark.parametrize(
        "kernel_size,padding",
        [
            (1, 0),
            (3, 0),
            (3, 1),
            (7, 3),
            ((1, 2), 0),
            ((5, 3), 1),
        ],
    )
    @pytest.mark.parametrize("stride", [None, 1, 3, (1, 3), (5, 1)])
    @pytest.mark.parametrize("in_shape", [(8, 1, 28, 28), (1, 3, 64, 32), (1, 64, 16, 16)])
    def test_maxpool_forward_pass(
        self, kernel_size: int | tuple, padding: int, stride: int | tuple, in_shape: tuple
    ) -> None:
        """
        Test output of forward pass for MaxPool2d layer compared to torch implementation
        """
        pool_base = nn.MaxPool2d(kernel_size, stride, padding)
        pool_new = modules.MaxPool2d(kernel_size, stride, padding)
        X = torch.randn(in_shape)
        out_base = pool_base(X)
        out_new = pool_new(X)
        torch.testing.assert_close(out_base, out_new)

    # -------------------Conv2d not opti-------------------

    @pytest.mark.parametrize("in_channels", [1, 3])
    @pytest.mark.parametrize("out_channels", [8])
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("image_size", [(16, 16)])
    @pytest.mark.parametrize(
        "kernel_size,padding",
        [
            (1, 0),
            (3, 1),
        ],
    )
    @pytest.mark.parametrize("stride", [1, 3])
    @pytest.mark.parametrize("bias", [False, True])
    def test_conv2d_not_opti_forward_pass(
        self,
        in_channels: int,
        out_channels: int,
        batch_size: int,
        image_size: tuple,
        kernel_size: int | tuple,
        padding: int,
        stride: int | tuple,
        bias: bool,
    ) -> None:
        """
        Test output of forward pass for our unoptimized conv layer compared to torch implementation.
        Because it is slow, reduced tests
        """
        conv_base = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        conv_new = modules.Conv2dNotOpti(in_channels, out_channels, kernel_size, stride, padding, bias)
        conv_new.weight = nn.Parameter(conv_base.weight)
        if bias and conv_base.bias is not None:
            conv_new.bias = nn.Parameter(conv_base.bias)
        X = torch.randn(batch_size, in_channels, *image_size)
        out_base = conv_base(X)
        out_new = conv_new(X)
        torch.testing.assert_close(out_base, out_new)

    # -------------------Conv2d opti-------------------

    @pytest.mark.parametrize(
        "in_channels,out_channels",
        [(1, 1), (1, 32), (32, 128), (128, 32)],
    )
    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("image_size", [(28, 28), (64, 32)])
    @pytest.mark.parametrize(
        "kernel_size,padding",
        [
            (1, 0),
            (3, 0),
            (3, 1),
            (7, 3),
            ((1, 2), 0),
            ((5, 3), 1),
        ],
    )
    @pytest.mark.parametrize("stride", [1, 3, (1, 3), (5, 1)])
    @pytest.mark.parametrize("bias", [False, True])
    def test_conv2d_opti_forward_pass(
        self,
        in_channels: int,
        out_channels: int,
        batch_size: int,
        image_size: tuple,
        kernel_size: int | tuple,
        padding: int,
        stride: int | tuple,
        bias: bool,
    ) -> None:
        """
        Test output of forward pass for our optimized Conv2d layer compared to torch implementation
        """
        conv_base = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        conv_new = modules.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        conv_new.weight = nn.Parameter(conv_base.weight)
        if bias and conv_base.bias is not None:
            conv_new.bias = nn.Parameter(conv_base.bias)
        X = torch.randn(batch_size, in_channels, *image_size)
        out_base = conv_base(X)
        out_new = conv_new(X)
        torch.testing.assert_close(out_base, out_new)


class TestCNN:
    # -------------------CNN forward pass -------------------

    @pytest.mark.parametrize(
        "implementation",
        [
            cnn.Implementation.MANUAL_BASE,
            cnn.Implementation.MANUAL_OPTI,
        ],
    )
    def test_cnn_forward_pass(self, implementation: cnn.Implementation) -> None:
        """
        Integration test of CNN using out implemented layers compared to using torch layers.
        Model was written for MNIST data so input of shape 1,1,28,28
        """
        cnn_base = cnn.CNN(cnn.Implementation.TORCH)
        cnn_new = cnn.CNN(implementation)

        cnn_new.load_state_dict(cnn_base.state_dict())

        X = torch.randn(1, 1, 28, 28)
        out_base = cnn_base(X)
        out_new = cnn_new(X)
        torch.testing.assert_close(out_base, out_new)
