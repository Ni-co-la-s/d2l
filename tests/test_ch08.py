import pytest
import torch
import torch.nn as nn

from ch_08_Modern_Convolutional_Neural_Networks import (
    alex_net,
    googlenet,
    nin,
    resnet,
    resnext,
    vgg,
)
import ch_08_Modern_Convolutional_Neural_Networks.utils as utils


class TestModules:
    # -------------------Compare implem ResNext blocks-------------------

    @pytest.mark.parametrize(
        "in_channels,out_channels,inter_channels,groups,strides,use_11_conv",
        [
            # Same in/out channels, no projection, stride 1
            (64, 64, 128, 32, 1, False),
            # Channel change, requires projection
            (64, 256, 128, 32, 1, True),
            # Stride 2 downsampling with projection
            (128, 256, 128, 32, 2, True),
            # groups == inter_channels
            (64, 64, 32, 32, 1, False),
            # groups == 1 (standard ResNet block)
            (32, 64, 64, 1, 1, True),
            (32, 128, 64, 8, 2, True),
        ],
    )
    def test_resnext_block_naive_matches_grouped_implementation(
        self,
        in_channels: int,
        out_channels: int,
        inter_channels: int,
        groups: int,
        strides: int,
        use_11_conv: bool,
    ) -> None:
        """
        Compares 2 implementations of ResNext blocks (with and without using Conv2d groups)
        """

        X = torch.randn(2, in_channels, 16, 16)

        naive = resnext.BlockNaive(in_channels, out_channels, inter_channels, groups, strides, use_11_conv)
        grouped = resnext.Block(in_channels, out_channels, inter_channels, groups, strides, use_11_conv)

        channels_per_g = inter_channels // groups

        # Copy parameters from naive (multi-branch) into grouped (single grouped conv).
        with torch.no_grad():
            for i, branch in enumerate(naive.branches):
                assert isinstance(branch, nn.Sequential)
                naive_1x1 = branch[0]  # Conv2d
                naive_bn1 = branch[1]  # BatchNorm2d
                naive_3x3 = branch[3]  # Conv2d
                naive_bn2 = branch[4]  # BatchNorm2d
                assert isinstance(naive_1x1, nn.Conv2d)
                assert isinstance(naive_bn1, nn.BatchNorm2d)
                assert isinstance(naive_3x3, nn.Conv2d)
                assert isinstance(naive_bn2, nn.BatchNorm2d)

                s = slice(i * channels_per_g, (i + 1) * channels_per_g)
                assert naive_bn1.weight is not None
                assert naive_bn1.bias is not None
                assert naive_bn1.running_mean is not None
                assert naive_bn1.running_var is not None
                assert naive_bn2.weight is not None
                assert naive_bn2.bias is not None
                assert naive_bn2.running_mean is not None
                assert naive_bn2.running_var is not None

                assert grouped.bn1.weight is not None
                assert grouped.bn1.bias is not None
                assert grouped.bn1.running_mean is not None
                assert grouped.bn1.running_var is not None
                assert grouped.bn2.weight is not None
                assert grouped.bn2.bias is not None
                assert grouped.bn2.running_mean is not None
                assert grouped.bn2.running_var is not None

                grouped.conv1.weight[s].copy_(naive_1x1.weight)
                grouped.bn1.weight[s].copy_(naive_bn1.weight)
                grouped.bn1.bias[s].copy_(naive_bn1.bias)
                grouped.bn1.running_mean[s].copy_(naive_bn1.running_mean)
                grouped.bn1.running_var[s].copy_(naive_bn1.running_var)

                grouped.conv2.weight[s].copy_(naive_3x3.weight)
                grouped.bn2.weight[s].copy_(naive_bn2.weight)
                grouped.bn2.bias[s].copy_(naive_bn2.bias)
                grouped.bn2.running_mean[s].copy_(naive_bn2.running_mean)
                grouped.bn2.running_var[s].copy_(naive_bn2.running_var)

            assert naive.bn1.weight is not None
            assert naive.bn1.bias is not None
            assert naive.bn1.running_mean is not None
            assert naive.bn1.running_var is not None
            assert grouped.bn3.weight is not None
            assert grouped.bn3.bias is not None
            assert grouped.bn3.running_mean is not None
            assert grouped.bn3.running_var is not None

            # Final projection conv + BN
            grouped.conv3.weight.copy_(naive.conv1.weight)
            grouped.bn3.weight.copy_(naive.bn1.weight)
            grouped.bn3.bias.copy_(naive.bn1.bias)
            grouped.bn3.running_mean.copy_(naive.bn1.running_mean)
            grouped.bn3.running_var.copy_(naive.bn1.running_var)

            # Shortcut 1x1 conv + BN (if present)
            if use_11_conv:
                assert naive.proj is not None
                assert naive.proj_bn is not None
                assert grouped.proj is not None
                assert grouped.proj_bn is not None
                assert naive.proj_bn.weight is not None
                assert naive.proj_bn.bias is not None
                assert naive.proj_bn.running_mean is not None
                assert naive.proj_bn.running_var is not None
                assert grouped.proj_bn.weight is not None
                assert grouped.proj_bn.bias is not None
                assert grouped.proj_bn.running_mean is not None
                assert grouped.proj_bn.running_var is not None

                grouped.proj.weight.copy_(naive.proj.weight)
                grouped.proj_bn.weight.copy_(naive.proj_bn.weight)
                grouped.proj_bn.bias.copy_(naive.proj_bn.bias)
                grouped.proj_bn.running_mean.copy_(naive.proj_bn.running_mean)
                grouped.proj_bn.running_var.copy_(naive.proj_bn.running_var)

        naive.eval()
        grouped.eval()
        with torch.no_grad():
            out_naive = naive(X)
            out_grouped = grouped(X)
        torch.testing.assert_close(out_naive, out_grouped, rtol=1e-4, atol=1e-4)

        naive.train()
        grouped.train()
        out_naive_train = naive(X)
        out_grouped_train = grouped(X)
        torch.testing.assert_close(out_naive_train, out_grouped_train, rtol=1e-4, atol=1e-4)

        out_naive_train.sum().backward()
        out_grouped_train.sum().backward()
        torch.testing.assert_close(
            naive.conv1.weight.grad,
            grouped.conv3.weight.grad,
            rtol=1e-4,
            atol=1e-4,
        )


class TestModels:
    # -------------------Test forward and backward pass models-------------------

    @pytest.mark.parametrize(
        "model_cls",
        [
            alex_net.AlexNet,
            vgg.VGGSmaller,
            vgg.VGG11,
            vgg.VGG16,
            vgg.VGG19,
            nin.NiN,
            googlenet.GoogLeNet,
            resnet.ResNet18,
            resnext.ResNext50_32x4d,
        ],
    )
    @pytest.mark.parametrize("input_shape", [(2, 3, 224, 224)])
    @pytest.mark.parametrize("num_classes", [10, 1000])
    def test_model_forward_backward(self, model_cls: nn.Module, input_shape: tuple, num_classes: int) -> None:
        model = model_cls(num_classes=num_classes)
        x = torch.randn(*input_shape)

        # Forward
        out = model(x)
        assert out.shape == (input_shape[0], num_classes)
        assert torch.isfinite(out).all(), "forward produced NaN/Inf"

        # Backward
        loss = out.sum()
        loss.backward()

        # Every parameter should have received a gradient
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name}: no gradient computed (dead code?)"
            assert torch.isfinite(p.grad).all(), f"{name}: NaN/Inf gradient"
            assert p.grad.abs().sum() > 0, f"{name}: zero gradient (disconnected?)"

    # -------------------Check parameter count for models-------------------

    @pytest.mark.parametrize(
        "model_cls,num_classes,expected_params",
        [
            (
                vgg.VGG11,
                1000,
                132_863_336,  # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html
            ),
            (
                vgg.VGG16,
                1000,
                138_357_544,  # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html
            ),
            (
                vgg.VGG19,
                1000,
                143_667_240,  # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html
            ),
            (
                resnet.ResNet18,
                1000,
                11_689_512,  # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
            ),
            (
                resnext.ResNext50_32x4d,
                1000,
                25_028_904,  # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnext50_32x4d.html
            ),
        ],
    )
    def test_model_param_count(self, model_cls: nn.Module, num_classes: int, expected_params: int) -> None:
        """
        Test parameter count for models using the methods implemented in utils file. Compare it with reference from paper/implem.
        - AlexNet is not included because the d2l chapter diverged from the original paper and the torchvision implem is from another paper
        - GoogLeNet is not included because the d2l chapter diverged from the original paper (no batch norm)
        """
        model = model_cls(num_classes=num_classes)
        dummy_tensor = torch.rand(1, 3, 224, 224)
        n = utils.get_parameter_count(model, dummy_tensor, verbose=False)
        assert n == expected_params, f"got {n:,} for model {model.__class__.__name__}, expected {expected_params:,}"

    # -------------------Check Flops count for models-------------------

    @pytest.mark.parametrize(
        "model_cls,num_classes,published",
        [
            (
                vgg.VGG11,
                1000,
                7.61 * 10**9,
            ),  # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html
            (
                vgg.VGG16,
                1000,
                15.47 * 10**9,
            ),  # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html
            (
                vgg.VGG19,
                1000,
                19.63 * 10**9,
            ),  # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html
            (
                resnet.ResNet18,
                1000,
                1.81 * 10**9,
            ),  # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
            (
                resnext.ResNext50_32x4d,
                1000,
                4.23 * 10**9,
            ),  # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
        ],
    )
    def test_flops_match_paper(self, model_cls: nn.Module, num_classes: int, published: float) -> None:
        """
        Test parameter count for models using the methods implemented in utils file. Compare it with reference from paper/implem.
        Because the utils method compute actual FLOPs and papers report the MACs instead, we divide our count by 2.
        Because the number reported is not exact, we accept a deviation of up to 0.5 percent
        - AlexNet is not included because the d2l chapter diverged from the original paper and the torchvision implem is from another paper
        - GoogLeNet is not included because the d2l chapter diverged from the original paper (no batch norm)
        """
        model = model_cls(num_classes=num_classes)
        dummy_tensor = torch.rand(1, 3, 224, 224)
        flops = utils.get_flop_count(model, dummy_tensor, verbose=False) / 2
        assert flops == pytest.approx(published, rel=0.005)
