import torch
import torch.nn as nn


def get_parameter_count(
    model: nn.Module, dummy_tensor: torch.Tensor | None = None, total_nb_params: int = None, verbose: bool = True
) -> int:
    """
    Get parameter count for a given model
    If model contains lazy layers, pass dummy tensor to materialize them.
    If total_nb_params is provided (can be obtained by running the function a first time), prints the proportion for each module
    """

    if dummy_tensor is not None:  # Necessary for lazy layers to be materialized
        model(dummy_tensor)

    nb_parameter_conv = 0
    nb_parameter_linear = 0

    if verbose:
        print("-" * 120)
        print("Breakdown per layer")

    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Conv2d)
            or isinstance(module, nn.LazyConv2d)
            or isinstance(module, nn.Linear)
            or isinstance(module, nn.LazyLinear)
        ):
            nb_param_weight = module.weight.numel()
            nb_param_bias = module.bias.numel() if module.bias is not None else 0
            nb_param = nb_param_weight + nb_param_bias
            if verbose:
                print(f"{name:<25}{module!r:<100}{nb_param:>12} parameters ({nb_param_bias} for the bias)", end="")
                if total_nb_params is not None:
                    print(f": {100 * nb_param / total_nb_params:.2f}% of total params")
                else:
                    print()

            if isinstance(module, nn.Conv2d) or isinstance(module, nn.LazyConv2d):
                nb_parameter_conv += nb_param

            elif isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear):
                nb_parameter_linear += nb_param

    nb_parameter_total = nb_parameter_linear + nb_parameter_conv

    if verbose:
        print("-" * 120)
        print(f"Total number of parameters: {nb_parameter_total}")
        if nb_parameter_conv > 0:
            print(
                f"Number of parameters for Conv layers: {nb_parameter_conv} ({100 * nb_parameter_conv / nb_parameter_total:.2f}% of count)"
            )
        if nb_parameter_linear > 0:
            print(
                f"Number of parameters for Linear layers: {nb_parameter_linear} ({100 * nb_parameter_linear / nb_parameter_total:.2f}% of count)"
            )

    return nb_parameter_total


def count_flop_forward(module: nn.Module, input: torch.tensor, output: torch.tensor):

    if isinstance(module, nn.Conv2d) or isinstance(module, nn.LazyConv2d):
        # TODO: Handle groups != 1
        H, W = output.shape[2], output.shape[3]
        ci, co = module.in_channels, module.out_channels
        kh, kw = module.kernel_size
        weights_ops = 2 * H * W * kh * kw * ci * co  # 2 because one for addition and one for multiplication
        bias_ops = H * W * co if module.bias is not None else 0
        module._flops = weights_ops + bias_ops
        module._flops_bias = bias_ops
        module._flops_weight = weights_ops

    elif isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear):
        ci, co = module.in_features, module.out_features
        weights_ops = 2 * ci * co
        bias_ops = co if module.bias is not None else 0
        module._flops = weights_ops + bias_ops
        module._flops_bias = bias_ops
        module._flops_weight = weights_ops


def get_flop_count(
    model: nn.Module, dummy_tensor: torch.tensor, total_nb_flops: int = None, verbose: bool = True
) -> int:

    hooks = []
    for module in model.modules():
        if (
            isinstance(module, nn.Conv2d)
            or isinstance(module, nn.LazyConv2d)
            or isinstance(module, nn.Linear)
            or isinstance(module, nn.LazyLinear)
        ):
            hooks.append(module.register_forward_hook(count_flop_forward))

    model(dummy_tensor)  # Necessary for lazy layers to be materialized and for the hook to work

    nb_flops_conv = 0
    nb_flops_linear = 0

    if verbose:
        print("-" * 120)
        print("Breakdown per layer")

    for name, module in model.named_modules():
        if hasattr(module, "_flops"):
            nb_flops = module._flops
            if verbose:
                print(f"{name:<25}{module!r:<100}{nb_flops:>12} flops ({module._flops_bias} for the bias)", end="")
                if total_nb_flops is not None:
                    print(f": {100 * nb_flops / total_nb_flops:.2f}% of total params")
                else:
                    print()

            if isinstance(module, nn.Conv2d) or isinstance(module, nn.LazyConv2d):
                nb_flops_conv += nb_flops

            elif isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear):
                nb_flops_linear += nb_flops

    nb_flops_total = nb_flops_linear + nb_flops_conv

    if verbose:
        print(f"Total number of flops: {nb_flops_total}")
        if nb_flops_conv > 0:
            print(
                f"Number of flopss for Conv layers: {nb_flops_conv} ({100 * nb_flops_conv / nb_flops_total:.2f}% of count)"
            )
        if nb_flops_linear > 0:
            print(
                f"Number of flopss for Linear layers: {nb_flops_linear} ({100 * nb_flops_linear / nb_flops_total:.2f}% of count)"
            )

    for h in hooks:
        h.remove()

    return nb_flops_total


def get_detail_model(model: nn.Module, dummy_tensor: torch.tensor):
    print("-" * 120)
    print(f"BREAKDOWN FOR {model.__class__.__name__}")
    print("-" * 120)

    print("PARAMETERS")
    # Retrieve params
    nb_params = get_parameter_count(model, dummy_tensor=dummy_tensor, verbose=False)
    # Print params
    get_parameter_count(model, dummy_tensor=dummy_tensor, total_nb_params=nb_params)

    print("-" * 120)

    print("FLOPS")
    # Retrieve flops
    nb_flops = get_flop_count(model, dummy_tensor=dummy_tensor, verbose=False)
    # Print params
    get_flop_count(model, dummy_tensor=dummy_tensor, total_nb_flops=nb_flops)
