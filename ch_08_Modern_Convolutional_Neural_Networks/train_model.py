"""
Train CNN models from chapter 8 on chosen dataset with chosen config.
Optionally log in wandb.
"""

import torch
import torch.nn as nn
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb

from collections.abc import Callable

import argparse

from ch_08_Modern_Convolutional_Neural_Networks.config import TrainConfig, Optim, Initialization

from ch_08_Modern_Convolutional_Neural_Networks.alex_net import AlexNet
from ch_08_Modern_Convolutional_Neural_Networks.vgg import VGGSmaller, VGG11, VGG16, VGG19
from ch_08_Modern_Convolutional_Neural_Networks.nin import NiN
from ch_08_Modern_Convolutional_Neural_Networks.googlenet import GoogLeNet
from ch_08_Modern_Convolutional_Neural_Networks.resnet import ResNet18

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ModelRegistry: dict[str, type[nn.Module]] = {
    "AlexNet": AlexNet,
    "VGGSmaller": VGGSmaller,
    "VGG11": VGG11,
    "VGG16": VGG16,
    "VGG19": VGG19,
    "NiN": NiN,
    "GoogLeNet": GoogLeNet,
    "ResNet18": ResNet18,
}


def get_model(name: str) -> type[nn.Module] | None:
    try:
        return ModelRegistry[name]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid model '{name}'. Choose from: {ModelRegistry.keys()}")


def get_optimizer(name: str) -> Optim | None:
    try:
        return Optim[name]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid optimizer '{name}'. Choose from: {[o.name for o in Optim]}")


def get_initializer(name: str) -> Initialization | None:
    try:
        return Initialization[name]
    except KeyError:
        raise argparse.ArgumentTypeError(
            f"Invalid initializer '{name}'. Choose from: {[o.name for o in Initialization]}"
        )


def init_weights(initialization: Initialization) -> Callable[[nn.Module], None]:
    def _init(m: nn.Module) -> None:
        if initialization == Initialization.Kaiming:
            if isinstance(m, (nn.Conv2d, nn.LazyConv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Linear, nn.LazyLinear)):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        elif initialization == Initialization.Xavier:
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.LazyConv2d, nn.LazyLinear)):
                nn.init.xavier_uniform_(m.weight)

    return _init


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run models on reduced version of ImageNet")

    parser.add_argument(
        "--model",
        type=get_model,
        required=True,
        help="Choose from: " + ", ".join(ModelRegistry.keys()),
    )

    parser.add_argument(
        "--dataset_name",
        help="Path of the dataset used by the dataloader",
        type=str,
    )

    parser.add_argument(
        "--project_name",
        help="Name of the wandb project. If not provided, train model without wandb",
        type=str,
    )

    parser.add_argument(
        "--run_name",
        help="Name of the wandb run",
        type=str,
    )

    parser.add_argument(
        "--group_name",
        help="Name of the wandb group",
        type=str,
    )

    parser.add_argument(
        "--optim",
        help="Name of the optimizer",
        type=get_optimizer,
    )

    parser.add_argument(
        "--lr",
        help="Learning rate used by the optimizer",
        type=float,
    )

    parser.add_argument(
        "--device",
        help='Device used for training. Choose "cuda" or "cpu""',
        type=str,
    )

    parser.add_argument(
        "--batch_size",
        help="Batch size used for training",
        type=int,
    )

    parser.add_argument(
        "--num_epochs",
        help="Number of epochs used for training",
        type=int,
    )

    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        default=False,
        help="Whether or not to use augmentation (default True)",
    )

    parser.add_argument(
        "--initialization",
        help="Name of the Initializer",
        type=get_initializer,
    )

    parser.add_argument(
        "--num_classes",
        help="Number of classes in the dataset (default 10 for imagenette)",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    config = TrainConfig()

    if args.project_name:
        config.project_name = args.project_name

    if args.run_name:
        config.run_name = args.run_name

    if args.group_name:
        config.group_name = args.group_name

    if args.optim:
        config.optim = args.optim

    if args.lr:
        config.lr = args.lr

    if args.device:
        config.device = args.device

    if args.batch_size:
        config.batch_size = args.batch_size

    if args.num_epochs:
        config.num_epochs = args.num_epochs

    if args.use_augmentation:
        config.use_augmentation = args.use_augmentation

    if args.dataset_name:
        config.dataset_name = args.dataset_name

    if args.initialization:
        config.initialization = args.initialization

    if args.num_classes:
        config.num_classes = args.num_classes

    print(config)

    model = args.model(num_classes=config.num_classes).to(config.device)

    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, 224, 224)  # Pass dummy input to materialize the lazy layers
        model(dummy_input.to(config.device))

    model.apply(init_weights(config.initialization))

    if config.project_name is not None:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            group=config.group_name,
            config={
                "config": {k: v for k, v in vars(config).items() if not callable(v)},
                "model": {
                    name: {
                        "params": param.numel(),
                    }
                    for name, param in model.named_parameters()
                },
            },
        )

        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("train_epoch/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")

    if config.use_augmentation:
        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.RandomErasing(p=0.2),
            ]
        )

    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.ImageFolder(root=os.path.join(config.dataset_name, "train"), transform=train_transform)

    val_dataset = datasets.ImageFolder(root=os.path.join(config.dataset_name, "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    optimizer: torch.optim.Optimizer

    if config.optim == Optim.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optim == Optim.SGD:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0

        for batch_idx, (X, y) in enumerate(train_loader):
            global_step = epoch * len(train_loader) + batch_idx
            X = X.to(config.device)
            y = y.to(config.device)

            optimizer.zero_grad()

            output = model(X)

            loss = criterion(output, y)

            step_accuracy = (output.argmax(1) == y).sum() / y.shape[0]
            step_loss = loss.item()

            train_loss += step_loss
            train_accuracy += step_accuracy

            if config.project_name is not None:
                wandb.log(
                    {
                        "train/accuracy": step_accuracy,
                        "train/loss": step_loss,
                        "global_step": global_step,
                    },
                )

            loss.backward()
            optimizer.step()

        train_accuracy /= len(train_loader)
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(val_loader):
                X = X.to(config.device)
                y = y.to(config.device)
                output = model(X)
                val_loss += criterion(output, y).item()
                val_accuracy += (output.argmax(1) == y).sum() / y.shape[0]

        val_accuracy /= len(val_loader)
        val_loss /= len(val_loader)

        if config.project_name is not None:
            wandb.log(
                {
                    "train_epoch/loss": train_loss,
                    "train_epoch/accuracy": train_accuracy,
                    "val/accuracy": val_accuracy,
                    "val/loss": val_loss,
                    "epoch": epoch,
                }
            )

        print(
            f"Epoch: {epoch + 1}, Train loss = {train_loss}, Train accuracy = {train_accuracy}, Val loss = {val_loss}, Val accuracy = {val_accuracy}"
        )
