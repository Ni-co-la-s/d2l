"""
Implementation of LeNet and training in wandb.
Option to select between:
- torch modules: (cuda or cpu)
- Unoptimized modules from modules.py with the 7 nested loops(cpu)
- Optimized modules from modules.py with im2col (cpu)
"""

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from enum import Enum
from dataclasses import dataclass

import wandb
import ch_07_Convolutional_Neural_Network.modules as modules
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import cast

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Implementation(Enum):
    TORCH = 1
    MANUAL_BASE = 2
    MANUAL_OPTI = 3


class Optim(Enum):
    ADAM = 1
    SGD = 2


class DatasetVersion(Enum):
    MNIST = 1
    FASHION_MNIST = 2


@dataclass
class Config:
    implem: Implementation = Implementation.TORCH
    device: str = "cuda"
    num_epochs: int = 10
    batch_size: int = 32
    optim: Optim = Optim.ADAM
    lr: float = 1e-2
    project_name: str = "d2l"
    run_name: str | None = None
    job_type: str = "CNN"
    dataset: DatasetVersion = DatasetVersion.MNIST


class CNN(nn.Module):
    def __init__(self, implem: Implementation):
        super().__init__()
        conv: type[nn.Module]
        pool: type[nn.Module]
        linear: type[nn.Module]
        if implem == Implementation.TORCH:
            conv = nn.Conv2d
            pool = nn.MaxPool2d
            linear = nn.Linear
        elif implem == Implementation.MANUAL_OPTI:
            conv = modules.Conv2d
            pool = modules.MaxPool2d
            linear = modules.Linear
        elif implem == Implementation.MANUAL_BASE:
            conv = modules.Conv2dNotOpti
            pool = modules.MaxPool2d
            linear = modules.Linear
        else:
            raise ValueError("Implementation must be TORCH, MANUAL_BASE or MANUAL_OPTI")

        self.conv1 = conv(1, 6, 5, 1, 2)
        self.pool1 = pool(2)
        self.conv2 = conv(6, 16, 5)
        self.pool2 = pool(2)
        self.lin1 = linear(400, 120)
        self.lin2 = linear(120, 84)
        self.lin3 = linear(84, 10)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.conv1(X)
        out = self.pool1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = F.relu(out)
        out = out.reshape(out.shape[0], -1)
        out = self.lin1(out)
        out = F.relu(out)
        out = self.lin2(out)
        out = F.relu(out)
        out = self.lin3(out)
        return cast(torch.Tensor, out)


def make_grid_image(tensor: torch.Tensor) -> Figure:
    tensor = tensor[:5, :, :]  # Not too mamy fig on graph
    C, H, W = tensor.shape
    fig, axes = plt.subplots(1, C, figsize=(C * 2, 2))
    for i in range(C):
        axes[i].imshow(tensor[i].cpu().detach().numpy(), cmap="gray")
        axes[i].axis("off")
    plt.tight_layout()
    return fig


def make_activation_bar(tensor: torch.Tensor) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.bar(range(len(tensor)), tensor.cpu().detach().numpy())
    return fig


def to_wandb_image(img_tensor: torch.Tensor) -> wandb.Image:
    arr = img_tensor.cpu().numpy()
    arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype("uint8")
    return wandb.Image(arr)


def log_activations(
    model: CNN, real_img: torch.Tensor, noisy_image: torch.Tensor, epoch: int, global_step: int
) -> None:
    model.eval()
    with torch.no_grad():
        for label, img in [("real", real_img), ("noisy", noisy_image)]:
            img = img.to(config.device)

            a1 = F.relu(model.pool1(model.conv1(img)))
            a2 = F.relu(model.pool2(model.conv2(a1)))
            out = a2.reshape(a2.shape[0], -1)
            a3 = F.relu(model.lin1(out))
            a4 = F.relu(model.lin2(a3))
            a5 = F.softmax(model.lin3(a4), dim=1)

            wandb.log(
                {
                    "epoch": epoch,
                    f"activations/{label}/input": wandb.Image(img.cpu().numpy()),
                    f"activations/{label}/conv1": wandb.Image(make_grid_image(a1[0])),
                    f"activations/{label}/conv2": wandb.Image(make_grid_image(a2[0])),
                    f"activations/{label}/lin1": wandb.Image(make_activation_bar(a3[0])),
                    f"activations/{label}/lin2": wandb.Image(make_activation_bar(a4[0])),
                    f"activations/{label}/output": wandb.Image(make_activation_bar(a5[0])),
                },
                step=global_step,
            )

    plt.close("all")


if __name__ == "__main__":
    X = torch.rand(1, 1, 28, 28)

    config = Config()
    config.implem = Implementation.TORCH
    config.device = "cuda"
    config.num_epochs = 10
    config.batch_size = 32
    config.optim = Optim.SGD
    config.lr = 1e-2
    config.project_name = "d2l"
    config.run_name = "torch_with_base_params"
    config.job_type = "CNN"
    config.dataset = DatasetVersion.FASHION_MNIST

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    if config.dataset == DatasetVersion.MNIST:
        train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)

        test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    else:
        train_dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)

        test_dataset = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = CNN(config.implem).to(config.device)

    if config.implem != Implementation.TORCH:
        model2 = CNN(Implementation.TORCH).to(config.device)
        model.load_state_dict(model2.state_dict())

    criterion = nn.CrossEntropyLoss()

    optimizer: torch.optim.Optimizer
    if config.optim == Optim.SGD:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    elif config.optim == Optim.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if config.project_name is not None:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            job_type=config.job_type,
            config={
                "config": {k: v for k, v in vars(config).items() if not callable(v)},
                "model": {
                    name: {
                        "shape": list(param.shape),
                        "params": param.numel(),
                    }
                    for name, param in model.named_parameters()
                },
            },
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/epoch_*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")

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
            accuracy = (output.argmax(1) == y).sum() / config.batch_size
            train_accuracy += accuracy

            loss = criterion(output, y)
            step_loss = loss.item()
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if config.project_name is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/accuracy": accuracy,
                        "train/loss": step_loss,
                    },
                    step=global_step,
                )

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)

        model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(test_loader):
                X = X.to(config.device)
                y = y.to(config.device)
                output = model(X)
                test_loss += criterion(output, y).item()
                accuracy = (output.argmax(1) == y).sum() / config.batch_size
                test_accuracy += accuracy

                if batch_idx == 0 and config.project_name is not None:
                    noisy = torch.rand(1, 1, 28, 28)
                    real = X[:1, :, :, :]
                    log_activations(model, real, noisy, epoch, global_step)

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)
        print(f"Epoch: {epoch}, Train loss = {train_loss}, Test loss = {test_loss}")

        if config.project_name is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "train/epoch_acc": train_accuracy,
                    "val/loss": test_loss,
                    "val/acc": test_accuracy,
                },
                step=global_step,
            )
