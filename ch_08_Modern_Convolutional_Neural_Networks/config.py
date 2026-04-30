from dataclasses import dataclass
from enum import Enum
import torch.nn as nn


class Optim(Enum):
    ADAM = 1
    SGD = 2


class Initialization(Enum):
    Xavier = 1
    Kaiming = 2


@dataclass
class TrainConfig:
    project_name: str = None
    run_name: str | None = "AlexNet_augmentation_adam"
    group_name: str | None = None
    dataset_name: str | None = "Imagenette"
    optim: Optim = Optim.ADAM
    lr: float = 0.0001
    device: str = "cuda"
    batch_size: int = 32
    num_epochs: int = 30
    use_augmentation: bool = False
    initialization: Initialization = Initialization.Xavier
    num_classes: int = 10


@dataclass
class ModelEntry:
    model: nn.Module
    config: TrainConfig
