"""
Based on the chapter 9.1 "Working with sequences".
Predict a time serie of a noisy sinusoidal function.
"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset, Subset
from enum import Enum

from dataclasses import dataclass
import wandb

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Model(Enum):
    LINEAR = 1
    MLP = 2


@dataclass
class Config:
    T: int = 1000
    noise_std: float = 0.2
    step: float = 0.01
    num_train: int | None = 604  # Number of steps to train on (between 0 and T-1). None means full dataset
    tau: int = 4  # Number of past observations used as input
    model: Model = Model.LINEAR
    num_epochs: int = 20
    project_name: str | None = "d2l"
    run_name: str | None = "linear_k1"
    k: int = 1  # number of autoregressive steps at eval time


class SineDataset(Dataset):
    def __init__(self, config: Config):
        self.config = config
        time = torch.arange(0, config.T, dtype=torch.float32)
        self.sine = torch.sin(config.step * time) + torch.randn(time.shape) * config.noise_std
        self.x = self.sine.unfold(0, config.tau, 1)[:-1]
        self.y = self.sine[config.tau :].unsqueeze(1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


if __name__ == "__main__":  # pragma: no cover
    config = Config()

    if config.project_name is not None:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config={"config": {k: v for k, v in vars(config).items() if not callable(v)}},
        )
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("eval/*")

    dataset = SineDataset(config)
    train_dataset = Subset(dataset, range(config.num_train)) if config.num_train is not None else dataset

    loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    if config.model == Model.LINEAR:
        model = nn.Sequential(nn.Linear(config.tau, 1))

    elif config.model == Model.MLP:
        model = nn.Sequential(
            nn.Linear(config.tau, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1),
        )

    else:
        raise ValueError("Model value is not valid")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0

        for X, y in loader:
            optimizer.zero_grad()
            output = model(X)

            loss = criterion(output, y)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(loader)
        if config.project_name is not None:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "epoch": epoch,
                }
            )

        print(f"Epoch: {epoch}, loss: {train_loss}")

    if config.project_name is not None:
        eval_loader = DataLoader(dataset, batch_size=128, shuffle=False)
        model.eval()
        # One step prediction with real data
        labels, single_preds = [], []

        with torch.no_grad():
            for X, y in eval_loader:
                output = model(X)
                labels.extend(y.squeeze())
                single_preds.extend(output.squeeze())

        # Autoregressive predictions after num_train

        sine = dataset.sine
        multi_preds = torch.zeros(config.T)
        with torch.no_grad():
            for i in range(config.tau, config.T - config.k):
                window = sine[i - config.tau : i].clone()
                for _ in range(config.k):
                    pred = model(window.unsqueeze(0))
                    window = torch.cat([window[1:], pred.squeeze().unsqueeze(0)])
                multi_preds[i + config.k] = pred.squeeze()

        for i, (label, single_pred, multi_pred) in enumerate(zip(labels, single_preds, multi_preds[config.tau :])):
            wandb.log(
                {
                    "eval/label": label,
                    "eval/single_preds": single_pred,
                    "eval/multi_preds": multi_pred,
                    "eval/step": i + config.tau,
                }
            )
