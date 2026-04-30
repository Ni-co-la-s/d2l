import pytest
import torch


@pytest.fixture(autouse=True)
def _config() -> None:
    torch.manual_seed(0)
