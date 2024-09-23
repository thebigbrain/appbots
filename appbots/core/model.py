import os.path

import torch
from torch import nn

from appbots.core.utils import get_models_dir


def save_model(model, name: str):
    path = os.path.join(get_models_dir(), f"{name}.pt")
    if os.path.exists(os.path.dirname(path)) is False:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, name: str):
    path = os.path.join(get_models_dir(), f"{name}.pt")
    if os.path.exists(path) is True:
        model.load_state_dict(torch.load(path, weights_only=True))
    return model


def is_model_exists(name: str) -> bool:
    path = os.path.join(get_models_dir(), f"{name}.pt")
    return os.path.exists(path) is True


class ModelCarer:
    epoch_losses: list[tuple[int, float]] = []
    average_loss: float = 0
    epoch = 0
    min_loss = 1.0

    EPSILON = 0.0001

    def __init__(self, model: nn.Module, name: str, num_epochs=100):
        self.model = model
        self.name = name
        self.num_epochs = num_epochs

    def load(self):
        self.model = load_model(self.model, self.name)
        return self.model

    def save_loss(self, epoch, loss: float):
        self.epoch_losses.append((epoch, loss))
        self.average_loss = loss

        if loss < self.min_loss:
            self.min_loss = loss

    def save_epoch(self, epoch: int):
        self.epoch = epoch
        if epoch > 5 and abs(self.average_loss - self.min_loss) < self.EPSILON:
            save_model(self.model, f"{self.name}_{epoch}")
            print(f"Save model at epoch {epoch} with loss {self.average_loss}")

    def done(self):
        save_model(self.model, self.name)

    def is_model_exists(self) -> bool:
        return is_model_exists(self.name)


class DictLossNormalizer:
    base_loss_dict: dict[str, float] = None

    def init_loss_dict(self, loss_dict: dict[str, torch.Tensor]):
        if self.base_loss_dict is not None:
            return

        self.base_loss_dict = {}
        for k, v in loss_dict.items():
            self.base_loss_dict[k] = v.item()

    def ave(self, loss_dict: dict[str, torch.Tensor]) -> tuple[float, dict[str, float]]:
        loss = 0.0
        ave_dict = {}
        for k, v in self.base_loss_dict.items():
            per_loss = loss_dict[k].item() / v
            loss += per_loss
            ave_dict[k] = per_loss

        return loss / len(self.base_loss_dict), ave_dict
