import os.path
import shutil
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from appbots.core.time_util import get_now_str
from appbots.core.utils import get_models_dir, get_runs_path, mkdir, get_model_path


def save_model(model:  nn.Module | dict, model_path: str):
    mkdir(model_path)
    state = model.state_dict() if isinstance(model, nn.Module) else model
    torch.save(state, model_path)


def load_model(model: nn.Module | dict, model_path: str):
    if os.path.exists(model_path) is True:
        state = torch.load(model_path, weights_only=True)
        if isinstance(model, dict):
            model.update(state)
        elif isinstance(model, nn.Module):
            model.load_state_dict(state)


def is_model_exists(name: str) -> bool:
    path = os.path.join(get_models_dir(), f"{name}.pt")
    return os.path.exists(path) is True


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


class Criterion:
    def __init__(self, weight_loss: nn.Module):
        self._weight_loss = weight_loss

    def loss(self, inputs, targets) -> torch.Tensor:
        return self._weight_loss(inputs, targets)


def plot_loss(train_losses: list[float], loss_path: str):
    plt.plot(train_losses)
    # plt.scatter(train_losses)
    plt.title('train/loss')
    plt.grid(True)

    mkdir(loss_path)
    plt.savefig(loss_path)
    plt.show()


class TrainingRun:
    best_loss: float = None
    running_loss: float = None
    loss: float = None

    epoch: int = 0

    def __init__(self, name: str, num_epochs=None, epoch_iters=None):
        self.name = name
        self.num_epochs = num_epochs
        self.epoch_iters = epoch_iters
        self.train_losses = []

    @property
    def get_train_path(self):
        return Path(get_runs_path(f"{self.name}"))

    @property
    def loss_path(self):
        return Path(get_runs_path(f"{self.name}/loss.png"))

    @property
    def weights_dir(self):
        return get_runs_path(f"{self.name}/weights/")

    @property
    def best_model_path(self) -> Path:
        return Path(os.path.join(self.weights_dir, f"best.pt"))

    @property
    def last_model_path(self):
        return Path(os.path.join(self.weights_dir, f"last.pt"))

    def plot_loss(self):
        plot_loss(train_losses=self.train_losses, loss_path=self.loss_path.__str__())

    def load_checkpoint(self, model: nn.Module):
        if not self.best_model_path.exists():
            self.load_model(model, self.best_model_path)
        else:
            self.load_model(model, self.last_model_path)

    def save_checkpoint(self, model: nn.Module):
        epoch_loss = self.running_loss / self.epoch_iters
        self.best_loss = epoch_loss if self.best_loss is None else self.best_loss
        is_best = epoch_loss < self.best_loss
        self.best_loss = min(self.best_loss, epoch_loss)

        state = dict(model=model.state_dict(), epoch=self.epoch, train_losses=self.train_losses)
        if is_best:
            state['loss'] = self.best_loss
            save_model(state, str(self.best_model_path))
        state['loss'] = self.loss
        save_model(state, str(self.last_model_path))

        self.train_losses.append(epoch_loss)

    def load_model(self, model: nn.Module, model_path: Path):
        print(f"load model: {model_path}")
        state = dict()
        load_model(state, str(model_path))

        if state.get('model'):
            model.load_state_dict(state.get('model'))
        self.best_loss = state.get('loss')
        self.train_losses = state.get('train_losses', [])

    def save_model(self, model_path: Path):
        if os.path.exists(self.best_model_path):
            shutil.copyfile(self.best_model_path, model_path)
        else:
            shutil.copyfile(self.last_model_path, model_path)


class Trainer:
    best_loss = None

    def __init__(self, model: nn.Module, name: str,
                 train_loader,
                 num_epochs=100,
                 criterion: nn.Module = None,
                 optimizer: torch.optim.Optimizer = None):
        self.model = model
        self.name = name
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.criterion = Criterion(criterion or nn.CrossEntropyLoss())
        self.optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        run_name = f"{self.name}/train_{get_now_str()}".replace(":", "-").replace(" ", "_")

        self.run = TrainingRun(name=run_name,
                               num_epochs=num_epochs,
                               epoch_iters=len(train_loader) if train_loader else 0)

    @property
    def model_path(self) -> Path:
        return Path(get_model_path(f"{self.name}.pt"))

    def load(self):
        if Path(self.model_path).exists():
            self.run.load_model(self.model, model_path=self.model_path)

    def done(self):
        self.run.save_model(self.model_path)

    def is_model_exists(self) -> bool:
        return self.model_path.exists()

    def reset(self):
        if self.model_path.exists():
            os.remove(self.model_path)

    def train(self):
        if self.train_loader is None:
            raise ValueError("train_loader is None")

        self.load()

        for self.run.epoch in tqdm(range(self.num_epochs), desc="Epoch", total=self.num_epochs):
            self.run.running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # 获取输入数据
                inputs, targets = data

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion.loss(outputs, targets)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.run.running_loss += loss.item()

            self.run.save_checkpoint(self.model)

        self.done()

        self.run.plot_loss()

    def predict(self, x):
        self.load()
        self.model.eval()
        return self.model(x)


if __name__ == '__main__':
    pass
