import os.path

import torch

from appbots.core.utils import get_root_dir


def save_model(model, name: str):
    path = os.path.join(get_root_dir(), '.models', f"{name}.pt")
    if os.path.exists(os.path.dirname(path)) is False:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, name: str):
    path = os.path.join(get_root_dir(), '.models', f"{name}.pt")
    if os.path.exists(path) is True:
        model.load_state_dict(torch.load(path))
    return model
