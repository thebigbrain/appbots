import torch

from appbots.core.images.builder import ImageTensorBuilder
from appbots.core.images.transforms import gray_transform
from appbots.core.nn_modules.canny import Canny


def predict(image_url: str):
    builder = ImageTensorBuilder()
    builder.load_from_url(image_url)

    gray: torch.Tensor = gray_transform(builder.tensor)
    model = Canny()
    boxes: list[torch.Tensor] = model(gray)

    h = builder.source.size[1]
    boxes = list(map(lambda b: (b / h).tolist(), boxes))

    return boxes
