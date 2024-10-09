import torch

from appbots.core.images.builder import ImageTensorBuilder
from appbots.core.images.transforms import gray_transform
from appbots.core.nn_modules.canny import Canny
from appbots.core.plot import add_boxes, plot_images
from appbots.datasets.annotation import get_bot_memo

if __name__ == "__main__":
    image = get_bot_memo(2101)
    builder = ImageTensorBuilder()
    builder.load_from_url(image.get('screenshot'))

    gray: torch.Tensor = gray_transform(builder.tensor)
    model = Canny()
    boxes: list[torch.Tensor] = model(gray)
    gray = gray
    add_boxes(gray, boxes)
    t = builder.tensor

    h = builder.source.size[1]
    boxes = list(map(lambda b: torch.Tensor(b / h), boxes))

    plot_images([t, gray])


