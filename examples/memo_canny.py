import torch
from torchvision.transforms import v2

from appbots.core.images.builder import ImageTensorBuilder
from appbots.core.images.transforms import gray_transform
from appbots.core.nn_modules.canny import Canny
from appbots.core.plots.plot import add_boxes, plot_images
from appbots.datasets.annotation import get_bot_memo

if __name__ == "__main__":
    image = get_bot_memo(2100)
    builder = ImageTensorBuilder()
    builder.load_from_url(image.get('screenshot'))

    gray: torch.Tensor = gray_transform(builder.tensor)
    model = Canny()
    boxes: list[torch.Tensor] = model(gray)
    add_boxes(gray, boxes)
    t = builder.tensor

    h = builder.source.size[1]
    boxes = list(map(lambda b: b / h, boxes))

    resized_gray = v2.Resize(30)(gray)
    plot_images([t, gray])
