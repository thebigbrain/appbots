import torch
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes

plt.rcParams["savefig.bbox"] = 'tight'


def show_tensor(bounded_images: torch.Tensor):
    if not isinstance(bounded_images, list):
        bounded_images = [bounded_images]
    for i, img in enumerate(bounded_images):
        plt.imshow(F.to_pil_image(img))
        plt.show()


def show_bounding_boxes(image, boxes):
    colors = ["blue", "red", "green"]
    result = draw_bounding_boxes(image, boxes[:len(colors)], colors=colors, width=5)
    show_tensor(result)
