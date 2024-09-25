import cv2
import numpy as np
import torch
from torch import nn

from appbots.core.images.builder import get_image_from_path
from appbots.core.images.transforms import gray_transform
from appbots.core.plot import plot_images
from appbots.core.utils import get_path


class Canny(nn.Module):
    def __init__(self, low_threshold=0.05, high_threshold=0.85):
        super().__init__()

        self.low_threshold = low_threshold * 255
        self.high_threshold = high_threshold * 255

    def forward(self, input_img: torch.Tensor):
        img = input_img.squeeze(0).numpy()
        # 边缘检测
        edges = cv2.Canny(img, self.low_threshold, self.high_threshold)

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)

        # 绘制边界框
        _boxes = []
        for i in range(1, num_labels):
            _x, _y, _w, _h = stats[i, :4]
            _boxes.append([_x, _y, _w, _h])

        return torch.tensor(np.array(_boxes))


def draw_boxes(img: torch.Tensor, bound_boxes: list[torch.Tensor]):
    img_np = img.squeeze(0).numpy()

    for box in bound_boxes:
        x, y, w, h = box.tolist()
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return torch.tensor(np.array([img_np]))


if __name__ == '__main__':
    # 定义图像转换
    t, _ = get_image_from_path(get_path("assets/test2.jpg"))
    gray: torch.Tensor = gray_transform(t)

    model = Canny()
    boxes = model(gray)

    box_img = draw_boxes(gray, boxes)

    plot_images([t, box_img])
