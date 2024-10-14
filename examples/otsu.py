import cv2
import torch

from appbots.core.images.transforms import gray_transform
from appbots.core.plots.plot import plot_images, add_boxes
from appbots.core.utils import get_assets


def find_bound_box(binary_tensor: torch.Tensor):
    nonzero_indices = torch.nonzero(binary_tensor)
    x_min = torch.min(nonzero_indices[:, 2])
    y_min = torch.min(nonzero_indices[:, 1])
    x_max = torch.max(nonzero_indices[:, 2])
    y_max = torch.max(nonzero_indices[:, 1])
    x_min, y_min, x_max, y_max = x_min.item(), y_min.item(), x_max.item(), y_max.item()
    return torch.tensor([x_min, y_min, x_max - x_min, y_max - y_min])


if __name__ == "__main__":
    # 读取图像
    img = cv2.imread(get_assets("test4-1.jpg"), 0)

    # Otsu阈值分割
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh_tensor = gray_transform(torch.tensor(thresh).unsqueeze(0))
    bound_box = find_bound_box(thresh_tensor)

    add_boxes(thresh_tensor, [bound_box], color=(50, 0, 0))
    plot_images([torch.tensor(img).unsqueeze(0), thresh_tensor])
