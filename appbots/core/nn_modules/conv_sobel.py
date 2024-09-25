import torch
from torch import nn
import torch.nn.functional as F


class SobelConv2d(nn.Module):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def __init__(self, low_threshold=0.05, high_threshold=0.15):
        super().__init__()

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward(self, x: torch.Tensor):
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)

        # 梯度幅值和方向
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        edges = non_max_suppression(grad_mag, grad_x, grad_y)

        final_edges = double_thresholding(
            edges=edges,
            high_threshold=self.high_threshold,
            low_threshold=self.low_threshold
        )

        return final_edges


# 非极大值抑制（简化实现）
def non_max_suppression(grad_mag, grad_x, grad_y):
    # ... 非极大值抑制的实现 ...
    grad_dir = torch.atan2(grad_y, grad_x)

    # 非极大值抑制
    H, W = grad_mag.shape[-2:]
    edges = torch.zeros_like(grad_mag)
    for row in range(1, H - 1):
        for col in range(1, W - 1):
            _dir = grad_dir[0, 0, row, col]
            if (-22.5 <= _dir <= 22.5) or (157.5 <= _dir <= 180) or (-157.5 <= _dir <= -180):
                mag_prev = grad_mag[0, 0, row, col - 1]
                mag_next = grad_mag[0, 0, row, col + 1]
            elif (22.5 < _dir <= 67.5) or (-112.5 <= _dir < -157.5):
                mag_prev = grad_mag[0, 0, row - 1, col + 1]
                mag_next = grad_mag[0, 0, row + 1, col - 1]
            elif (67.5 < _dir <= 112.5) or (-67.5 <= _dir < -22.5):
                mag_prev = grad_mag[0, 0, row - 1, col]
                mag_next = grad_mag[0, 0, row + 1, col]
            else:
                mag_prev = grad_mag[0, 0, row - 1, col - 1]
                mag_next = grad_mag[0, 0, row + 1, col + 1]
            if grad_mag[0, 0, row, col] >= mag_prev and grad_mag[0, 0, row, col] >= mag_next:
                edges[0, 0, row, col] = grad_mag[0, 0, row, col]
    return edges


# 双阈值检测（简化实现）
def double_thresholding(edges, high_threshold, low_threshold):
    # ... 双阈值检测的实现 ...
    # 双阈值处理
    strong_i, weak_i = edges > high_threshold, (edges >= low_threshold) & (edges <= high_threshold)
    edges[weak_i] = 0
    edges[strong_i] = 255

    return edges.squeeze(0).squeeze(0)
