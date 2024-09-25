import numpy as np
import torch
from torch import nn


# 定义高斯核
def gaussian_kernel(size, sigma=1.0):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return torch.from_numpy(g).float()


class GaussianConv2d(nn.Conv2d):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, sigma=1.0):
        super(GaussianConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False
        )

        # 使用高斯核进行卷积
        kernel = gaussian_kernel(kernel_size, sigma)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # 扩展维度以适应卷积操作
        self.weight = nn.Parameter(data=kernel)
        self.weight.requires_grad = False  # 不需要梯度更新
