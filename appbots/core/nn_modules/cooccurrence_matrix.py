import numpy as np
import torch
import torch.nn as nn
import cv2
from matplotlib import pyplot as plt

from appbots.core.utils import get_path


class CooccurrenceMatrix(nn.Module):
    def __init__(self, levels=256):
        super(CooccurrenceMatrix, self).__init__()

        self.levels = levels

    def forward(self, x):
        """
        计算图像的共现矩阵

        Args:
            x: 输入图像，形状为[batch_size, channels * height * width]

        Returns:
            cooccurrence_matrices: 一个列表，每个元素是一个batch_size的共现矩阵的堆叠
        """
        batch_size, _ = x.shape
        levels = self.levels

        cooccurrence_matrices = []
        cooccurrence_matrix = torch.zeros(batch_size, levels, levels)
        for i in range(batch_size):
            _img = x[i, :].numpy().astype(np.uint8)  # 假设输入为单通道灰度图
            cooccurrence_matrix[i] = cv2.calcHist(
                [_img], [0], None, [levels],
                [0, levels])
        cooccurrence_matrices.append(cooccurrence_matrix)
        return cooccurrence_matrices


if __name__ == "__main__":
    img = cv2.imread(get_path("assets/test.png"), 0)
    # plt.imshow(img)

    levels = 25

    hist = cv2.calcHist(
        [img], [0], None, [levels],
        [0, levels])

    print(hist.shape)

    # 显示直方图
    plt.plot(hist)
    plt.xlabel('Bins')
    plt.ylabel('Number of Pixels')
    plt.title('Histogram')

    plt.show()
