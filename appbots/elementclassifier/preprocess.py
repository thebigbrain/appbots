import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import numpy as np

from appbots.core.utils import get_path


def color_segmentation(img_path, lower_color, upper_color):
    """
    颜色分割函数

    Args:
        img_path: 图像路径
        lower_color: 颜色下限，BGR格式
        upper_color: 颜色上限，BGR格式

    Returns:
        mask: 分割后的掩码图像
    """

    img = cv2.imread(img_path)
    # 将BGR转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 根据颜色范围创建掩码
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # 形态学操作，去除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_contours(mask):
    """
    查找轮廓

    Args:
        mask: 掩码图像

    Returns:
        contours: 轮廓列表
    """

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


if __name__ == '__main__':
    # 示例用法
    img_path = get_path('assets/test.png')
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = color_segmentation(img_path, lower_blue, upper_blue)
    contours = find_contours(mask)

    # 绘制轮廓
    img = cv2.imread(img_path)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 10)

    plt.imshow(img)
    plt.axis('off')
    plt.show()
