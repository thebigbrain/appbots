import cv2
import numpy as np
import torch
from PIL import Image


# 定义图像变换
def normalize_image(img, target_size=(100, 200)):
    # 转换为浮点数并归一化
    _img = img.astype(np.float32) / 255.0
    # 调整形状
    _img = cv2.resize(_img, target_size, interpolation=cv2.INTER_AREA)
    # 转换为Tensor
    _tensor_img = torch.from_numpy(_img).permute(2, 0, 1)
    return _tensor_img


def add_alpha_channel(img):
    """
    为图像添加alpha通道，并填充为255

    Args:
        img: 输入图像，numpy数组

    Returns:
        添加alpha通道后的图像
    """

    if img.shape[-1] == 3:
        alpha_channel = np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8) * 255
        img = np.concatenate([img, alpha_channel], axis=-1)
    return img


if __name__ == "__main__":
    # 读取图像
    source_img = cv2.imread("../../../assets/test.png", cv2.IMREAD_UNCHANGED)
    tensor_img = normalize_image(source_img)

    print(tensor_img.shape)

    img_np = tensor_img.numpy()
    img_np = img_np * 255

    img_np = np.transpose(img_np, (1, 2, 0))

    print(img_np.shape)

    img_pil = Image.fromarray(img_np.astype(np.uint8))

    img_pil.save('../../tmp/output_image.png')
