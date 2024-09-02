import numpy as np
import torch
from PIL import Image


def generate_sample_image():
    # 创建RGB通道数据
    height, width, channels = 200, 100, 4
    rgb_data = np.random.rand(height, width, channels)

    # 创建alpha通道数据
    # alpha_channel = np.ones((height, width, 1))

    # 合并RGB和alpha通道
    # rgba_image = np.concatenate([rgb_data, alpha_channel], axis=-1)

    # return rgba_image

    return rgb_data


if __name__ == "__main__":
    i = generate_sample_image()
    # print(i.shape)
    # img = Image.fromarray((i * 255).astype(np.uint8))
    # img.save('../../tmp/sample.png')

    s = torch.from_numpy(i).permute(2, 0, 1)
    t = torch.tensor([s], dtype=torch.float)
    print(t)

