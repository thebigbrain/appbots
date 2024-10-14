import cv2
import torch

from appbots.core.images.builder import ImageTensorBuilder
from appbots.core.images.transforms import gray_transform
from appbots.core.model import Trainer
from appbots.core.nn_modules.unet import UNet
from appbots.core.plots.plot import plot_images, add_boxes
from appbots.core.utils import get_assets

model = UNet(in_channels=3, out_channels=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(model=model,
                  name="color-segmentation",
                  train_loader=None,
                  num_epochs=10,
                  optimizer=optimizer)


def find_min_bounding_box(gray: torch.Tensor, pixel_x: int, pixel_y: int):
    # 将图像转换为灰度图（假设根据灰度值进行分割）

    # 设置阈值（这里使用Otsu算法）
    thresh, img_bw = cv2.threshold(gray.permute(1, 2, 0).numpy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 获取连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bw, connectivity=8)

    # 找到目标像素点所在的连通区域
    label = labels[pixel_y, pixel_x]

    # 计算最小包围盒
    return stats[label, :4]


if __name__ == '__main__':
    # trainer.train()

    # 读取图像
    builder = ImageTensorBuilder()
    builder.load_from_path(get_assets("test.png"))
    t = builder.tensor

    gray = gray_transform(t)

    # 寻找最小包围盒
    bbox = find_min_bounding_box(gray, 100, 300)
    print(gray.shape, bbox)
    add_boxes(gray, [bbox],)

    plot_images([t, gray])

