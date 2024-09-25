import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import v2


def plot_image(img: torch.Tensor):
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.show()


def plot_images(images: list[torch.Tensor]):
    plt.figure(figsize=(16, 16))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.axis('off')
    plt.show()


def plot_surface(img: torch.Tensor):
    transform = v2.Resize((200, 100))
    img = transform(img)
    img = img.squeeze(0).numpy()

    # 创建图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    X, Y = np.meshgrid(x, y)

    # 绘制曲面
    surf = ax.plot_surface(X, Y, img, cmap='viridis')
    # 设置标题等
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Image Surface Plot')

    plt.show()


def plot_boxes(img: torch.Tensor, boxes: list[list[torch.Tensor]]):
    # 绘制边界框
    plt.imshow(img.permute(1, 2, 0).numpy())
    for box in boxes:
        x, y, w, h = box
        rect = plt.Rectangle((x, y), w, h, fill=False, color='red')
        plt.gca().add_patch(rect)
    plt.show()
