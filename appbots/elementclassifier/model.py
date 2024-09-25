import cv2
import torch
import torch.nn as nn

from appbots.core.images.builder import get_image_from_path
from appbots.core.nn_modules.cooccurrence_matrix import CooccurrenceMatrix
from appbots.core.utils import get_path


# 定义网络结构
class ElementClassifier(nn.Module):

    def __init__(self, num_classes, in_channels=3):
        super(ElementClassifier, self).__init__()

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=in_channels, kernel_size=1)
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            CooccurrenceMatrix(),
        )

        # 将combined_features输入到全连接层或其他网络层
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        # 添加注意力机制
        attention_map = torch.sigmoid(self.attention(x))
        x = x * attention_map

        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    img, _ = get_image_from_path(get_path("assets/test.png"))
    model = ElementClassifier(num_classes=10)
    model(img.squeeze(0))
