import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import detection

from appbots.ui_detection.dataset import UIDataset

num_epochs = 5
num_classes = 10

# 数据加载器
train_dataset: UIDataset = ...  # 你的训练数据集
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 模型定义
model = detection.ssd300_vgg16(num_classes=21)  # 21为类别数+背景类
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练循环
for epoch in range(num_epochs):
    for images, targets in train_loader:
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    pass
