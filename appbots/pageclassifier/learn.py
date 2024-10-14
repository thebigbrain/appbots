import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from appbots.core.model import load_model, save_model
from appbots.core.utils import get_model_path
from appbots.pageclassifier.dataset import UiClassifierDataset
from appbots.pageclassifier.model import get_model, MODEL_NAME


def train():
    num_epochs = 5

    # 数据加载器
    train_dataset: UiClassifierDataset = UiClassifierDataset()  # 你的训练数据集
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 模型定义
    model = get_model()
    load_model(model, get_model_path(MODEL_NAME))

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练循环
    for epoch in range(num_epochs):
        print(f"start epoch {epoch} ...")
        for images, labels in train_loader:
            # 前向传播
            outputs = model(images)
            # labels.clone().detach()
            targets = torch.tensor(labels, dtype=torch.float)
            loss = criterion(outputs, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"loss: {loss.item()}")

    save_model(model, get_model_path(MODEL_NAME))


if __name__ == '__main__':
    train()
