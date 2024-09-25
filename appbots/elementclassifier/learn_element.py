import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from appbots.core.images.transforms import build_transform
from appbots.core.model import ModelCarer
from appbots.core.utils import get_cache_dir
from appbots.elementclassifier.model import ElementClassifier


# 创建数据集
train_dataset = torchvision.datasets.ImageFolder(
    root=get_cache_dir(),
    transform=build_transform(auto_normalize=False)
)
data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=4,
    shuffle=True,
)

# 创建模型、优化器等
carer = ModelCarer(
    model=ElementClassifier(num_classes=10),
    name="element_classifier",
    num_epochs=15
)
carer.load()
model = carer.model
num_epochs = carer.num_epochs

criterion = nn.CrossEntropyLoss()
bbox_loss = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters())


def train():
    # 训练模型
    for epoch in range(num_epochs):
        # ... 训练过程
        for images in data_loader:
            outputs = model(images)
            loss = criterion(outputs, labels) + bbox_loss(predicted_bboxes, target_bboxes)
            carer.save_loss(epoch, loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    carer.done()


def evaluate():
    # 测试模型
    # ...
    pass