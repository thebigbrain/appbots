import os.path

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from appbots.core.images.dataset import build_coco_dataloader
from appbots.core.images.utils import read_image
from appbots.core.utils import get_cache_dir, get_root_dir

# 数据准备
coco_data_dir = get_cache_dir()
ann_file = os.path.join(get_root_dir(), ".coco/coco.json")
data_loader = build_coco_dataloader(ann_file, data_dir=coco_data_dir)

# 定义模型
model = FasterRCNN(backbone=torchvision.models.resnet50(pretrained=True),
                   num_classes=2)  # 假设有2个类别
# 修改锚框生成器
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
model.rpn.anchor_generator = anchor_generator

# 定义损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

losses = torch.tensor(0.0)
# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = torch.tensor(sum(loss for loss in loss_dict.values()))
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} loss: {losses}")

# 保存模型
# save_model(model, name="faster_rcnn")

# 预测
with torch.no_grad():
    image = read_image("assets/test.png")
    image.unsqueeze(0)
    prediction = model(image)
    print(prediction)