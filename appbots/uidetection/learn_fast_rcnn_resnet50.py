import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from appbots.coco.config import CocoConfig
from appbots.core.images.builder import ImageTensorBuilder
from appbots.core.images.dataset import build_coco_dataset, rcnn_collate
from appbots.core.images.transforms import denormalize
from appbots.core.model import ModelCarer, DictLossNormalizer
from appbots.uidetection.transforms import CocoTransforms

from appbots.uidetection.utils import show_bounding_boxes

CocoConfig.load()

categories = CocoConfig.get_categories()
scale = 1000

coco_transforms = CocoTransforms(categories=categories, scale=scale)
num_classes = len(categories)

# 定义模型
carer = ModelCarer(
    model=fasterrcnn_resnet50_fpn_v2(num_classes=num_classes),
    name="faster_rcnn_resnet50_v2",
    num_epochs=15
)
carer.load()


def train():
    # 数据准备
    dataset = build_coco_dataset(
        ann_file=CocoConfig.ann_file,
        data_dir=CocoConfig.coco_data_dir,
        transforms=coco_transforms
    )

    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=rcnn_collate)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(carer.model.parameters(), lr=0.005, momentum=0.9)
    dict_loss_normalizer = DictLossNormalizer()
    losses = torch.tensor(1.0, requires_grad=True)

    # 训练循环
    for epoch in range(carer.num_epochs):
        for images, targets in data_loader:
            loss_dict = carer.model(images, targets)

            dict_loss_normalizer.init_loss_dict(loss_dict)

            loss, normed_loss_dict = dict_loss_normalizer.ave(loss_dict)

            carer.save_loss(epoch, loss)

            print(loss, normed_loss_dict)

            losses = torch.tensor(loss, requires_grad=True)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch} average normalized loss: {losses}")
        carer.save_epoch(epoch)

    # 保存模型
    carer.done()


def predict(test_img_path: str):
    carer.model.eval()

    if not carer.is_model_exists():
        return

    # 预测
    with torch.no_grad():
        builder = ImageTensorBuilder()
        builder.load_from_path(test_img_path)
        image = builder.tensor.unsqueeze(0)
        predictions = carer.model(image)

        prediction = predictions[0]
        boxes = prediction['boxes']

        print(prediction)
        show_bounding_boxes(denormalize(builder.tensor), boxes)


if __name__ == "__main__":
    # train()
    # predict("assets/bus.jpg")
    # predict("assets/test.png")
    predict("assets/test2.jpg")
