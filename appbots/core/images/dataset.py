from typing import Any

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder, CocoDetection

from appbots.core.images.transforms import default_transform
from appbots.core.utils import get_path


def build_dataset_from_images(images_folder, transform=default_transform):
    # 创建数据集
    dataset = ImageFolder(root=get_path(images_folder), transform=transform)

    # 查看数据集的长度和类别
    print(len(dataset))  # 输出数据集中的样本数量
    print(dataset.classes)  # 输出类别名称的列表
    print(dataset.class_to_idx)  # 输出类别名称到索引的映射
    print(dataset.targets)  # 输出所有样本的标签

    return dataset


def build_coco_dataset(
        ann_file: str,
        data_dir: str = None,
        transforms=None
) -> Dataset:
    # 创建 COCO 数据集实例
    coco_dataset = CocoDetection(
        root=get_path('assets/coco/data') if data_dir is None else data_dir,
        annFile=get_path(f"assets/coco/{ann_file}.json") if data_dir is None else ann_file,
        transforms=transforms
    )
    return coco_dataset


def rcnn_collate(data: list[Any]):
    _images = []
    _targets = []
    for item in data:
        image, target = item
        _images.append(image)
        _targets.append(target)
    return _images, _targets


if __name__ == '__main__':
    # data_loader = create_dataloader_from_images("assets/classifiers")
    ds = build_coco_dataset(ann_file="example")
    # 遍历数据加载器
    for images, targets in ds:
        # images: 形状为[batch_size, 3, H, W]的Tensor
        # targets: 包含图像ID、类别、边界框等信息的字典
        print(images)
        print(targets)
