from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, CocoDetection

from appbots.core.images.transforms import default_transform
from appbots.core.utils import get_path


def build_dataloader_from_images(images_folder, transform=default_transform):
    # 创建数据集
    dataset = ImageFolder(root=get_path(images_folder), transform=transform)

    # 查看数据集的长度和类别
    print(len(dataset))  # 输出数据集中的样本数量
    print(dataset.classes)  # 输出类别名称的列表
    print(dataset.class_to_idx)  # 输出类别名称到索引的映射
    print(dataset.targets)  # 输出所有样本的标签

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    return dataloader


def build_coco_dataloader(ann_file: str, data_dir: str = None, transform=default_transform) -> DataLoader:
    # 创建 COCO 数据集实例
    dataset = CocoDetection(
        root=get_path('assets/coco/data') if data_dir is None else data_dir,
        annFile=get_path(f"assets/coco/{ann_file}.json") if data_dir is None else ann_file,
        transform=transform
    )
    return DataLoader(dataset, batch_size=2, shuffle=True)


if __name__ == '__main__':
    # data_loader = create_dataloader_from_images("assets/classifiers")
    data_loader = build_coco_dataloader(ann_file="example")
    # 遍历数据加载器
    for images, targets in data_loader:
        # images: 形状为[batch_size, 3, H, W]的Tensor
        # targets: 包含图像ID、类别、边界框等信息的字典
        print(images.shape)
        print(targets)
