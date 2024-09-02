import torch
from torchvision import transforms
from torch.utils.data import Dataset

transform_noise = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class UIDataset(Dataset):
    def __init__(self, root, transform=None):
        # ... 初始化代码，读取标注文件等
        self.transform = transform

    def __getitem__(self, idx):
        # ... 读取图像和对应的标注信息
        image = ...
        target = ...
        if self.transform:
            image = self.transform(image)
        return image, target

    def __len__(self):
        # ... 返回数据集的样本数量
        return 0
