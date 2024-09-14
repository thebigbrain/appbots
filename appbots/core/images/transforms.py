import torch
from torchvision.transforms import v2

normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 定义数据变换
default_transform = v2.Compose([
    v2.Resize(size=150),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    normalize,
])


def denormalize(normalized_img):
    original_img = normalized_img * torch.tensor(normalize.std).view(3, 1, 1) \
                    + torch.tensor(normalize.mean).view(3, 1, 1)
    return original_img


def build_transform(size=(300, 150)) -> v2.Compose:
    return v2.Compose([
        v2.Resize(size=size),
        v2.RGB(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])
