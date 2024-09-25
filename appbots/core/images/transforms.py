import torch
from torchvision.transforms import v2

normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def denormalize(normalized_img: torch.Tensor):
    original_img = normalized_img * torch.tensor(normalize.std).view(3, 1, 1) \
                   + torch.tensor(normalize.mean).view(3, 1, 1)
    return original_img


class CocoTransform:
    def __call__(self, images, targets) -> (torch.Tensor, torch.Tensor):
        pass


def build_transform(width=100, height=200, auto_normalize=True, resize=True) -> v2.Compose:
    _transforms = [
        v2.RGB(),
    ]

    if resize:
        _transforms.append(v2.Resize(size=(height, width)),)

    _transforms.append(v2.ToImage())

    if auto_normalize:
        _transforms.append(v2.ToDtype(torch.float32, scale=True))
        _transforms.append(normalize)

    return v2.Compose(_transforms)


default_transform = build_transform()

gray_transform = v2.Compose(
    [
        v2.Grayscale(),
        v2.ToImage()
    ]
)
