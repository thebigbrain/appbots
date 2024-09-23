import torch
import torch.nn.functional as F

from appbots.core.images.transforms import CocoTransform, build_transform


def xywh_to_cxcywh(boxes: torch.Tensor):
    x, y, w, h = boxes
    cx = x + w / 2
    cy = y + h / 2
    return torch.tensor([cx, cy, abs(w), abs(h)], dtype=torch.float)


def center_to_min_max(center: torch.Tensor):
    cx, cy, w, h = center
    hw = w / 2
    hh = h / 2

    min_x = cx - hw
    min_y = cy - hh
    max_x = cx + hw
    max_y = cy + hh
    return torch.tensor([min_x, min_y, max_x, max_y], dtype=torch.float)


def xywh_to_min_max(boxes: torch.Tensor):
    center = xywh_to_cxcywh(boxes)
    return center_to_min_max(center)


class CocoTransforms(CocoTransform):
    def __init__(self, categories, scale=1000):
        self.categories = categories
        self.scale = scale
        self.transform = build_transform(auto_normalize=False)

    @property
    def num_classes(self):
        return len(self.categories)

    def __call__(self, image, target):
        boxes = torch.stack([self.transform_box(item) for item in target])
        labels = torch.tensor([item['category_id'] for item in target])

        image = self.transform(image)
        target = {
            "boxes": boxes,
            "labels": labels
        }
        return image, target

    def transform_box(self, target):
        bbox = torch.tensor(target['bbox'], dtype=torch.float) * self.scale
        return xywh_to_min_max(bbox)


if __name__ == '__main__':
    classes = torch.LongTensor(0)
    # one_hot = torch.nn.functional.one_hot(classes, num_classes=3)
    print(classes)
