import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from appbots.core.coco import CocoConfig
from appbots.core.images.builder import read_image
from appbots.core.images.transforms import gray_transform
from appbots.core.model import Trainer
from appbots.core.plots.plot import plot_images


def transform(x, size):
    _transform = v2.Compose([
        v2.RGB(),
        v2.Resize(size=size),
        v2.ToImage(),
    ])
    return _transform(x)


class FcnDataset(Dataset):
    data = []
    num_classes = 1
    _images_map = dict()

    def __init__(self, size=640):
        self.size = size

        self._load()

    def _load(self):
        coco: dict
        with open(CocoConfig.ann_file, 'r') as f:
            coco = json.load(f)
        coco_images = coco['images']
        coco_annotations = coco['annotations']
        coco_cats = coco['categories']

        self.num_classes = len(coco_cats)

        for item in coco_images:
            self._images_map[item['id']] = item

        for item in coco_annotations:
            _image_item = self._images_map[item['image_id']] or dict()
            if _image_item.get('annotations') is None:
                _image_item['annotations'] = []
            _image_item['annotations'].append(item)

        self.data = []
        for item in self._images_map.values():
            if item.get('annotations') is not None:
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item: dict = self.data[index]
        annotations = item.get('annotations', [])
        source_image = read_image(item.get('file_name'))

        h, w = item['height'] * self.size, item['width'] * self.size
        h = int(h)
        w = int(w)

        source = transform(source_image, size=(h, w)) / 255.0

        target = torch.zeros((h, w), dtype=torch.long)
        for a in annotations:
            cat_id = int(a.get('category_id'))
            bbox = np.array(a.get('bbox')) * h

            x, y, w, h = bbox
            # 计算框的坐标
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            target[y1:y2, x1:x2] = cat_id

        return source, target


class FCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(FCN, self).__init__()
        # 将最后一个池化层改为卷积层
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(96, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        out = self.classifier(x)
        out = v2.Resize(size=x.shape[2:])(out)
        return out


dataset = FcnDataset()
train_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
model = FCN(num_classes=dataset.num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
trainer = Trainer(model, "fcn-classifier",
                  train_loader=train_loader,
                  num_epochs=10,
                  criterion=criterion, optimizer=optimizer)

if __name__ == '__main__':
    # trainer.reset()
    # trainer.train()

    s, t = FcnDataset()[3]
    o = trainer.predict(s.unsqueeze(0))

    resized = v2.Resize(30)(s)

    softmax_output = nn.Softmax(dim=1)(o)
    _, predicted = torch.max(softmax_output, dim=1)

    print(s.shape, resized.shape, t.shape, predicted.shape)

    plot_images([s, resized, t.unsqueeze(0), predicted])
