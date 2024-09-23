import json
import os
from typing import TypedDict

from appbots.coco.annotation import CocoAnnotation
from appbots.coco.category import CocoCategory
from appbots.coco.image import CocoImage

from appbots.core.utils import get_cache_dir, get_coco_dir


class Coco(TypedDict):
    annotations: list[CocoAnnotation]
    categories: list[CocoCategory]
    images: list[CocoImage]


class CocoConfig:
    ann_file = os.path.join(get_coco_dir(), "coco.json")
    coco_data_dir = get_cache_dir()
    coco: Coco = None

    @classmethod
    def save(cls, data):
        with open(cls.ann_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls):
        with open(cls.ann_file, 'r') as f:
            cls.coco = json.load(f)

        return cls.coco

    @classmethod
    def get_categories(cls):
        return cls.coco['categories']


if __name__ == "__main__":
    pass
