import json
import os
from pathlib import Path
from typing import TypedDict

from appbots.core.coco.annotation import CocoAnnotation
from appbots.core.coco.category import CocoCategory
from appbots.core.coco.image import CocoImage

from appbots.core.utils import get_cache_dir, get_coco_dir, mkdir


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
        mkdir(get_coco_dir())
        with open(cls.ann_file, 'w') as af:
            json.dump(data, af, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls):
        with open(cls.ann_file, 'r') as f:
            cls.coco = json.load(f)

        return cls.coco

    @classmethod
    def get_images(cls):
        return cls.coco['images']

    @classmethod
    def get_categories(cls):
        return cls.coco['categories']

    @classmethod
    def get_categories_dict(cls):
        return {cat['id']: cat for cat in cls.get_categories()}

    @classmethod
    def get_category(cls, cat_id: str):
        return next(filter(lambda x: x['id'] == cat_id, cls.get_categories()))


if __name__ == "__main__":
    pass
