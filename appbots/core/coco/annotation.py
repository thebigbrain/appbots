from typing import TypedDict, OrderedDict, TypeAlias

from appbots.core.coco.utils import load_json
from appbots.core.db import get_table

CocoSegmentation: TypeAlias = list[list[float]]


class CocoAnnotation(TypedDict):
    id: int
    image_id: int
    category_id: int
    segmentation: CocoSegmentation
    area: float
    bbox: list[float]
    iscrowd: int


class CocoAnnotationUtil:
    @classmethod
    def load_coco_annotations(cls):
        coco_annotations_table = get_table('coco_annotations')
        iters = coco_annotations_table.find(_limit=100, order_by=['-id'])
        annotations = []
        for item in iters:
            annotations.append(cls.build_coco_annotation(item))
        return annotations

    @classmethod
    def build_coco_annotation(cls, data: OrderedDict):
        segmentation = data.get('segmentation')
        bbox = data.get('bbox')

        coco_annotation: CocoAnnotation = {k: v for k, v in data.items() if k in CocoAnnotation.__annotations__}
        coco_annotation['segmentation'] = load_json(segmentation, [])
        coco_annotation['bbox'] = load_json(bbox, [])
        return coco_annotation

