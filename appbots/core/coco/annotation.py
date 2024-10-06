import json
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

    @classmethod
    def create_bboxes_annotations(cls, image_id, bboxes):
        coco_annotations_table = get_table('coco_annotations')
        rows = []
        for b in bboxes:
            rows.append(dict(image_id=image_id, bbox=json.dumps(b)))
        coco_annotations_table.insert_many(rows)

    @classmethod
    def save_bbox_annotation(
            cls,
            image_id: int,
            bbox: list[float],
            coco_annotation_id: int = None
    ):
        coco_annotations_table = get_table('coco_annotations')
        data = {
            "image_id": image_id,
            "bbox": json.dumps(bbox),
        }
        if coco_annotation_id:
            data['id'] = coco_annotation_id
            coco_annotations_table.update(data, ['id'])
        else:
            coco_annotations_table.insert(data)

    @classmethod
    def set_bboxes_generated(cls, image_id):
        get_table('bot_memories').update({'bboxes_generated': True, "id": image_id}, ['id'])


if __name__ == '__main__':
    CocoAnnotationUtil.save_bbox_annotation(1, [1, 2, 3, 4])
    # CocoAnnotationUtil.set_bboxes_generated(2101)
