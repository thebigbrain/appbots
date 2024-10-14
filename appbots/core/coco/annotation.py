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
        iters = cls.load_annotations_raw()
        annotations = []
        for item in iters:
            annotations += cls.build_coco_annotations(item)
        return annotations

    @classmethod
    def load_annotations_raw(cls):
        coco_annotations_table = get_table('coco_annotations')
        annotations_iter = coco_annotations_table.find(_limit=100, order_by=['-id'])
        return list(annotations_iter)

    @classmethod
    def build_coco_annotations(cls, data: OrderedDict):
        annotations: list[CocoAnnotation] = load_json(data.get('annotations'), [])
        return annotations

    @classmethod
    def update_bboxes_annotations(cls, image_id, bboxes):
        coco_annotations_table = get_table('coco_annotations')
        coco_anno_id_table = get_table("coco_annotation_ids")
        rows = []
        for b in bboxes:
            _id = coco_anno_id_table.insert({})
            rows.append(dict(image_id=image_id, bbox=b, id=_id))
        coco_annotations_table.insert(dict(id=image_id, annotations=json.dumps(rows)))

    @classmethod
    def set_bboxes_generated(cls, image_id):
        get_table('bot_memories').update({'bboxes_generated': True, "id": image_id}, ['id'])


if __name__ == '__main__':
    # CocoAnnotationUtil.set_bboxes_generated(2101)
    pass
