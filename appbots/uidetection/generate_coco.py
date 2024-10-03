import torch

from appbots.core.coco import CocoAnnotationUtil
from appbots.core.coco import CocoCategoryUtil
from appbots.core.coco import CocoImageUtil
from appbots.core.coco import CocoConfig
from appbots.uidetection.transforms import xywh_to_cxcywh, cxcywh_to_xywh


def transform_annotation(ann, w, h = 1.0):
    box = torch.tensor(ann['bbox'])
    box = xywh_to_cxcywh(box)
    box = cxcywh_to_xywh(box)
    x, y, _w, _h = box
    box = torch.tensor([x + w / 2, h / 2 - y, _w, _h], dtype=torch.float)
    ann['bbox'] = box.tolist()
    return ann


def generate_coco_file():
    annotations = []
    images_dict = {}

    CocoCategoryUtil.load_categories()

    for annotation in CocoAnnotationUtil.load_coco_annotations():
        image_id = annotation.get('image_id')
        img = CocoImageUtil.get(image_id)
        if img is None:
            continue
        images_dict[image_id] = img
        annotation = transform_annotation(annotation, w=img['width'], h=img['height'])
        annotations.append(annotation)

    images = list(images_dict.values())

    CocoConfig.save({
        "images": images,
        "annotations": annotations,
        "categories": CocoCategoryUtil.categories
    })


if __name__ == "__main__":
    generate_coco_file()
