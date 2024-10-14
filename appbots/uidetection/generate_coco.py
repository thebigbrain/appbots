from appbots.core.coco import CocoAnnotationUtil
from appbots.core.coco import CocoCategoryUtil
from appbots.core.coco import CocoImageUtil
from appbots.core.coco import CocoConfig


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
        annotations.append(annotation)

    images = list(images_dict.values())

    CocoConfig.save({
        "images": images,
        "annotations": annotations,
        "categories": CocoCategoryUtil.categories
    })


if __name__ == "__main__":
    generate_coco_file()
