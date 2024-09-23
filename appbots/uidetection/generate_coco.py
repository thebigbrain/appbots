from appbots.coco.annotation import CocoAnnotationUtil
from appbots.coco.category import CocoCategoryUtil
from appbots.coco.image import CocoImageUtil
from appbots.coco.config import CocoConfig


def generate_coco_file():
    annotations = []
    images_dict = {}

    CocoCategoryUtil.load_categories()

    for annotation in CocoAnnotationUtil.load_coco_annotations():
        image_id = annotation['image_id']
        images_dict[image_id] = CocoImageUtil.get(image_id)
        annotations.append(annotation)

    images = list(images_dict.values())

    CocoConfig.save({
        "images": images,
        "annotations": annotations,
        "categories": CocoCategoryUtil.categories
    })


if __name__ == "__main__":
    generate_coco_file()
