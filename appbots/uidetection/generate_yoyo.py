from appbots.core.coco import CocoConfig
from appbots.core.utils import get_coco_dir
from appbots.core.yolo.coco2json import convert_coco_json
from appbots.core.yolo.config import create_yolo


def generate_yolo():
    name = 'coco'
    coco = CocoConfig.load()
    cats = coco.get("categories")
    create_yolo(name, cats)

    convert_coco_json(
        get_coco_dir(),  # directory with *.json
    )


if __name__ == "__main__":
    generate_yolo()
