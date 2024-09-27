import os
from pycocotools.coco import COCO
import yaml

from appbots.core.utils import get_yolo_path
from appbots.core.yolo.config import create_yolo


def generate_yolo():
    create_yolo("coco", [])


if __name__ == "__main__":
    generate_yolo()
