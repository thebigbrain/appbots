from os import path
from pathlib import Path
from typing import Union

import numpy as np
import torch

from appbots.core.coco import CocoConfig
from appbots.core.coco.transforms import cxcywh_to_xywh
from appbots.core.images.builder import ImageTensorBuilder
from appbots.core.images.transforms import gray_transform
from appbots.core.plots.plot import add_boxes, plot_images
from appbots.core.utils import get_coco_dir, get_yolo_path
from appbots.core.yolo.coco2json import convert_coco_json
from appbots.core.yolo.config import create_yolo


def convert2float(arr: list) -> list[float]:
    return [float(x) for x in arr]


def read_yolo_labels(label_file: str) -> list[torch.Tensor]:
    if path.exists(label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            cat_boxes = [x.strip().split() for x in lines]
            return [torch.tensor(convert2float(x[1:]), dtype=torch.float) for x in cat_boxes]
    return []


def convert_yolo2coco_box(box: Union[torch.Tensor, list[float]], w: int, h: int) -> list[int]:
    box = np.array(box, dtype=np.float64)
    box[[0, 2]] *= w  # normalize x
    box[[1, 3]] *= h  # normalize y

    box = box.astype(np.int32)

    return cxcywh_to_xywh(box.tolist())


def generate_yolo():
    name = 'coco'
    coco = CocoConfig.load()
    cats = coco.get("categories")
    create_yolo(name, cats)

    convert_coco_json(
        get_coco_dir(),  # directory with *.json
    )


def validate_if_generate_ok():
    example_file = get_yolo_path("coco/images/NVwjkxCrXOc8owynQUd1Cw==.jpg")
    label_file = get_yolo_path(f"coco/labels/{Path(example_file).stem}.txt")

    print(example_file, label_file)

    builder = ImageTensorBuilder()
    builder.load_from_path(example_file)

    c, h, w = builder.tensor.shape

    # box = [
    #     0.03031547524420186,
    #     0.08023952095808384,
    #     0.1221556886227545,
    #     0.07065868263473052
    # ]
    #
    # print(box, coco_box2yolo_label(box, w/h, 1.0))

    boxes = read_yolo_labels(label_file)
    # print(boxes)
    boxes = list(map(lambda b: torch.tensor((convert_yolo2coco_box(b, w, h)), dtype=torch.int), boxes))
    # print(builder.tensor.shape, boxes)

    # boxes.append(torch.tensor(np.array(box) * h, dtype=torch.int))

    t = gray_transform(builder.tensor)

    add_boxes(t, bound_boxes=boxes)
    plot_images([builder.tensor, t])


if __name__ == "__main__":
    # generate_yolo()
    validate_if_generate_ok()
