import os

from appbots.core.utils import get_yolo_path, write_lines


def get_yolo(name: str):
    yaml_path = get_yolo_path(name)
    if not os.path.exists(yaml_path):
        os.mkdir(yaml_path)
    return os.path.join(yaml_path, "config.yaml")


def create_yolo(name: str, categories: list[str]):
    data = [
        f'path: ../{name}',
        'train: ../train.txt',
        'val: ../val.txt',
        '',
        'names:',
        *[f"  {i}: {cat}" for i, cat in enumerate(categories)]
    ]

    write_lines(get_yolo(name), data)


if __name__ == "__main__":
    cats = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench"]
    create_yolo("coco128", cats)
    create_yolo("coco8", ["person", "cat", "dog"])

