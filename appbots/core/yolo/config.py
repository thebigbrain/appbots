import os

from appbots.core.utils import get_yolo_path, write_lines


def get_yolo(name: str):
    yaml_path = get_yolo_path(name)
    if not os.path.exists(yaml_path):
        os.mkdir(yaml_path)
    return os.path.join(yaml_path, "config.yaml")


def create_yolo(name: str, categories: list[dict]):
    data = [
        f'path: {get_yolo_path(name)}',
        f'train: images',
        f'val: images',
        '',
        'names:',
        *[f"  {cat.get('id') }: {cat.get('name')}" for cat in categories]
    ]

    write_lines(get_yolo(name), data)


if __name__ == "__main__":
    cats = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench"]
    create_yolo("coco128", [{"id": i, "name": x} for i, x in enumerate(cats)])
    create_yolo("coco8", [{"id": i, "name": x} for i, x in enumerate(["person", "cat", "dog"])])

