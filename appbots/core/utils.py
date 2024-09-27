import os
from os import path
from pathlib import Path


def get_root_dir():
    return path.dirname(path.abspath(path.join(__file__, "../../")))


def get_aide_dir():
    return path.abspath(path.join(get_root_dir(), ".aide"))


def get_cache_dir():
    return path.join(get_aide_dir(), ".cache")


def get_models_dir():
    return path.join(get_aide_dir(), ".models")


def get_coco_dir():
    return path.join(get_aide_dir(), ".coco")


def get_path(path_str: str):
    return path.join(get_root_dir(), path_str)


def get_model_path(name: str):
    return path.join(get_models_dir(), name)


def get_assets(name: str):
    return get_path(f"assets/{name}")


def get_coco_path(name: str):
    return get_path(path.join(get_coco_dir(), name))


def get_yolo_dir():
    return path.join(get_aide_dir(), ".yolo")


def get_yolo_path(name: str):
    return path.join(get_yolo_dir(), name)


def mkdir(dir_str: str):
    _dir = os.path.dirname(dir_str)
    Path(_dir).mkdir(exist_ok=True)


def write_lines(file_path: str, data: list[str]):
    mkdir(file_path)
    with open(file_path, 'w') as f:
        print('write to file', file_path)
        for line in data:  # data是一个列表，每个元素代表一行
            f.write(line + '\n')


if __name__ == "__main__":
    print(get_root_dir())
    print(get_path("assets"))
