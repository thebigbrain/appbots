from os import path


def get_root_dir():
    return path.dirname(path.abspath(path.join(__file__, "../../")))


def get_aide_dir():
    return path.abspath(path.join(get_root_dir(), ".aide"))


def get_cache_dir():
    return path.abspath(path.join(get_aide_dir(), ".cache"))


def get_models_dir():
    return path.abspath(path.join(get_aide_dir(), ".models"))


def get_coco_dir():
    return path.abspath(path.join(get_aide_dir(), ".coco"))


def get_path(path_str: str):
    return path.abspath(path.join(get_root_dir(), path_str))


if __name__ == "__main__":
    print(get_root_dir())
    print(get_path("assets"))
