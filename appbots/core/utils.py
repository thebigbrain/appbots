from os import path


def get_root_dir():
    return path.dirname(path.abspath(path.join(__file__, "../../")))


def get_cache_dir():
    return path.abspath(path.join(get_root_dir(), ".cache"))


def get_path(path_str: str):
    return path.abspath(path.join(get_root_dir(), path_str))


if __name__ == "__main__":
    print(get_root_dir())
    print(get_path("assets"))
