import base64
import hashlib
import os

import cv2
import requests
from torch import Tensor

from appbots.core.images.normalize import normalize_image
from appbots.core.utils import get_cache_dir


def hash_url(url):
    r = base64.urlsafe_b64encode(hashlib.md5(url.encode("utf-8")).digest())
    return str(r.decode("utf-8"))


def download_image(url: str):
    # url = url.replace("\"", "")
    temp_dir = get_cache_dir()
    temp_file = os.path.join(temp_dir, f"{hash_url(url)}.jpg")
    if os.path.exists(temp_file):
        return temp_file

    response = requests.get(url=url, stream=True)
    with open(temp_file, 'wb') as out_file:
        for chunk in response.iter_content(1024):
            out_file.write(chunk)

    return temp_file


def image_to_tensor(image_path, size=None):
    if size is None:
        size = (100, 200)
    # 读取图片
    source_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 转换为Tensor
    img_tensor = normalize_image(source_img, target_size=size)
    return img_tensor


def url_to_tensor(url, size=None) -> Tensor:
    img = download_image(url)
    img_tensor = image_to_tensor(img, size=size)
    return img_tensor


if __name__ == "__main__":
    test_url = "http://192.168.10.115:9000/screenshots/1724855380642.jpg"
    print(f'"{test_url}"'.replace("\"", ""))
    print(hash_url(test_url))
