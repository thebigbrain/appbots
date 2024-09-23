import base64
import hashlib
import os

import requests
from PIL import Image
from torch import Tensor
from torchvision import transforms

from appbots.core.images.transforms import build_transform, denormalize
from appbots.core.utils import get_cache_dir, get_path


def hash_url(url):
    r = base64.urlsafe_b64encode(hashlib.md5(url.encode("utf-8")).digest())
    return str(r.decode("utf-8"))


def download_image(url: str):
    temp_dir = get_cache_dir()
    temp_file = os.path.join(temp_dir, f"{hash_url(url)}.jpg")
    if os.path.exists(temp_file):
        return temp_file

    response = requests.get(url=url, stream=True)
    with open(temp_file, 'wb') as out_file:
        for chunk in response.iter_content(1024):
            out_file.write(chunk)

    return temp_file


def image_to_tensor(image_path, size=None) -> (Tensor, Image.Image):
    source_img = Image.open(image_path).convert("RGB")
    w, h = source_img.size
    transform = build_transform(size=[h, w] if size is None else size)
    return transform(source_img), source_img


def url_to_tensor(url, size=None) -> (Tensor, Image.Image):
    img = download_image(url)
    img_tensor, source_img = image_to_tensor(img, size=size)
    return img_tensor, source_img


def read_image(image_path, size=None) -> (Tensor, Image.Image):
    return image_to_tensor(get_path(image_path), size=size)


def to_pil_image(img_tensor: Tensor) -> Image.Image:
    original_img = denormalize(img_tensor)
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(original_img)
    return img_pil


class ImageTensorBuilder:
    tensor: Tensor
    source: Image.Image
    image_path: str

    def to_pil(self) -> Image.Image:
        return to_pil_image(self.tensor)

    def load_from_path(self, image_path: str, size=None):
        self.image_path = image_path
        self.tensor, self.source = read_image(image_path, size=size)

    def load_from_url(self, url: str, size=None):
        self.image_path = download_image(url)
        self.tensor, self.source = image_to_tensor(self.image_path, size=size)


if __name__ == "__main__":
    # test_url = "http://192.168.10.115:9000/screenshots/1724855380642.jpg"
    # print(hash_url(test_url))

    image_size = (200, 100)
    t, _ = read_image("assets/test.png", size=image_size)
    pil_image = to_pil_image(t)
    pil_image.show()
