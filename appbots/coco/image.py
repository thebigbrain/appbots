from typing import TypedDict, OrderedDict

from appbots.core.images.builder import ImageTensorBuilder
from appbots.datasets.annotation import get_bot_memo


class CocoImage(TypedDict):
    id: int
    width: int
    height: int
    file_name: str


class CocoImageUtil:
    @classmethod
    def get(cls, image_id) -> CocoImage:
        builder = ImageTensorBuilder()
        bm = get_bot_memo(image_id)
        builder.load_from_url(bm.get('screenshot'))

        (width, height) = builder.source.size

        return cls.build_coco_image({
            "id": bm.get('id'),
            "width": width / height,
            "height": 1.0,
            "file_name": builder.image_path
        })

    @classmethod
    def build_coco_image(cls, data: dict) -> CocoImage:
        coco_image: CocoImage = {k: v for k, v in data.items() if k in CocoImage.__annotations__}
        return coco_image
