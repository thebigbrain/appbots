import torch

from appbots.core.coco import CocoAnnotationUtil, CocoAnnotation, CocoCategoryUtil
from appbots.core.images.builder import ImageTensorBuilder
from appbots.core.images.transforms import gray_transform
from appbots.core.nn_modules.canny import Canny
from appbots.datasets.annotation import get_not_bboxes_generated_memos

if __name__ == '__main__':
    memos = get_not_bboxes_generated_memos()
    print(f"共{len(memos)}个未生成框框")

    for memo in memos:
        ss = memo.get('screenshot')
        img_id = memo.get("id")

        builder = ImageTensorBuilder()
        builder.load_from_url(ss)

        gray: torch.Tensor = gray_transform(builder.tensor)
        model = Canny()
        boxes: torch.Tensor = model(gray)
        print(f"检测到{len(boxes)}个框框")
        h = builder.source.size[1]
        boxes = boxes / h
        CocoAnnotationUtil.create_bboxes_annotations(img_id, boxes.tolist())
        CocoAnnotationUtil.set_bboxes_generated(img_id)
