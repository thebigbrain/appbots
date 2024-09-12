import torch

from appbots.core.model import load_model
from appbots.datasets.annotation import update_tag_prediction, get_anno, get_tag_annotations
from appbots.pageclassifier.images import url_to_tensor
from appbots.pageclassifier.labels import LabelCategory
from appbots.pageclassifier.model import get_model, MODEL_NAME


def predict(img: torch.Tensor):
    model = get_model()
    load_model(model, MODEL_NAME)

    model.eval()

    batch = img.unsqueeze(0)

    _prediction = model(batch).squeeze(0).softmax(0)

    _class_id = _prediction.argmax().item()
    _score = _prediction[_class_id].item()
    _prediction_label = LabelCategory.get_label(_class_id)
    return _prediction_label, _score, _class_id, _prediction


def predict_anno(anno_id: int):
    anno = get_anno(anno_id)
    image_tensor = url_to_tensor(url=anno.get("screenshot"))
    p, s, i, prediction = predict(image_tensor)

    update_tag_prediction(anno.get('id'), p)
    print(p, s, i, prediction)


if __name__ == '__main__':
    iters = get_tag_annotations(limit=100)
    for item in iters:
        predict_anno(item.get('anno_id'))
