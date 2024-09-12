import torch
import torch.nn.functional as F

from appbots.core.db import get_db


class LabelCategory:
    _labels = None

    @classmethod
    def get_labels(cls):
        if cls._labels is None:
            db = get_db()
            cls._labels = db.get_table("image_labels")
        return cls._labels

    @classmethod
    def get_num_classes(cls):
        return len(cls.get_labels())

    @classmethod
    def get_dict(cls):
        return {label['name']: label['id'] for _, label in enumerate(cls.get_labels())}

    @classmethod
    def get_label(cls, idx: int) -> str:
        for _, label in enumerate(cls.get_labels()):
            if label['id'] == idx:
                return label['name']


def one_hot_encode(labels, label_dict):
    indices = [label_dict[label] for label in labels]
    indices_tensor = torch.tensor(indices)
    return F.one_hot(indices_tensor, len(label_dict))


def get_labels_tensor(labels) -> torch.Tensor:
    label_dict = LabelCategory.get_dict()
    label_tensors = one_hot_encode(labels, label_dict)
    result = label_tensors[0]
    for i in range(1, len(label_tensors)):
        result += label_tensors[i]
    return result


if __name__ == "__main__":
    r = get_labels_tensor(['福利', '弹窗'])
    print(r, r.shape)
