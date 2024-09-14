from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from appbots.datasets.annotation import get_tag_annotations, get_anno
from appbots.core.images.utils import url_to_tensor
from appbots.pageclassifier.labels import get_labels_tensor


class UiClassifierDataset(Dataset):
    def __init__(self):
        # ... 初始化代码，读取标注文件等
        self.data = list(get_tag_annotations())

    def __getitem__(self, idx):
        # ... 读取图像和对应的标注信息
        d = self.data[idx]
        anno = get_anno(anno_id=d.get("id"))
        image_tensor = url_to_tensor(url=anno.get("screenshot"))
        tags = d.get("annotation", "").split(',') if d.get("annotation", "") is not None else ['未知']
        tags_tensor = get_labels_tensor(tags)
        return image_tensor, tags_tensor

    def __len__(self):
        # ... 返回数据集的样本数量
        return len(self.data)


if __name__ == "__main__":
    training_data = UiClassifierDataset()

    train_dataloader = DataLoader(training_data, batch_size=32)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))

    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img: Tensor = train_features[0].squeeze()
    img = img.permute(1, 2, 0)

    label = train_labels[0].squeeze()
    print(f"Label: {label}")

    plt.imshow(img, cmap="gray")
    plt.show()
