import requests
from torch.utils.data import Dataset

from appbots.datasets.annotation import get_tag_annotations, get_anno
from appbots.core.images.utils import url_to_tensor


class UiBotDataset(Dataset):
    def __init__(self):
        # ... 初始化代码，读取标注文件等
        self.data = list(get_tag_annotations(limit=100))

    def __getitem__(self, idx):
        # ... 读取图像和对应的标注信息
        d = self.data[idx]
        anno = get_anno(anno_id=d.get("id"))
        image_tensor = url_to_tensor(url=anno.get("screenshot"))
        return image_tensor

    def __len__(self):
        # ... 返回数据集的样本数量
        return len(self.data)


def get_latest_screenshot(device: str) -> str:
    response = requests.get(f"http://192.168.10.115:8900/screenshot/latest?device={device}")
    return response.text


if __name__ == "__main__":
    print(get_latest_screenshot(device="192.168.10.112"))
