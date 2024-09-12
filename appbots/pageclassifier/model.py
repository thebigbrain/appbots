from torchvision.models import ResNet50_Weights, resnet50
import torch.nn as nn

from appbots.pageclassifier.labels import LabelCategory

MODEL_NAME = "page_classifier"


def get_model():
    num_classes = LabelCategory.get_num_classes()
    # 模型定义
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # 冻结前几层
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后的全连接层
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)  # num_classes为您的类别数
    return model
