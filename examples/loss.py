import torch
import torch.nn as nn

# 创建损失函数
criterion = nn.CrossEntropyLoss()

# 假设输入是一个batch的logits，目标是每个样本的类别索引
inputs = torch.randn(4, 3, 640, 287)  # 10个样本，每个样本有3个类别
target = torch.empty((4, 640, 287), dtype=torch.long).random_(3)  # 每个样本的真实类别

# 计算损失
loss = criterion(inputs, target)

if __name__ == '__main__':
    print(inputs.shape)
    print(target.shape)
    print(loss)
