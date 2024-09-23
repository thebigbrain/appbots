import torch
import torch.nn.functional as F

# 创建一个随机输入张量和目标张量
# cross_input = torch.randn(3, 5)
# target = torch.tensor([1, 0, 2])
# target = torch.randint(5, (3,), dtype=torch.int64)
# target = torch.randn(3, 5).softmax(dim=1)
cross_input = torch.tensor([[-0.0925,  0.0197, -0.2582],
        [-0.3236, -0.4186, -0.0864],
        [-0.1060, -0.1761, -0.0670],
        [ 0.0175, -0.1561,  0.0966],
        [-0.0155, -0.1763,  0.1507],
        [-0.0155, -0.1763,  0.1507]])

target = torch.tensor([0,
        0,
        0,
        0,
        2,
        1])
# 计算交叉熵损失
loss = F.cross_entropy(cross_input, target)


if __name__ == "__main__":
    print(loss, cross_input, target)
