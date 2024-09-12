import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        # 考虑到输入是图像，使用卷积神经网络提取特征
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        # 全连接层输出每个动作对应的Q值
        self.fc = nn.Linear(self.conv(torch.zeros(1, *input_shape)).shape[1], n_actions)

    def forward(self, state):
        x = self.conv(state)
        q_values = self.fc(x)
        return q_values
