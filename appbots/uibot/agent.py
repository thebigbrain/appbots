import os
from typing import Any, SupportsFloat

import numpy as np
import torch
import torch.nn as nn

from appbots.core.agent import Agent
from appbots.core.device import device
from appbots.core.env import Env


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


class UiAgent(Agent):
    dqn: DQN
    target_dqn: DQN
    path: str

    batch_size = 8

    learning_rate = 0.001
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.99

    def __init__(self, name: str, env: Env, epsilon=0.1):
        self.env = env
        self.epsilon = epsilon
        self.path = os.path.join(os.path.abspath(".models"), f"{name}.pth")

        input_shape = env.observation_space.shape
        n_actions = env.n_actions

        print(f"{input_shape} {n_actions}")

        self._load(input_shape, n_actions)

        self.target_dqn = DQN(input_shape, n_actions)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()

    def _load(self, input_shape, n_actions):
        if os.path.exists(self.path):
            self.dqn = torch.load(self.path, weights_only=False)
            print("模型加载成功")
        else:
            self.dqn = DQN(input_shape=input_shape,
                           n_actions=n_actions)
            print("模型文件不存在")

    def save(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        torch.save(self.target_dqn, self.path)

    def get_action(self, state: Any) -> Any:
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            state = state.type(torch.float).unsqueeze(0)
            q_values = self.dqn(state)
            action = q_values.max(1)[1].item()
        return action

    def update(self, state: Any,
               action: Any,
               reward: SupportsFloat,
               done: bool,
               next_state: Any, ):
        self.train(state, action, reward, done, next_state)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, state, action, reward, done, next_state):
        state = state.type(torch.float).unsqueeze(0).to(device)
        next_state = next_state.type(torch.float).unsqueeze(0).to(device)
        done = torch.tensor(done, dtype=torch.bool).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        action = torch.tensor(action).to(device)

        q_values = self.dqn(state)

        q_values = q_values.gather(1, action.unsqueeze(0).unsqueeze(1))

        next_q_values = self.target_dqn(next_state)
        next_q_values[done] = 0
        max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
        target_q_values = reward + (self.gamma * max_next_q_values)

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @property
    def optimizer(self):
        return torch.optim.Adam(self.dqn.parameters(), lr=self.learning_rate)

    @property
    def criterion(self):
        return nn.MSELoss()


if __name__ == '__main__':
    # s = torch.rand(4, 200, 100).unsqueeze(0)
    # print(s.shape)
    #
    # dqn = DQN(s.shape, 8)
    # print(dqn(s).shape)
    d = list((torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])))
    print(torch.stack(d).shape)
