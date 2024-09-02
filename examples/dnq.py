import random

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 超参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('CartPole-v1')  # 替换为你的环境
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_size = 64
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
buffer_size = 10000
target_update = 100

# 创建DQN网络
dqn = DQN(state_dim, action_dim).to(device)
target_dqn = DQN(state_dim, action_dim).to(device)
target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()

# 优化器和损失函数
optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 经验回放
replay_buffer = []


def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        q_values = dqn(state)
        return q_values.max(1)[1].item()


def train(batch_size):
    state, action, reward, next_state, done = zip(*random.sample(replay_buffer, batch_size))
    state = torch.tensor(state, dtype=torch.float).to(device)
    action = torch.tensor(action).to(device)
    reward = torch.tensor(reward, dtype=torch.float).unsqueeze(1).to(device)
    next_state = torch.tensor(next_state, dtype=torch.float).to(device)
    done = torch.tensor(done, dtype=torch.bool).unsqueeze(1).to(device)

    q_values = dqn(state)
    q_values = q_values.gather(1, action.unsqueeze(1))

    next_q_values = target_dqn(next_state)
    next_q_values[done] = 0.0
    max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
    target_q_values = reward + (gamma * max_next_q_values)

    loss = criterion(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# 训练循环
for episode in range(1000):
    state = env.reset()
    for t in range(1000):
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        if len(replay_buffer) > buffer_size:
            replay_buffer.pop(0)
        state = next_state

        if len(replay_buffer) >= batch_size:
            train(batch_size)

        if done:
            break

        if episode % target_update == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

