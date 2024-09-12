import os
from typing import Any, SupportsFloat

import numpy as np
import torch
import torch.nn as nn

from appbots.core.agent import Agent
from appbots.core.device import device
from appbots.core.env import Env
from appbots.core.model import load_model, save_model
from appbots.uibot.dqn import DQN


class UiBotAgent(Agent):
    dqn: DQN
    target_dqn: DQN
    path: str
    name: str

    batch_size = 8

    learning_rate = 0.001
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.99

    def __init__(self, name: str, env: Env, epsilon=0.1):
        self.env = env
        self.epsilon = epsilon
        self.name = name

        self.path = os.path.join(os.path.abspath(".models"), f"{name}.pt")

        input_shape = env.observation_space.shape
        n_actions = env.n_actions

        print(f"{input_shape} {n_actions}")

        self.dqn = load_model(DQN(input_shape, n_actions), name=name)

        self.target_dqn = DQN(input_shape, n_actions)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()

    def save(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        save_model(self.target_dqn, self.name)

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
