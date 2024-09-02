import random
from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Discrete, Box, flatdim, MultiDiscrete

from appbots.core.env import Env
from appbots.core.images.gen import generate_sample_image


class UiElementEnv(Env):
    _steps = 0
    _current_obs = None

    def __init__(self):
        self.observation_space = Box(low=0, high=1.0, shape=(4, 200, 100), dtype=np.float32)

        self.action_space = Discrete(8 + 200 * 100)

    @property
    def n_actions(self) -> Any:
        return flatdim(self.action_space)


    def _get_info(self):
        return {
            "step": self._steps
        }

    def _get_next_state(self):
        img = generate_sample_image()
        return torch.from_numpy(img).permute(2, 0, 1)

    def _get_reward(self, obs, action):
        return random.random()

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_next_state()
        info = self._get_info()

        self._current_obs = observation

        return observation, info

    def step(self, action):
        terminated = self._steps > 100

        reward = self._get_reward(self._current_obs, action)
        state = self._get_next_state()

        info = self._get_info()
        trunked = False

        if terminated:
            self._steps = 0

        return state, reward, terminated, trunked, info

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    space = MultiDiscrete([8, 200, 100])
    print(f"{flatdim(Discrete(8 + 200 * 100))}")
    # sample = space.sample()
    # print(f"{space}, {space / np.array([200, 100])}")

