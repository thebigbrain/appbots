from typing import Any

import numpy as np
from gymnasium.spaces import Discrete, Box, flatdim

from appbots.core.env import Env
from appbots.pageclassifier.images import url_to_tensor
from appbots.uibot.dataset import get_latest_screenshot

NORMAL_REWARD = 1.0 * 10
STEPS_PER_RUN = 60


class UiBotEnv(Env):
    _steps = 0
    _current_coins = 0
    device = "192.168.10.112"

    def __init__(self):
        image_height = 100
        image_width = 50
        self.shape = (4, image_height, image_width)

        self.observation_space = Box(low=0, high=1.0, shape=self.shape, dtype=np.float32)

        self.action_space = Discrete(8 + image_width * image_height)

    @property
    def n_actions(self) -> Any:
        return flatdim(self.action_space)

    def _get_info(self):
        return {
            "step": self._steps,
            "coins": self._current_coins
        }

    def _get_screenshot(self) -> str:
        return get_latest_screenshot(self.device)

    def _get_state(self):
        screenshot = self._get_screenshot()
        image_tensor = url_to_tensor(url=screenshot, size=(self.shape[2], self.shape[1]))
        return image_tensor

    def _get_coins(self) -> float:
        pass

    def _get_reward(self, state, action):
        if self._steps > STEPS_PER_RUN:
            coins = self._get_coins()
            reward = (coins - self._current_coins) / STEPS_PER_RUN
            self._current_coins = coins
            return reward

        return NORMAL_REWARD

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        state = self._get_state()
        info = self._get_info()

        return state, info

    def step(self, action):
        state = self._get_state()

        info = self._get_info()
        trunked = self._steps > STEPS_PER_RUN

        reward = self._get_reward(state, action)
        terminated = reward < NORMAL_REWARD * 0.8

        done = trunked or terminated
        if done:
            self._steps = 0

        return state, reward, terminated, trunked, info

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    pass
