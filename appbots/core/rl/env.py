from typing import Any

import gymnasium


class Env(gymnasium.Env):
    @property
    def n_actions(self) -> Any:
        raise NotImplemented()
