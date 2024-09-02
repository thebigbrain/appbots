from typing import Any, override, Sequence

import numpy as np
from gymnasium import spaces
from gymnasium.spaces.space import T_cov


class Tree(spaces.Space):
    def __init__(
            self,
            seed: int | np.random.Generator | None = None,
    ):
        super().__init__(None, None, seed)

    @override
    @property
    def is_np_flattenable(self) -> bool:
        return False

    @override
    def sample(self, mask: Any | None = None) -> T_cov:
        pass

    @override
    def contains(self, x: Any) -> bool:
        return False

    @override
    def from_jsonable(self, sample_n: list[Any]) -> list[T_cov]:
        pass

    @override
    def to_jsonable(self, sample_n: Sequence[T_cov]) -> list[Any]:
        pass
