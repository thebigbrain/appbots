from typing import Any, SupportsFloat


class Agent:
    def get_action(self, state: Any) -> Any:
        raise NotImplementedError

    def update(self, state: Any,
               action: Any,
               reward: SupportsFloat,
               terminated: bool,
               next_state: Any, ):
        raise NotImplementedError

    def save(self):
        pass
