import json
from typing import Any


def load_json(value: str, default: Any = None) -> Any:
    return json.loads(value) if value else default
