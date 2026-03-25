from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


@dataclass
class TelemetryLogger:
    path: Path

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, event: dict[str, Any]) -> None:
        self._fh.write(json.dumps(_to_plain(event), sort_keys=True) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __del__(self) -> None:
        self.close()

    # Context-manager protocol to prevent resource leaks.
    def __enter__(self) -> TelemetryLogger:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()
