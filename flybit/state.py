"""Persistent identity and desktop-body state for Flybit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass
class FlybitState:
    created_at: str
    launches: int = 0
    x: float | None = None
    y: float | None = None
    heading: float = 0.0

    @classmethod
    def new(cls) -> "FlybitState":
        return cls(
            created_at=datetime.now(timezone.utc).isoformat(),
            launches=0,
        )


def state_dir() -> Path:
    base = Path.home() / "AppData" / "Local" / "Flybit"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _path() -> Path:
    return state_dir() / "state.json"


def save_state(state: FlybitState) -> None:
    _path().write_text(
        json.dumps(asdict(state), indent=2),
        encoding="utf-8",
    )


def load_state() -> FlybitState:
    path = _path()
    if not path.exists():
        state = FlybitState.new()
    else:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            allowed = {
                "created_at",
                "launches",
                "x",
                "y",
                "heading",
            }
            state = FlybitState(
                **{
                    key: value
                    for key, value in raw.items()
                    if key in allowed
                }
            )
        except (
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            state = FlybitState.new()

    state.launches += 1
    save_state(state)
    return state
