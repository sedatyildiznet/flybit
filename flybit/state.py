"""Persistent identity, body position and care state for Flybit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


STATE_SCHEMA = 2


@dataclass
class FlybitState:
    created_at: str
    launches: int = 0
    x: float | None = None
    y: float | None = None
    heading: float = 0.0
    hunger: float = 0.35
    feedings: int = 0
    last_feed_at: str | None = None
    schema: int = STATE_SCHEMA

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
    state.hunger = max(0.0, min(1.0, float(state.hunger)))
    state.schema = STATE_SCHEMA
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
            old_schema = int(raw.get("schema", 1))
            allowed = {
                "created_at",
                "launches",
                "x",
                "y",
                "heading",
                "hunger",
                "feedings",
                "last_feed_at",
                "schema",
            }
            state = FlybitState(
                **{
                    key: value
                    for key, value in raw.items()
                    if key in allowed
                }
            )

            # alpha.3/alpha.4 could persist a body directly on a screen edge.
            # The old edge model could then strand the upgraded organism there.
            # Reset position once when migrating to the new state schema.
            if old_schema < STATE_SCHEMA:
                state.x = None
                state.y = None
                state.heading = 0.0
                state.schema = STATE_SCHEMA
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
