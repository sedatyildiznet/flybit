"""Persistent identity, body position and care state for Flybit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


STATE_SCHEMA = 3


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
    lifespan_days: float | None = None
    energy: float = 0.82
    activity_trait: float | None = None
    boldness_trait: float | None = None
    curiosity_trait: float | None = None
    panel_x: int | None = None
    panel_y: int | None = None
    panel_w: int | None = None
    panel_h: int | None = None
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
                "lifespan_days",
                "energy",
                "activity_trait",
                "boldness_trait",
                "curiosity_trait",
                "panel_x",
                "panel_y",
                "panel_w",
                "panel_h",
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
            # Reset position only for the legacy locomotion migration. Later
            # schema upgrades keep the organism's physical location and life.
            if old_schema < 2:
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
