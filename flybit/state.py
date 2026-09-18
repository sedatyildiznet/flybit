"""Persistent identity, body position and care state for Flybit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


STATE_SCHEMA = 6


@dataclass
class FlybitState:
    created_at: str
    launches: int = 0
    display_name: str = "Flybit"
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
    sleep_pressure: float = 0.35
    last_simulated_at: str | None = None
    grooming_need: float = 0.18
    threat_memory: float = 0.0
    panel_x: int | None = None
    panel_y: int | None = None
    panel_w: int | None = None
    panel_h: int | None = None
    schema: int = STATE_SCHEMA

    @classmethod
    def new(cls) -> "FlybitState":
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            created_at=now,
            launches=0,
            last_simulated_at=now,
        )


def state_dir() -> Path:
    base = Path.home() / "AppData" / "Local" / "Flybit"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _path() -> Path:
    return state_dir() / "state.json"


def elapsed_since_last_simulation(
    state: FlybitState,
    *,
    now: datetime | None = None,
) -> float:
    """Wall-clock seconds since the organism was last persisted."""
    raw = getattr(state, "last_simulated_at", None)
    if not raw:
        return 0.0
    try:
        then = datetime.fromisoformat(raw)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return 0.0
    current = now or datetime.now(timezone.utc)
    return max(0.0, (current - then).total_seconds())


def mark_simulated_now(state: FlybitState) -> None:
    state.last_simulated_at = datetime.now(timezone.utc).isoformat()


def normalize_display_name(value: str | None) -> str:
    """Return a safe, compact organism name for persistent UI identity."""
    text = " ".join(str(value or "").split()).strip()
    return (text[:32] or "Flybit")


def save_state(state: FlybitState) -> None:
    state.hunger = max(0.0, min(1.0, float(state.hunger)))
    state.sleep_pressure = max(
        0.0,
        min(1.0, float(getattr(state, "sleep_pressure", 0.35))),
    )
    state.display_name = normalize_display_name(state.display_name)
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
                "display_name",
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
                "sleep_pressure",
                "last_simulated_at",
                "grooming_need",
                "threat_memory",
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
            state.display_name = normalize_display_name(
                getattr(state, "display_name", "Flybit")
            )
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
