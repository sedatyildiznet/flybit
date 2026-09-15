"""Persistent application metadata for Flybit.

Neural-state persistence will be added once the plasticity/state format is
stable. This file intentionally does not invent personality or behaviour stats.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass
class FlybitState:
    created_at: str
    launches: int = 0

    @classmethod
    def new(cls) -> "FlybitState":
        return cls(created_at=datetime.now(timezone.utc).isoformat(), launches=0)


def state_dir() -> Path:
    base = Path.home() / "AppData" / "Local" / "Flybit"
    base.mkdir(parents=True, exist_ok=True)
    return base


def load_state() -> FlybitState:
    path = state_dir() / "state.json"
    if not path.exists():
        state = FlybitState.new()
    else:
        try:
            state = FlybitState(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            state = FlybitState.new()
    state.launches += 1
    path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    return state
