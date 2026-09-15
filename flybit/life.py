"""Persistent life-history and phenotype model for Flybit."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib

from .state import FlybitState


@dataclass(frozen=True)
class LifeSnapshot:
    age_seconds: float
    age_days: float
    lifespan_days: float
    remaining_days: float
    life_progress: float
    alive: bool
    vitality: float
    energy: float
    sex: str
    activity: float
    boldness: float
    curiosity: float


class LifeModel:
    """Wall-clock life history for one persistent digital organism.

    Trait values are deterministic for the individual and do not directly choose
    actions. They act as physiological gain factors and are displayed as part of
    the organism's identity.
    """

    def __init__(self, state: FlybitState) -> None:
        self.state = state
        if state.lifespan_days is None:
            digest = hashlib.sha256(state.created_at.encode("utf-8")).digest()
            state.lifespan_days = 35.0 + (digest[0] / 255.0) * 20.0
            state.activity_trait = 0.35 + (digest[1] / 255.0) * 0.55
            state.boldness_trait = 0.20 + (digest[2] / 255.0) * 0.65
            state.curiosity_trait = 0.25 + (digest[3] / 255.0) * 0.65
        self._energy = float(getattr(state, "energy", 0.82))

    @staticmethod
    def _parse(value: str) -> datetime:
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

    def tick(self, dt: float, *, motor_load: float, hunger: float) -> None:
        drain = max(0.0, float(dt)) * (
            0.000015
            + 0.000045 * max(0.0, min(1.0, motor_load))
            + 0.000025 * max(0.0, min(1.0, hunger))
        )
        recovery = max(0.0, float(dt)) * 0.000020 * (1.0 - hunger)
        self._energy = max(0.0, min(1.0, self._energy - drain + recovery))
        self.state.energy = self._energy

    def feed(self, amount: float = 1.0) -> None:
        self._energy = min(1.0, self._energy + 0.20 * max(0.1, float(amount)))
        self.state.energy = self._energy

    def snapshot(self) -> LifeSnapshot:
        now = datetime.now(timezone.utc)
        born = self._parse(self.state.created_at)
        age_seconds = max(0.0, (now - born).total_seconds())
        age_days = age_seconds / 86400.0
        lifespan = max(1.0, float(self.state.lifespan_days or 45.0))
        progress = max(0.0, min(1.0, age_days / lifespan))
        alive = progress < 1.0
        senescence = max(0.0, (progress - 0.65) / 0.35)
        vitality = max(0.0, min(1.0, self._energy * (1.0 - 0.65 * senescence)))
        return LifeSnapshot(
            age_seconds=age_seconds,
            age_days=age_days,
            lifespan_days=lifespan,
            remaining_days=max(0.0, lifespan - age_days),
            life_progress=progress,
            alive=alive,
            vitality=vitality,
            energy=self._energy,
            sex="Male",
            activity=float(self.state.activity_trait or 0.5),
            boldness=float(self.state.boldness_trait or 0.5),
            curiosity=float(self.state.curiosity_trait or 0.5),
        )
