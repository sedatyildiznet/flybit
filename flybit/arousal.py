"""Short-term threat arousal from looming perception and neural escape output."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ArousalSnapshot:
    threat_arousal: float


class ThreatArousalModel:
    """Sensitization state around threat perception and DN escape activity.

    Looming can raise readiness before take-off, while an actual neural escape
    produces a stronger sensitization pulse. The state still never issues a
    movement command by itself.
    """

    def __init__(self, *, decay_seconds: float = 18.0) -> None:
        self.decay_seconds = max(1.0, float(decay_seconds))
        self._value = 0.0

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def tick(
        self,
        dt: float,
        *,
        escape_drive: float,
        sensory_threat: float = 0.0,
    ) -> None:
        dt = max(0.0, float(dt))
        escape = self._clamp01(escape_drive)
        sensory = self._clamp01(sensory_threat)
        self._value *= math.exp(-dt / self.decay_seconds)
        if sensory > 0.0:
            self._value = max(
                self._value,
                self._clamp01(0.10 + 0.62 * sensory),
            )
        if escape > 0.0:
            self._value = max(
                self._value,
                self._clamp01(0.30 + 0.70 * escape),
            )

    def snapshot(self) -> ArousalSnapshot:
        return ArousalSnapshot(threat_arousal=float(self._value))
