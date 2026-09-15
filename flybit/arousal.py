"""Short-term threat arousal driven by the organism's own neural escape output."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ArousalSnapshot:
    threat_arousal: float


class ThreatArousalModel:
    """Sensitization state that follows DN escape activity.

    The state never triggers escape itself. It only changes global neural
    readiness after the nervous system has already produced an escape response.
    """

    def __init__(self, *, decay_seconds: float = 18.0) -> None:
        self.decay_seconds = max(1.0, float(decay_seconds))
        self._value = 0.0

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def tick(self, dt: float, *, escape_drive: float) -> None:
        dt = max(0.0, float(dt))
        escape = self._clamp01(escape_drive)
        self._value *= math.exp(-dt / self.decay_seconds)
        if escape > 0.0:
            self._value = max(
                self._value,
                self._clamp01(0.25 + 0.75 * escape),
            )

    def snapshot(self) -> ArousalSnapshot:
        return ArousalSnapshot(threat_arousal=float(self._value))
