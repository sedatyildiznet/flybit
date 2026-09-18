"""Runtime ethogram recording and deterministic behaviour calibration helpers."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EthogramSummary:
    elapsed_seconds: float
    transitions: int
    mode_fraction: dict[str, float]
    mean_bout_seconds: dict[str, float]
    mean_speed: float
    mean_abs_turn_rate: float
    speed_cv: float


class EthogramRecorder:
    """Record behaviour without influencing it."""

    def __init__(self) -> None:
        self.elapsed = 0.0
        self.transitions = 0
        self._mode_time: dict[str, float] = defaultdict(float)
        self._bout_total: dict[str, float] = defaultdict(float)
        self._bout_count: dict[str, int] = defaultdict(int)
        self._current_mode: str | None = None
        self._current_bout = 0.0
        self._speed_sum = 0.0
        self._speed_sq_sum = 0.0
        self._turn_sum = 0.0
        self._samples = 0

    def update(
        self,
        dt: float,
        *,
        mode: str,
        speed: float,
        turn_rate: float,
    ) -> None:
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return
        mode = str(mode or "unknown")

        if self._current_mode is None:
            self._current_mode = mode
        elif mode != self._current_mode:
            self._bout_total[self._current_mode] += self._current_bout
            self._bout_count[self._current_mode] += 1
            self._current_bout = 0.0
            self._current_mode = mode
            self.transitions += 1

        self.elapsed += dt
        self._mode_time[mode] += dt
        self._current_bout += dt
        s = max(0.0, float(speed))
        self._speed_sum += s
        self._speed_sq_sum += s * s
        self._turn_sum += abs(float(turn_rate))
        self._samples += 1

    def snapshot(self) -> EthogramSummary:
        total = max(self.elapsed, 1e-9)
        mode_fraction = {
            key: value / total
            for key, value in sorted(self._mode_time.items())
        }

        bout_total = dict(self._bout_total)
        bout_count = dict(self._bout_count)
        if self._current_mode is not None and self._current_bout > 0.0:
            bout_total[self._current_mode] = (
                bout_total.get(self._current_mode, 0.0)
                + self._current_bout
            )
            bout_count[self._current_mode] = (
                bout_count.get(self._current_mode, 0) + 1
            )

        mean_bout = {
            key: bout_total[key] / max(1, bout_count.get(key, 0))
            for key in sorted(bout_total)
        }

        if self._samples:
            mean_speed = self._speed_sum / self._samples
            mean_sq = self._speed_sq_sum / self._samples
            variance = max(0.0, mean_sq - mean_speed * mean_speed)
            speed_cv = (
                math.sqrt(variance) / mean_speed
                if mean_speed > 1e-9
                else 0.0
            )
            mean_turn = self._turn_sum / self._samples
        else:
            mean_speed = 0.0
            speed_cv = 0.0
            mean_turn = 0.0

        return EthogramSummary(
            elapsed_seconds=self.elapsed,
            transitions=self.transitions,
            mode_fraction=mode_fraction,
            mean_bout_seconds=mean_bout,
            mean_speed=mean_speed,
            mean_abs_turn_rate=mean_turn,
            speed_cv=speed_cv,
        )


def wtsb_category(mode: str) -> str:
    """Map Flybit modes onto stop/walk/turn/boundary ethogram categories."""
    mode = str(mode)
    if mode == "boundary":
        return "boundary"
    if mode == "turn":
        return "sharp_turn"
    if mode in {"walk", "forage"}:
        return "curved_walk"
    if mode in {"idle", "groom", "feed", "sleep"}:
        return "stop"
    return "other"
