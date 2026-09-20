"""Modeled long-form sleep episodes and stage-dependent responsiveness."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SleepSnapshot:
    episode_seconds: float
    stage: str
    neural_noise_gain: float
    locomotor_leakage: float
    responsiveness: float
    woke: bool = False


class SleepEpisodeModel:
    STAGE_GAIN = {
        "awake": (1.0, 1.0, 1.0), "drowsy": (.72, .12, .72),
        "light": (.48, .035, .48), "deep": (.22, .008, .24),
        "micro-awake": (.85, .06, .88),
    }

    def __init__(self) -> None:
        self.episode_seconds = 0.0
        self._micro_awake_remaining = 0.0

    def tick(self, dt: float, *, sleeping: bool, threat: float = 0.0) -> SleepSnapshot:
        dt = max(0.0, float(dt))
        if not sleeping:
            self.episode_seconds = 0.0
            return self._snapshot("awake", woke=False)
        self.episode_seconds += dt
        wake_threshold = .16 if self.episode_seconds < 2 else (.25 if self.episode_seconds < 7 else .36)
        if threat >= wake_threshold:
            return self._snapshot("awake", woke=True)
        if self._micro_awake_remaining > 0:
            self._micro_awake_remaining -= dt
            return self._snapshot("micro-awake")
        # Brief deterministic arousals every ~75 s prevent perfectly inert sleep.
        phase = self.episode_seconds % 75.0
        if self.episode_seconds > 20 and phase < dt:
            self._micro_awake_remaining = .35
            return self._snapshot("micro-awake")
        stage = "drowsy" if self.episode_seconds < 1.8 else ("light" if self.episode_seconds < 7 else "deep")
        return self._snapshot(stage)

    def offline_progress(self, seconds: float, sleep_pressure: float) -> float:
        hours = min(16.0, max(0.0, float(seconds)) / 3600.0)
        return max(0.0, min(1.0, sleep_pressure * math.exp(-hours / 5.5) + .22 * (1 - math.exp(-hours / 5.5))))

    def _snapshot(self, stage: str, woke: bool = False) -> SleepSnapshot:
        noise, leakage, response = self.STAGE_GAIN[stage]
        return SleepSnapshot(self.episode_seconds, stage, noise, leakage, response, woke)

