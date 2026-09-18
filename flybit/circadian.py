"""Modeled circadian and sleep-pressure physiology for Flybit.

This layer never issues a movement command. It only produces global wake/rest
modulation for the nervous-system bridge, leaving stimulus processing and motor
choice to MaleCNS.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

from .state import FlybitState


@dataclass(frozen=True)
class CircadianSnapshot:
    local_hour: float
    ambient_luminance: float
    wake_drive: float
    sleep_pressure: float
    rest_drive: float


class CircadianModel:
    """Compact dawn/dusk activity + homeostatic sleep-pressure model.

    The shape is an explicit computational approximation, not a claim that
    Flybit reproduces the complete Drosophila clock network.
    """

    def __init__(self, state: FlybitState) -> None:
        self.state = state
        self._local_hour = self._hour_now()
        self._ambient_luminance = 0.5
        self._wake_drive = 0.5
        self._rest_drive = 0.3
        self._recompute()

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _hour_now() -> float:
        now = datetime.now().astimezone()
        return (
            now.hour
            + now.minute / 60.0
            + now.second / 3600.0
        )

    @staticmethod
    def _circular_distance(hour: float, center: float) -> float:
        return abs((hour - center + 12.0) % 24.0 - 12.0)

    @classmethod
    def _peak(
        cls,
        hour: float,
        center: float,
        width: float,
    ) -> float:
        d = cls._circular_distance(hour, center)
        return math.exp(-0.5 * (d / max(width, 0.1)) ** 2)

    def _recompute(self) -> None:
        morning = self._peak(self._local_hour, 8.0, 2.2)
        evening = self._peak(self._local_hour, 19.0, 2.4)
        light = self._clamp01(self._ambient_luminance)

        self._wake_drive = self._clamp01(
            0.16
            + 0.34 * morning
            + 0.34 * evening
            + 0.18 * light
        )

        pressure = self._clamp01(self.state.sleep_pressure)
        self._rest_drive = self._clamp01(
            0.62 * pressure * (1.0 - 0.30 * self._wake_drive)
            + 0.38 * (1.0 - self._wake_drive)
        )

    def tick(
        self,
        dt: float,
        *,
        motor_load: float,
        ambient_luminance: float,
        local_hour: float | None = None,
    ) -> None:
        dt = max(0.0, float(dt))
        self._local_hour = (
            self._hour_now()
            if local_hour is None
            else float(local_hour) % 24.0
        )
        self._ambient_luminance = self._clamp01(ambient_luminance)

        self._recompute()

        load = self._clamp01(motor_load)
        wake = self._wake_drive

        # Pressure rises slowly while active/wake-biased and recovers during
        # quiet low-wake periods. Timescales are deliberately conservative.
        rise = dt / (9.0 * 3600.0) * (
            0.30 + 0.70 * max(load, wake)
        )
        recovery = dt / (5.5 * 3600.0) * (
            (1.0 - wake) * (1.0 - load)
        )
        self.state.sleep_pressure = self._clamp01(
            self.state.sleep_pressure + rise - recovery
        )
        self._recompute()

    def elapse(self, seconds: float) -> None:
        """Approximate rest/homeostatic change while the app was closed."""
        seconds = max(0.0, float(seconds))
        if seconds <= 0.0:
            return
        hours = min(24.0, seconds / 3600.0)
        # Closed-app time is treated as low-locomotion time. Sleep pressure
        # relaxes toward a moderate rested baseline instead of freezing.
        target = 0.28
        alpha = 1.0 - math.exp(-hours / 5.5)
        self.state.sleep_pressure = self._clamp01(
            self.state.sleep_pressure
            + (target - self.state.sleep_pressure) * alpha
        )
        self._local_hour = self._hour_now()
        self._recompute()

    def snapshot(self) -> CircadianSnapshot:
        return CircadianSnapshot(
            local_hour=float(self._local_hour),
            ambient_luminance=float(self._ambient_luminance),
            wake_drive=float(self._wake_drive),
            sleep_pressure=float(self.state.sleep_pressure),
            rest_drive=float(self._rest_drive),
        )
