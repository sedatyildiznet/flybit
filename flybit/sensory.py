"""Temporal sensory dynamics for the Flybit desktop organism.

This module converts physical desktop measurements into sensory cues. It does
not choose actions. Threat salience is diagnostic telemetry: it must never be
used as a direct escape command. Behaviour still emerges from the visual
MaleCNS route and the descending-neuron motor decoder.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class SensoryDynamics:
    """One temporal observation of the desktop around the organism."""

    timestamp: float
    dt: float
    cursor_distance: float
    cursor_speed: float
    cursor_acceleration: float
    closing_speed: float
    cursor_bearing: float
    cursor_angular_radius: float
    looming_rate: float
    time_to_collision: float | None
    threat_salience: float
    mechanosensory_disturbance: float
    optic_flow: float
    ambient_luminance: float
    retinal_loom_left: float
    retinal_loom_right: float
    loom_habituation: float


class DesktopMotionModel:
    """Track cursor and panoramic motion without selecting behaviour.

    Threat salience is an observer metric. The neural core never receives it as
    an action. The cursor reaches MaleCNS through the retinal luminance panorama,
    where its angular extent naturally grows as it approaches the fly.
    """

    def __init__(self, *, cursor_radius: float = 14.0) -> None:
        self.cursor_radius = max(1.0, float(cursor_radius))
        self._last_time: float | None = None
        self._last_cursor: tuple[float, float] | None = None
        self._last_body: tuple[float, float] | None = None
        self._last_cursor_speed = 0.0
        self._last_distance: float | None = None
        self._last_angular_radius: float | None = None
        self._last_luminance: np.ndarray | None = None
        self._last_radial_luminance: np.ndarray | None = None
        self._loom_habituation = 0.0

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _retinal_loom(
        previous: np.ndarray | None,
        current: np.ndarray,
    ) -> tuple[float, float]:
        """Estimate left/right dark-edge expansion from raw panorama frames.

        This is a modeled visual-motion transducer. It sees only luminance
        changes, never cursor identity or semantic desktop labels.
        """
        if previous is None:
            return 0.0, 0.0
        prev = np.asarray(previous, dtype=np.float32).reshape(-1)
        cur = np.asarray(current, dtype=np.float32).reshape(-1)
        if len(prev) != len(cur) or len(cur) < 16:
            return 0.0, 0.0

        growth = np.maximum(
            (1.0 - cur) - (1.0 - prev),
            0.0,
        )
        half = len(growth) // 2

        def score(values: np.ndarray) -> float:
            if values.size == 0:
                return 0.0
            k = max(4, min(16, values.size // 20))
            strongest = np.partition(values, -k)[-k:]
            return float(np.clip(strongest.mean() * 2.8, 0.0, 1.0))

        return score(growth[:half]), score(growth[half:])

    @staticmethod
    def _retinal_loom_radial(
        previous: np.ndarray | None,
        current: np.ndarray | None,
    ) -> tuple[float, float] | None:
        """Estimate coherent expansion across multiple retinal distance bands.

        The brain-facing panorama stays one-dimensional for compatibility, but
        looming can use the uncollapsed radial samples. A candidate expansion
        must therefore be supported by more than one distance band and by a
        short contiguous angular neighbourhood, reducing false alarms from a
        single dark pixel or abrupt local UI redraw.
        """
        if previous is None or current is None:
            return None
        prev = np.asarray(previous, dtype=np.float32)
        cur = np.asarray(current, dtype=np.float32)
        if (
            prev.ndim != 2
            or cur.ndim != 2
            or prev.shape != cur.shape
            or prev.shape[0] < 2
            or prev.shape[1] < 16
        ):
            return None

        growth = np.maximum(prev - cur, 0.0)
        coherence = np.mean(growth > 0.055, axis=0)
        angular = np.mean(growth, axis=0) * (
            0.35 + 0.65 * coherence
        )
        kernel = np.ones(5, dtype=np.float32) / 5.0
        smooth = np.convolve(angular, kernel, mode="same")
        half = len(smooth) // 2

        def score(values: np.ndarray) -> float:
            if values.size == 0:
                return 0.0
            threshold = float(np.percentile(values, 90.0))
            coherent = values[values >= threshold]
            if coherent.size == 0:
                return 0.0
            return float(
                np.clip(coherent.mean() * 4.0, 0.0, 1.0)
            )

        return score(smooth[:half]), score(smooth[half:])

    @staticmethod
    def _optic_flow(
        previous: np.ndarray | None,
        current: np.ndarray,
        dt: float,
    ) -> float:
        """Estimate signed circular panorama shift in revolutions/second."""
        if previous is None:
            return 0.0

        a = np.asarray(previous, dtype=np.float32).reshape(-1)
        b = np.asarray(current, dtype=np.float32).reshape(-1)
        if len(a) != len(b) or len(a) < 8:
            return 0.0

        a = a - float(a.mean())
        b = b - float(b.mean())
        energy = float(np.linalg.norm(a) * np.linalg.norm(b))
        if energy < 1e-5:
            return 0.0

        corr = np.fft.irfft(
            np.fft.rfft(a) * np.conj(np.fft.rfft(b)),
            n=len(a),
        )
        shift = int(np.argmax(corr))
        if shift > len(a) // 2:
            shift -= len(a)

        return float(shift / max(1, len(a)) / max(dt, 1e-3))

    def update(
        self,
        *,
        body_x: float,
        body_y: float,
        heading: float,
        cursor_x: float,
        cursor_y: float,
        luminance: np.ndarray,
        radial_luminance: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> SensoryDynamics:
        now = time.monotonic() if timestamp is None else float(timestamp)
        dt = (
            max(1e-3, min(0.25, now - self._last_time))
            if self._last_time is not None
            else 0.020
        )

        dx = float(cursor_x) - float(body_x)
        dy = float(cursor_y) - float(body_y)
        distance = max(1.0, math.hypot(dx, dy))
        bearing_world = math.atan2(dy, dx)
        bearing = (bearing_world - float(heading) + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        angular_radius = math.atan2(self.cursor_radius, distance)

        cursor_speed = 0.0
        relative_speed = 0.0
        if self._last_cursor is not None:
            dcx = float(cursor_x) - self._last_cursor[0]
            dcy = float(cursor_y) - self._last_cursor[1]
            cursor_speed = math.hypot(dcx, dcy) / dt

            if self._last_body is not None:
                prev_rx = self._last_cursor[0] - self._last_body[0]
                prev_ry = self._last_cursor[1] - self._last_body[1]
                relative_speed = math.hypot(
                    dx - prev_rx,
                    dy - prev_ry,
                ) / dt

        cursor_acceleration = (
            (cursor_speed - self._last_cursor_speed) / dt
            if self._last_time is not None
            else 0.0
        )
        closing_speed = (
            (self._last_distance - distance) / dt
            if self._last_distance is not None
            else 0.0
        )
        looming_rate = (
            (angular_radius - self._last_angular_radius) / dt
            if self._last_angular_radius is not None
            else 0.0
        )

        time_to_collision = (
            distance / closing_speed
            if closing_speed > 1.0
            else None
        )

        loom_component = self._clamp01(max(0.0, looming_rate) / 1.25)
        close_component = self._clamp01(max(0.0, closing_speed) / 900.0)
        near_component = self._clamp01((260.0 - distance) / 240.0)
        ttc_component = (
            self._clamp01((1.20 - time_to_collision) / 1.20)
            if time_to_collision is not None
            else 0.0
        )
        threat_salience = self._clamp01(
            0.42 * loom_component
            + 0.28 * ttc_component
            + 0.20 * close_component
            + 0.10 * near_component
        )

        motion_speed = max(cursor_speed, relative_speed)
        mechanosensory_disturbance = self._clamp01(
            (motion_speed / 1200.0)
            * math.exp(-distance / 90.0)
            * (0.35 + 0.65 * self._clamp01(closing_speed / 500.0))
        )

        lum = np.asarray(luminance, dtype=np.float32).reshape(-1)
        radial_loom = self._retinal_loom_radial(
            self._last_radial_luminance,
            radial_luminance,
        )
        if radial_loom is None:
            raw_loom_left, raw_loom_right = self._retinal_loom(
                self._last_luminance,
                lum,
            )
        else:
            raw_loom_left, raw_loom_right = radial_loom
        raw_loom = max(raw_loom_left, raw_loom_right)
        if raw_loom > 0.04:
            self._loom_habituation = self._clamp01(
                self._loom_habituation
                + dt * raw_loom / 2.8
            )
        else:
            self._loom_habituation = self._clamp01(
                self._loom_habituation
                - dt / 12.0
            )

        # Repeated harmless expansion gradually loses gain, while a strong
        # stimulus retains a minimum response. This is sensory habituation,
        # never a direct change to motor output.
        habituation_gain = 1.0 - 0.68 * self._loom_habituation
        retinal_loom_left = self._clamp01(
            raw_loom_left
            * (habituation_gain + 0.18 * raw_loom_left)
        )
        retinal_loom_right = self._clamp01(
            raw_loom_right
            * (habituation_gain + 0.18 * raw_loom_right)
        )

        optic_flow = self._optic_flow(self._last_luminance, lum, dt)
        ambient_luminance = float(np.clip(lum.mean(), 0.0, 1.0))

        self._last_time = now
        self._last_cursor = (float(cursor_x), float(cursor_y))
        self._last_body = (float(body_x), float(body_y))
        self._last_cursor_speed = cursor_speed
        self._last_distance = distance
        self._last_angular_radius = angular_radius
        self._last_luminance = lum.copy()
        self._last_radial_luminance = (
            None
            if radial_luminance is None
            else np.asarray(
                radial_luminance,
                dtype=np.float32,
            ).copy()
        )

        return SensoryDynamics(
            timestamp=now,
            dt=dt,
            cursor_distance=distance,
            cursor_speed=cursor_speed,
            cursor_acceleration=cursor_acceleration,
            closing_speed=closing_speed,
            cursor_bearing=bearing,
            cursor_angular_radius=angular_radius,
            looming_rate=looming_rate,
            time_to_collision=time_to_collision,
            threat_salience=threat_salience,
            mechanosensory_disturbance=mechanosensory_disturbance,
            optic_flow=optic_flow,
            ambient_luminance=ambient_luminance,
            retinal_loom_left=retinal_loom_left,
            retinal_loom_right=retinal_loom_right,
            loom_habituation=float(self._loom_habituation),
        )
