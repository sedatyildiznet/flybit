"""Biological control boundary for Flybit.

World state enters only through sensory transduction. MaleCNS activity is then
read out as neural telemetry and identified descending-neuron motor channels.
No mouse/window rule selects a movement here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain import FlyBrain
from flybrain.eyes import Blob, Eyes

from .motion import MotorActivity


@dataclass(frozen=True)
class NeuralSnapshot:
    step: int
    total_spikes: int
    descending_spikes: int
    descending_active: int
    visual_projection_spikes: int
    photoreceptor_rms: float
    lamina_rms: float
    visual_center: float
    visual_half_width: float
    motor: MotorActivity
    active_brain_points: tuple[tuple[float, float], ...]


class FlybitNeuralCore:
    """One independent MaleCNS simulation for the desktop organism."""

    MOTOR_FALLBACKS = {
        "forward_L": (["DNg100"], "L"),
        "forward_R": (["DNg100"], "R"),
        "steer_L": (["DNa02"], "L"),
        "steer_R": (["DNa02"], "R"),
        "escape_L": (["DNp01"], "L"),
        "escape_R": (["DNp01"], "R"),
        "backward_L": (["MDN"], "L"),
        "backward_R": (["MDN"], "R"),
    }

    def __init__(
        self,
        *,
        device: str = "auto",
        seed: int = 64,
    ) -> None:
        self.brain = FlyBrain(
            device=device,
            seed=seed,
            graded_visual=True,
        )
        self.eyes = Eyes(self.brain.azimuth)

        self.descending = self.brain.cells(
            ["descending_neuron"]
        )
        self.visual_projection = self.brain.cells(
            ["visual_projection"]
        )

        self._descending_mask = np.zeros(
            self.brain.n,
            dtype=np.bool_,
        )
        self._descending_mask[self.descending] = True

        self._visual_projection_mask = np.zeros(
            self.brain.n,
            dtype=np.bool_,
        )
        self._visual_projection_mask[
            self.visual_projection
        ] = True

        self.motor_groups: dict[str, np.ndarray] = {}
        for name, (types, side) in self.MOTOR_FALLBACKS.items():
            group = self.brain.groups.get(name)
            if group is None or len(group) == 0:
                group = self.brain.cells(
                    types,
                    side=side,
                )
            self.motor_groups[name] = np.asarray(
                group,
                dtype=np.int64,
            )

        self._brain_xy = self._normalize_brain_positions()

    @property
    def device(self) -> str:
        return self.brain.device

    @property
    def neuron_count(self) -> int:
        return int(self.brain.n)

    @property
    def graded_cell_count(self) -> int:
        return int(len(self.brain.graded))

    @staticmethod
    def _rms(values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        return float(
            np.sqrt(
                np.mean(
                    np.square(
                        values.astype(
                            np.float64,
                            copy=False,
                        )
                    )
                )
            )
        )

    def _normalize_brain_positions(self) -> np.ndarray:
        positions = self.brain.positions
        result = np.full(
            (self.brain.n, 2),
            np.nan,
            dtype=np.float32,
        )
        if positions is None or len(positions) != self.brain.n:
            return result

        raw = np.asarray(
            positions[:, :2],
            dtype=np.float64,
        )
        valid = np.all(np.isfinite(raw), axis=1)
        if not np.any(valid):
            return result

        values = raw[valid]
        lo = np.percentile(values, 1.0, axis=0)
        hi = np.percentile(values, 99.0, axis=0)
        span = np.maximum(hi - lo, 1.0)
        normalized = np.clip(
            (values - lo) / span,
            0.0,
            1.0,
        )
        result[valid] = normalized.astype(np.float32)
        return result

    def brain_layout(
        self,
        max_points: int = 3200,
    ) -> tuple[tuple[float, float], ...]:
        valid = np.flatnonzero(
            np.all(
                np.isfinite(self._brain_xy),
                axis=1,
            )
        )
        if len(valid) == 0:
            return ()

        stride = max(
            1,
            int(np.ceil(len(valid) / max_points)),
        )
        chosen = valid[::stride][:max_points]
        return tuple(
            (
                float(self._brain_xy[index, 0]),
                float(self._brain_xy[index, 1]),
            )
            for index in chosen
        )

    def _active_points(
        self,
        fired: np.ndarray,
        max_points: int = 180,
    ) -> tuple[tuple[float, float], ...]:
        if fired.size == 0:
            return ()

        valid = fired[
            np.all(
                np.isfinite(self._brain_xy[fired]),
                axis=1,
            )
        ]
        if valid.size == 0:
            return ()

        stride = max(
            1,
            int(np.ceil(valid.size / max_points)),
        )
        chosen = valid[::stride][:max_points]
        return tuple(
            (
                float(self._brain_xy[index, 0]),
                float(self._brain_xy[index, 1]),
            )
            for index in chosen
        )

    @staticmethod
    def _fraction_fired(
        fired: np.ndarray,
        group: np.ndarray,
    ) -> float:
        if fired.size == 0 or group.size == 0:
            return 0.0
        hits = np.count_nonzero(
            np.isin(
                fired,
                group,
                assume_unique=False,
            )
        )
        return float(hits / max(1, group.size))

    def _motor_activity(
        self,
        fired: np.ndarray,
    ) -> MotorActivity:
        value = lambda name: self._fraction_fired(
            fired,
            self.motor_groups[name],
        )
        return MotorActivity(
            forward_left=value("forward_L"),
            forward_right=value("forward_R"),
            steer_left=value("steer_L"),
            steer_right=value("steer_R"),
            escape_left=value("escape_L"),
            escape_right=value("escape_R"),
            backward_left=value("backward_L"),
            backward_right=value("backward_R"),
        )

    def step_visual_target(
        self,
        center: float | None,
        half_width: float = 0.035,
    ) -> NeuralSnapshot:
        """Advance one neural timestep from a raw visual object.

        center and half_width describe retinal geometry only. No looming,
        target, threat or behaviour classifier is used.
        """
        blobs: list[Blob] = []
        visual_center = 0.0
        visual_half_width = 0.0

        if center is not None:
            visual_center = float(
                np.clip(center, -1.0, 1.0)
            )
            visual_half_width = float(
                np.clip(
                    half_width,
                    0.008,
                    0.75,
                )
            )
            blobs.append(
                Blob(
                    center=visual_center,
                    half_width=visual_half_width,
                    darkness=1.0,
                )
            )

        eye_drive = self.eyes.contrast_drive(
            blobs,
            dt=self.brain.dt,
        )
        fired = self.brain.step(
            eye_drive=eye_drive
        )

        fired_np = np.asarray(
            fired,
            dtype=np.int64,
        )

        if fired_np.size:
            descending_hits = (
                self._descending_mask[fired_np]
            )
            visual_hits = (
                self._visual_projection_mask[
                    fired_np
                ]
            )
            descending_spikes = int(
                descending_hits.sum()
            )
            descending_active = int(
                np.unique(
                    fired_np[descending_hits]
                ).size
            )
            visual_projection_spikes = int(
                visual_hits.sum()
            )
        else:
            descending_spikes = 0
            descending_active = 0
            visual_projection_spikes = 0

        photoreceptor_rms = self._rms(
            self.brain.membrane_values(
                ["R1-6", "R7", "R8"]
            )
        )
        lamina_rms = self._rms(
            self.brain.membrane_values(
                ["L1", "L2", "L3"]
            )
        )

        return NeuralSnapshot(
            step=int(self.brain.steps),
            total_spikes=int(fired_np.size),
            descending_spikes=descending_spikes,
            descending_active=descending_active,
            visual_projection_spikes=(
                visual_projection_spikes
            ),
            photoreceptor_rms=photoreceptor_rms,
            lamina_rms=lamina_rms,
            visual_center=visual_center,
            visual_half_width=visual_half_width,
            motor=self._motor_activity(fired_np),
            active_brain_points=self._active_points(
                fired_np
            ),
        )
