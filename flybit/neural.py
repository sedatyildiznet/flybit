"""Biological control boundary for Flybit.

World state enters only through sensory transduction. MaleCNS activity is then
read out as neural telemetry and identified descending-neuron motor channels.
No mouse/window rule selects movement here.
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
    homeostatic_drive: float
    vitality: float
    motor_rates: tuple[tuple[str, float], ...]
    active_brain_points: tuple[tuple[float, float], ...]


class FlybitNeuralCore:
    """One independent MaleCNS simulation for the desktop organism."""

    MOTOR_FALLBACKS = {
        "forward_L": (["DNg100"], "L"),
        "forward_R": (["DNg100"], "R"),
        "steer_L": (["DNa02"], "L"),
        "steer_R": (["DNa02"], "R"),
        "steer1_L": (["DNa01"], "L"),
        "steer1_R": (["DNa01"], "R"),
        "escape_L": (["DNp01"], "L"),
        "escape_R": (["DNp01"], "R"),
        "backward_L": (["MDN"], "L"),
        "backward_R": (["MDN"], "R"),
        # Optional 2026 forward-walking modulatory population. This group is
        # used only when the MaleCNS annotation actually contains the type.
        "dopa_L": (["DopaMeander"], "L"),
        "dopa_R": (["DopaMeander"], "R"),
        # MaleCNS splits DNg02 into subtypes such as DNg02_a/c/g.
        "flight_L": (["DNg02*"], "L"),
        "flight_R": (["DNg02*"], "R"),
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
                group = self._resolve_cells(
                    types,
                    side=side,
                )
            self.motor_groups[name] = np.asarray(
                group,
                dtype=np.int64,
            )

        self._brain_xy = self._normalize_brain_positions()
        self._motor_rates = {
            name: 0.0
            for name in self.motor_groups
        }
        self._homeostatic_drive = 0.0
        self._vitality = 1.0
        self._activity_trait = 0.5
        self._boldness_trait = 0.5
        self._curiosity_trait = 0.5
        self._base_tonic = float(self.brain.tonic)
        self._base_noise_hz = float(self.brain.noise_hz)

    def set_homeostasis(
        self,
        hunger: float,
        vitality: float,
        activity_trait: float = 0.5,
        boldness_trait: float = 0.5,
        curiosity_trait: float = 0.5,
    ) -> None:
        """Update global physiology without selecting a direction or action.

        Individual traits modulate tonic/arousal statistics only. They never
        issue steering, feeding or escape commands.
        """
        self._homeostatic_drive = float(np.clip(hunger, 0.0, 1.0))
        self._vitality = float(np.clip(vitality, 0.0, 1.0))
        self._activity_trait = float(np.clip(activity_trait, 0.0, 1.0))
        self._boldness_trait = float(np.clip(boldness_trait, 0.0, 1.0))
        self._curiosity_trait = float(np.clip(curiosity_trait, 0.0, 1.0))

        vitality_gain = 0.35 + 0.65 * self._vitality
        tonic_gain = (
            0.72
            + 0.28 * self._activity_trait
            + 0.12 * self._homeostatic_drive
            + 0.08 * self._boldness_trait
        ) * vitality_gain
        noise_gain = (
            0.60
            + 0.35 * self._activity_trait
            + 0.25 * self._curiosity_trait
            + 0.15 * self._homeostatic_drive
        ) * (0.55 + 0.45 * self._vitality)

        self.brain.tonic = self._base_tonic * float(
            np.clip(tonic_gain, 0.20, 1.35)
        )
        self.brain.noise_hz = self._base_noise_hz * float(
            np.clip(noise_gain, 0.20, 1.65)
        )

    def _resolve_cells(
        self,
        types: list[str],
        *,
        side: str | None = None,
    ) -> np.ndarray:
        """Resolve exact MaleCNS types and optional prefix patterns.

        A trailing "*" means "all flywireType subtypes with this prefix".
        This matters for descending populations such as DNg02, represented in
        MaleCNS as DNg02_a, DNg02_c, DNg02_g, ... rather than one exact type.
        """
        exact = [
            value
            for value in types
            if not value.endswith("*")
        ]
        prefixes = [
            value[:-1]
            for value in types
            if value.endswith("*")
        ]

        mask = np.zeros(self.brain.n, dtype=np.bool_)
        if exact:
            mask |= np.isin(
                self.brain.cell_type,
                exact,
            )

        if prefixes:
            names = np.asarray(
                self.brain.cell_type,
                dtype=str,
            )
            for prefix in prefixes:
                mask |= np.char.startswith(
                    names,
                    prefix,
                )

        if side:
            mask &= np.asarray(
                self.brain.side,
                dtype=str,
            ) == side

        return np.flatnonzero(mask).astype(np.int64)

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
        """Decode short-window firing rate from identified descending groups.

        A single stochastic spike must not become a full body command. We
        estimate an exponential firing rate over ~140 ms, then map activity
        above a quiet baseline into 0..1 motor drive.
        """
        tau = 0.140
        alpha = 1.0 - np.exp(
            -float(self.brain.dt) / tau
        )

        for name, group in self.motor_groups.items():
            fraction = self._fraction_fired(
                fired,
                group,
            )
            observed_hz = (
                fraction
                / max(float(self.brain.dt), 1e-6)
            )
            self._motor_rates[name] += (
                observed_hz
                - self._motor_rates[name]
            ) * alpha

        def decode(
            name: str,
            baseline_hz: float,
            span_hz: float,
        ) -> float:
            return float(
                np.clip(
                    (
                        self._motor_rates[name]
                        - baseline_hz
                    )
                    / span_hz,
                    0.0,
                    1.0,
                )
            )

        # DNg100 remains the primary forward-walking command. DNa01/DNa02
        # activity also contributes locomotor drive because bilateral
        # activation of these steering DNs increases walking. Their left/right
        # difference is still decoded separately as steering.
        dng_l = decode("forward_L", 1.2, 12.0)
        dng_r = decode("forward_R", 1.2, 12.0)

        dna02_l = decode("steer_L", 1.5, 14.0)
        dna02_r = decode("steer_R", 1.5, 14.0)
        dna01_l = decode("steer1_L", 1.5, 14.0)
        dna01_r = decode("steer1_R", 1.5, 14.0)

        dna_l = max(dna01_l, dna02_l)
        dna_r = max(dna01_r, dna02_r)
        dna_locomotor = 0.32 * (dna_l + dna_r)

        dopa_l = decode("dopa_L", 1.0, 10.0)
        dopa_r = decode("dopa_R", 1.0, 10.0)
        dopa_drive = 0.35 * (dopa_l + dopa_r)

        # Hunger is an internal arousal state, not a target selector. It raises
        # locomotor readiness while preserving the network's left/right choice.
        arousal = 0.72 + 0.38 * self._homeostatic_drive
        arousal *= 0.82 + 0.28 * self._activity_trait
        arousal *= 0.92 + 0.12 * self._boldness_trait
        arousal *= 0.90 + 0.14 * self._curiosity_trait
        arousal *= 0.35 + 0.65 * self._vitality
        forward_l = float(
            np.clip(
                (dng_l + dna_locomotor + dopa_drive) * arousal,
                0.0,
                1.0,
            )
        )
        forward_r = float(
            np.clip(
                (dng_r + dna_locomotor + dopa_drive) * arousal,
                0.0,
                1.0,
            )
        )

        escape_l_spike = (
            self._fraction_fired(
                fired,
                self.motor_groups["escape_L"],
            )
            > 0.0
        )
        escape_r_spike = (
            self._fraction_fired(
                fired,
                self.motor_groups["escape_R"],
            )
            > 0.0
        )

        return MotorActivity(
            forward_left=forward_l,
            forward_right=forward_r,
            steer_left=float(
                np.clip(
                    0.65 * dna02_l + 0.35 * dna01_l,
                    0.0,
                    1.0,
                )
            ),
            steer_right=float(
                np.clip(
                    0.65 * dna02_r + 0.35 * dna01_r,
                    0.0,
                    1.0,
                )
            ),
            # One Giant Fiber / DNp01 action potential is sufficient for the
            # fast escape take-off, so do not hide it behind a rate threshold.
            escape_left=(
                1.0
                if escape_l_spike
                else decode("escape_L", 0.8, 7.0)
            ),
            escape_right=(
                1.0
                if escape_r_spike
                else decode("escape_R", 0.8, 7.0)
            ),
            backward_left=decode(
                "backward_L", 1.5, 14.0
            ),
            backward_right=decode(
                "backward_R", 1.5, 14.0
            ),
            flight_left=decode(
                "flight_L", 1.2, 14.0
            ),
            flight_right=decode(
                "flight_R", 1.2, 14.0
            ),
        )

    def _finish_step(
        self,
        fired,
        *,
        visual_center: float = 0.0,
        visual_half_width: float = 0.0,
    ) -> NeuralSnapshot:
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
            visual_center=float(visual_center),
            visual_half_width=float(visual_half_width),
            motor=self._motor_activity(fired_np),
            homeostatic_drive=self._homeostatic_drive,
            vitality=self._vitality,
            motor_rates=tuple(
                (name, float(rate))
                for name, rate in sorted(self._motor_rates.items())
            ),
            active_brain_points=self._active_points(
                fired_np
            ),
        )

    def step_visual_luminance(
        self,
        luminance: np.ndarray,
        source_azimuth: np.ndarray,
    ) -> NeuralSnapshot:
        """Advance from a raw angular desktop luminance panorama.

        Both arrays are 1-D. Values are linearly interpolated onto the measured
        photoreceptor azimuths; no objects/features are recognized here.
        """
        lum = np.asarray(
            luminance,
            dtype=np.float32,
        )
        az = np.asarray(
            source_azimuth,
            dtype=np.float32,
        )
        if (
            lum.ndim != 1
            or az.ndim != 1
            or len(lum) != len(az)
            or len(lum) < 2
        ):
            raise ValueError(
                "luminance/source_azimuth must be equal-length 1-D arrays"
            )

        order = np.argsort(az)
        receptor_luminance = np.interp(
            self.brain.azimuth,
            az[order],
            lum[order],
            left=float(lum[order][0]),
            right=float(lum[order][-1]),
        ).astype(np.float32)

        eye_drive = self.eyes.contrast_from_luminance(
            receptor_luminance,
            dt=self.brain.dt,
        )
        fired = self.brain.step(
            eye_drive=eye_drive
        )
        return self._finish_step(fired)

    def step_visual_target(
        self,
        center: float | None,
        half_width: float = 0.035,
    ) -> NeuralSnapshot:
        """Compatibility path for simple synthetic retinal blobs."""
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
        return self._finish_step(
            fired,
            visual_center=visual_center,
            visual_half_width=visual_half_width,
        )
