"""Biological control boundary for Flybit.

World state is encoded only as sensory input. MaleCNS neural dynamics determine
all downstream activity; this module does not choose behaviours.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain import FlyBrain
from flybrain.eyes import Blob, Eyes


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


class FlybitNeuralCore:
    """One independent MaleCNS simulation for the desktop organism."""

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

    def step_visual_target(
        self,
        center: float | None,
    ) -> NeuralSnapshot:
        """Advance one neural timestep from the raw desktop visual target.

        No looming/target/threat classifier is used. A dark blob is rendered
        into the compound-eye scene, converted into signed adapting
        photoreceptor contrast, then passed through the mixed graded/spiking
        MaleCNS network.
        """
        blobs: list[Blob] = []
        visual_center = 0.0

        if center is not None:
            visual_center = float(
                np.clip(
                    center,
                    -1.0,
                    1.0,
                )
            )
            blobs.append(
                Blob(
                    center=visual_center,
                    half_width=0.035,
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
            total_spikes=int(
                fired_np.size
            ),
            descending_spikes=descending_spikes,
            descending_active=descending_active,
            visual_projection_spikes=(
                visual_projection_spikes
            ),
            photoreceptor_rms=photoreceptor_rms,
            lamina_rms=lamina_rms,
            visual_center=visual_center,
        )
