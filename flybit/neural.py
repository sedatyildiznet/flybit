"""Biological control boundary for Flybit.

This module deliberately contains no behaviour rules. World state is encoded as
sensory input, the MaleCNS network is stepped, and only measured neural activity
is exposed to the rest of the application.
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
    visual_center: float


class FlybitNeuralCore:
    """Owns one independent MaleCNS simulation.

    v0.1 intentionally uses the upstream raw photoreceptor route rather than
    FeatureDetectors. The current upstream LIF model is known to lose much of
    this signal in the graded early visual pathway; Flybit reports that honestly
    instead of bypassing the eye with scripted behaviour.
    """

    def __init__(self, *, device: str = "auto", seed: int = 64) -> None:
        self.brain = FlyBrain(device=device, seed=seed)
        self.eyes = Eyes(self.brain.azimuth)
        self.descending = self.brain.cells(["descending_neuron"])
        self._descending_mask = np.zeros(self.brain.n, dtype=np.bool_)
        self._descending_mask[self.descending] = True

    @property
    def device(self) -> str:
        return self.brain.device

    @property
    def neuron_count(self) -> int:
        return int(self.brain.n)

    def step_visual_target(self, center: float | None) -> NeuralSnapshot:
        """Advance one biological timestep from a raw visual target.

        center is azimuth in [-1, 1]. None means an empty field. The target is
        rendered onto the 6,006 MaleCNS photoreceptors by the upstream eye model.
        No LC/LPLC feature neurons are injected directly.
        """
        blobs: list[Blob] = []
        visual_center = 0.0
        if center is not None:
            visual_center = float(np.clip(center, -1.0, 1.0))
            blobs.append(Blob(center=visual_center, half_width=0.035, darkness=1.0))

        eye_drive = self.eyes.drive(blobs)
        fired = self.brain.step(eye_drive=eye_drive)
        fired_np = np.asarray(fired, dtype=np.int64)
        if fired_np.size:
            descending_spikes = int(self._descending_mask[fired_np].sum())
            descending_active = int(np.unique(fired_np[self._descending_mask[fired_np]]).size)
        else:
            descending_spikes = 0
            descending_active = 0

        return NeuralSnapshot(
            step=int(self.brain.steps),
            total_spikes=int(fired_np.size),
            descending_spikes=descending_spikes,
            descending_active=descending_active,
            visual_center=visual_center,
        )
