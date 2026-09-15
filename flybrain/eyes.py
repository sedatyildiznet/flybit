"""Visual transduction helpers for flybrain.

Two routes exist:

* Eyes: projects a simple 1-D scene onto the 6,006 MaleCNS photoreceptors.
  `drive()` preserves the original positive upstream encoding.
  `contrast_drive()` is Flybit's signed, adapting photoreceptor signal for
  mixed graded visual dynamics.
* FeatureDetectors: the original shortcut that directly drives identified
  projection neurons. Flybit's desktop organism path deliberately does not use
  this shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BACKGROUND = 0.9


@dataclass
class Blob:
    center: float
    half_width: float
    darkness: float


def render(azimuth: np.ndarray, blobs: list[Blob]) -> np.ndarray:
    lum = np.full(len(azimuth), BACKGROUND, np.float32)
    for b in blobs:
        inside = np.abs(azimuth - b.center) <= b.half_width
        lum[inside] = np.minimum(
            lum[inside],
            BACKGROUND * (1 - b.darkness),
        )
    return lum


class Eyes:
    def __init__(self, azimuth: np.ndarray):
        self.azimuth = azimuth
        self.previous: np.ndarray | None = None
        # Begin adapted to the neutral background. This makes a cursor/object
        # already present on the first frame produce retinal contrast instead
        # of being silently absorbed as the initial baseline.
        self.adaptation = np.full(
            len(azimuth),
            BACKGROUND,
            dtype=np.float32,
        )

    def drive(self, blobs: list[Blob]) -> np.ndarray:
        """Original fly.ai positive visual drive, retained for compatibility."""
        lum = render(self.azimuth, blobs)
        change = (
            np.zeros_like(lum)
            if self.previous is None
            else np.abs(lum - self.previous)
        )
        self.previous = lum
        return np.clip(0.45 * lum + 1.6 * change, 0, 1)

    def contrast_from_luminance(
        self,
        luminance: np.ndarray,
        *,
        dt: float = 0.020,
        adaptation_tau: float = 0.250,
    ) -> np.ndarray:
        """Signed adapting contrast from raw photoreceptor luminance.

        luminance must already be sampled/interpolated onto self.azimuth.
        """
        lum = np.asarray(luminance, dtype=np.float32)
        if lum.shape != self.azimuth.shape:
            raise ValueError(
                f"expected luminance shape {self.azimuth.shape}, got {lum.shape}"
            )

        baseline = np.maximum(
            self.adaptation,
            np.float32(0.05),
        )
        contrast = (lum - self.adaptation) / baseline

        tau = max(float(adaptation_tau), float(dt), 1e-6)
        alpha = np.float32(1.0 - np.exp(-float(dt) / tau))
        self.adaptation += alpha * (lum - self.adaptation)

        return np.clip(
            contrast,
            -1.0,
            1.0,
        ).astype(np.float32)

    def contrast_drive(
        self,
        blobs: list[Blob],
        *,
        dt: float = 0.020,
        adaptation_tau: float = 0.250,
    ) -> np.ndarray:
        """Signed adapting photoreceptor drive for graded visual dynamics.

        The rendered luminance is expressed relative to a slowly adapting local
        luminance estimate. Positive contrast depolarizes the model
        photoreceptor; negative contrast hyperpolarizes it. The method performs
        sensory transduction only: it does not classify objects, looming,
        targets, threats or behaviours.
        """
        lum = render(self.azimuth, blobs)

        return self.contrast_from_luminance(
            lum,
            dt=dt,
            adaptation_tau=adaptation_tau,
        )


# Original task-specific shortcut parameters. These remain available to upstream
# fly.ai experiments but are not used by Flybit's desktop organism path.
ENCODER = {
    "loom_gain": 10.0,
    "loom_size": 0.0,
    "chase_base": 0.6,
    "chase_gain": 0.2,
    "threat_max": 0.8,
    "shot_gain": 10.0,
    "cap": 0.8,
}

CHANNELS = {
    "loom": ["LPLC2"],
    "threat": ["LC4"],
    "shot": ["LPLC1"],
    "chase": ["LC10a"],
}


class FeatureDetectors:
    """Task shortcut retained for upstream compatibility.

    This bypasses the early graded visual system and directly stimulates
    identified projection neurons. Flybit does not use it for biological
    desktop control.
    """

    def __init__(self, brain, **encoder):
        unknown = set(encoder) - set(ENCODER)
        if unknown:
            raise ValueError(
                f"unknown encoder parameters: {sorted(unknown)}"
            )
        self.p = {
            k: np.asarray(encoder.get(k, v), np.float32)
            for k, v in ENCODER.items()
        }
        self.cells = {
            ch: {
                s: brain.cells(types, s)
                for s in "LR"
            }
            for ch, types in CHANNELS.items()
        }
        self.previous: dict = {}
        self.last = {
            f"{ch}{s}": 0.0
            for ch in CHANNELS
            for s in "LR"
        }

    @property
    def loom(self):
        return self.cells["loom"]

    @property
    def chase(self):
        return self.cells["chase"]

    def inject(self, opp=None, shots=(), threat: float = 0.0) -> list:
        p = self.p
        drive = {
            key: np.float32(0.0)
            for key in self.last
        }
        seen = {}

        def angle_and_growth(key, dx, size):
            angle = size / max(abs(dx), 8.0)
            seen[key] = angle
            return angle, max(
                0.0,
                angle - self.previous.get(key, angle),
            )

        if opp is not None:
            dx, size = opp
            s = "L" if dx < 0 else "R"
            angle, growth = angle_and_growth("opp", dx, size)
            drive[f"loom{s}"] = np.clip(
                growth * p["loom_gain"] + angle * p["loom_size"],
                0,
                p["cap"],
            )
            drive[f"chase{s}"] = np.clip(
                p["chase_base"] + p["chase_gain"] * angle,
                0,
                p["cap"],
            )
            drive[f"threat{s}"] = (
                p["threat_max"]
                * np.float32(np.clip(threat, 0, 1))
            )

        for key, dx, size in shots:
            s = "L" if dx < 0 else "R"
            _, growth = angle_and_growth(key, dx, size)
            drive[f"shot{s}"] = np.maximum(
                drive[f"shot{s}"],
                np.clip(
                    growth * p["shot_gain"],
                    0,
                    p["cap"],
                ),
            )

        self.previous = seen
        self.last = {
            key: float(np.mean(amount))
            for key, amount in drive.items()
        }
        return [
            (
                self.cells[key[:-1]][key[-1]],
                amount,
            )
            for key, amount in drive.items()
            if np.any(amount > 0)
        ]


def blob_for(dx: float, size: float, darkness: float) -> Blob:
    distance = max(abs(dx), 8.0)
    return Blob(
        center=float(np.clip(dx / 110.0, -1, 1)),
        half_width=float(
            np.clip(size / distance * 0.5, 0.03, 0.7)
        ),
        darkness=darkness,
    )
