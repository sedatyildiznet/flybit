"""Modeled bilateral compound-eye receptive fields and motion estimates."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EyeField:
    azimuth: np.ndarray
    elevation: np.ndarray
    luminance: np.ndarray


@dataclass(frozen=True)
class MotionField:
    horizontal_flow: float
    vertical_flow: float
    expansion: float
    rotation: float
    radial_coherence: float


@dataclass(frozen=True)
class CompoundEyeSnapshot:
    left: EyeField
    right: EyeField
    motion: MotionField
    male_cns_panorama: np.ndarray


class CompoundEyeModel:
    """Approximate ommatidial acceptance without changing MaleCNS input size."""

    def __init__(self, bins: int = 384, elevation_bands: tuple[float, ...] = (-.55, -.2, .1, .4),
                 acceptance_angle_deg: float = 4.8) -> None:
        self.bins = int(bins)
        self.elevation_bands = np.asarray(elevation_bands, dtype=np.float32)
        self.acceptance_angle_deg = float(acceptance_angle_deg)
        self.azimuth = np.linspace(-np.pi, np.pi, self.bins, endpoint=False, dtype=np.float32)
        self._previous: np.ndarray | None = None

    def process(self, compound: np.ndarray) -> CompoundEyeSnapshot:
        values = np.asarray(compound, dtype=np.float32)
        if values.ndim == 3:
            # Input convention is elevation/offset x radius x azimuth.
            field = values.mean(axis=1)
        elif values.ndim == 2:
            field = values
        else:
            raise ValueError("compound field must be [elevation,radius,azimuth] or [elevation,azimuth]")
        if field.shape[1] != self.bins:
            raise ValueError("compound azimuth bins do not match model")
        if field.shape[0] != len(self.elevation_bands):
            source = np.linspace(0, 1, field.shape[0])
            target = np.linspace(0, 1, len(self.elevation_bands))
            field = np.vstack([np.interp(target, source, field[:, i]) for i in range(self.bins)]).T.astype(np.float32)
        # Small circular blur approximates ommatidial acceptance angle.
        half_width = max(1, int(round(self.acceptance_angle_deg / (360.0 / self.bins))))
        kernel = np.ones(2 * half_width + 1, dtype=np.float32) / (2 * half_width + 1)
        padded = np.concatenate((field[:, -half_width:], field, field[:, :half_width]), axis=1)
        filtered = np.vstack([np.convolve(row, kernel, mode="valid") for row in padded])
        half = self.bins // 2
        left_values, right_values = filtered[:, :half], filtered[:, half:]
        delta = np.zeros_like(filtered) if self._previous is None else filtered - self._previous
        dx = np.diff(delta, axis=1, append=delta[:, :1])
        dy = np.diff(delta, axis=0, append=delta[-1:, :])
        dark_growth = np.maximum(-delta, 0.0)
        horizontal = float(np.clip(np.mean(dx) * 20.0, -2.0, 2.0))
        vertical = float(np.clip(np.mean(dy) * 20.0, -2.0, 2.0))
        expansion = float(np.clip(np.percentile(dark_growth, 92) * 4.0, 0.0, 1.0))
        rotation = float(np.clip(np.mean(dx * np.sign(self.azimuth)[None, :]) * 35.0, -2.0, 2.0))
        coherence = float(np.mean(dark_growth > .035))
        self._previous = filtered.copy()
        panorama = filtered.mean(axis=0).astype(np.float32)
        return CompoundEyeSnapshot(
            EyeField(self.azimuth[:half].copy(), self.elevation_bands.copy(), left_values.copy()),
            EyeField(self.azimuth[half:].copy(), self.elevation_bands.copy(), right_values.copy()),
            MotionField(horizontal, vertical, expansion, rotation, coherence), panorama)

