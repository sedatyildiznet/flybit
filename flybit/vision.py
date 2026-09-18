"""Raw desktop luminance sampling for Flybit's compound-eye input.

The brain-facing panorama remains one-dimensional for MaleCNS compatibility,
while a small multi-row facet field is retained for motion coherence.
"""
from __future__ import annotations

import math

import numpy as np

from PySide6.QtCore import QPoint
from PySide6.QtGui import QGuiApplication, QImage, qGray

from .sensory import DesktopMotionModel, SensoryDynamics


class DesktopRetinaSampler:
    """Sample local desktop luminance through a compact compound-eye field."""

    def __init__(
        self,
        *,
        bins: int = 384,
        radii: tuple[int, ...] = (42, 72, 110, 165, 240, 340),
        facet_offsets: tuple[int, ...] = (-9, 0, 9),
    ) -> None:
        self.bins = int(bins)
        self.radii = tuple(int(r) for r in radii)
        self.facet_offsets = tuple(int(v) for v in facet_offsets)
        self.azimuth = np.linspace(
            -1.0,
            1.0,
            self.bins,
            endpoint=False,
            dtype=np.float32,
        )
        self.motion = DesktopMotionModel(cursor_radius=14.0)
        self._last_radial_luminance = np.full(
            (len(self.radii), self.bins),
            0.9,
            dtype=np.float32,
        )
        self._last_compound_luminance = np.full(
            (
                len(self.facet_offsets),
                len(self.radii),
                self.bins,
            ),
            0.9,
            dtype=np.float32,
        )

    @staticmethod
    def _screen_at(point: QPoint):
        screen = QGuiApplication.screenAt(point)
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        return screen

    def sample(
        self,
        *,
        x: float,
        y: float,
        heading: float,
        cursor: QPoint | None = None,
        food: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """Return brain-facing luminance in [0, 1] for angular bins."""
        origin = QPoint(int(round(x)), int(round(y)))
        screen = self._screen_at(origin)
        if screen is None:
            self._last_compound_luminance = np.full(
                (
                    len(self.facet_offsets),
                    len(self.radii),
                    self.bins,
                ),
                0.9,
                dtype=np.float32,
            )
            self._last_radial_luminance = self._last_compound_luminance.mean(
                axis=0
            )
            return np.full(
                self.bins,
                0.9,
                dtype=np.float32,
            )

        geom = screen.geometry()
        local_x = x - geom.left()
        local_y = y - geom.top()

        pad = max(self.radii) + max(
            abs(v) for v in self.facet_offsets
        ) + 5
        capture_left = max(0, int(math.floor(local_x - pad)))
        capture_top = max(0, int(math.floor(local_y - pad)))
        capture_right = min(
            geom.width(),
            int(math.ceil(local_x + pad + 1)),
        )
        capture_bottom = min(
            geom.height(),
            int(math.ceil(local_y + pad + 1)),
        )
        capture_width = max(1, capture_right - capture_left)
        capture_height = max(1, capture_bottom - capture_top)

        pixmap = screen.grabWindow(
            0,
            capture_left,
            capture_top,
            capture_width,
            capture_height,
        )
        image = pixmap.toImage().convertToFormat(
            QImage.Format.Format_RGB32
        )
        sample_x = local_x - capture_left
        sample_y = local_y - capture_top

        compound = np.full(
            (
                len(self.facet_offsets),
                len(self.radii),
                self.bins,
            ),
            0.9,
            dtype=np.float32,
        )

        for i, az in enumerate(self.azimuth):
            angle = heading + float(az) * math.pi
            cs = math.cos(angle)
            sn = math.sin(angle)
            # Perpendicular offset creates a narrow multi-row receptive field
            # around each ray rather than collapsing all local spatial detail.
            px_axis = -sn
            py_axis = cs

            for row, offset in enumerate(self.facet_offsets):
                ox = px_axis * offset
                oy = py_axis * offset
                for radius_index, radius in enumerate(self.radii):
                    px = int(round(sample_x + cs * radius + ox))
                    py = int(round(sample_y + sn * radius + oy))
                    if (
                        0 <= px < image.width()
                        and 0 <= py < image.height()
                    ):
                        compound[row, radius_index, i] = (
                            qGray(image.pixel(px, py)) / 255.0
                        )

        radial_values = compound.mean(axis=0)
        values = radial_values.mean(axis=0)

        def overlay_object(
            ox: float,
            oy: float,
            radius: float,
            luminance: float,
        ) -> None:
            dx = float(ox) - x
            dy = float(oy) - y
            distance = max(1.0, math.hypot(dx, dy))
            bearing = math.atan2(dy, dx)
            relative = (
                bearing - heading + math.pi
            ) % (2.0 * math.pi) - math.pi
            center = relative / math.pi
            half_width = min(
                0.40,
                max(
                    0.006,
                    math.atan2(radius, distance) / math.pi,
                ),
            )
            delta = np.abs(
                (
                    self.azimuth - center + 1.0
                ) % 2.0 - 1.0
            )
            mask = delta <= half_width
            values[mask] = np.minimum(
                values[mask],
                np.float32(luminance),
            )
            radial_values[:, mask] = np.minimum(
                radial_values[:, mask],
                np.float32(luminance),
            )
            compound[:, :, mask] = np.minimum(
                compound[:, :, mask],
                np.float32(luminance),
            )

        # The OS often omits the hardware cursor from screen capture. Its dark
        # retinal silhouette is inserted as optics only, never as an action.
        if cursor is not None:
            overlay_object(
                float(cursor.x()),
                float(cursor.y()),
                14.0,
                0.07,
            )

        if food is not None:
            fx, fy, fr = food
            overlay_object(
                float(fx),
                float(fy),
                max(2.0, float(fr)),
                0.38,
            )

        self._last_compound_luminance = np.clip(
            compound,
            0.0,
            1.0,
        ).astype(np.float32)
        self._last_radial_luminance = np.clip(
            radial_values,
            0.0,
            1.0,
        ).astype(np.float32)

        return np.clip(
            values,
            0.0,
            1.0,
        ).astype(np.float32)

    def sample_with_dynamics(
        self,
        *,
        x: float,
        y: float,
        heading: float,
        cursor: QPoint | None = None,
        food: tuple[float, float, float] | None = None,
        timestamp: float | None = None,
    ) -> tuple[np.ndarray, SensoryDynamics | None]:
        luminance = self.sample(
            x=x,
            y=y,
            heading=heading,
            cursor=cursor,
            food=food,
        )
        if cursor is None:
            return luminance, None
        dynamics = self.motion.update(
            body_x=x,
            body_y=y,
            heading=heading,
            cursor_x=float(cursor.x()),
            cursor_y=float(cursor.y()),
            luminance=luminance,
            radial_luminance=self._last_radial_luminance,
            compound_luminance=self._last_compound_luminance,
            timestamp=timestamp,
        )
        return luminance, dynamics
