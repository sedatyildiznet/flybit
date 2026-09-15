"""Raw desktop luminance sampling for Flybit's compound-eye input.

No object detection, OCR, window classification or threat heuristic is used.
The sampler reads screen pixels around the fly and reduces them to a 1-D angular
luminance panorama. That panorama is interpolated onto MaleCNS photoreceptors by
the neural layer.
"""
from __future__ import annotations

import math

import numpy as np

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QGuiApplication, QImage, qGray


class DesktopRetinaSampler:
    """Sample the screen around the fly as raw luminance rays."""

    def __init__(
        self,
        *,
        bins: int = 384,
        radii: tuple[int, ...] = (42, 72, 110, 165, 240, 340),
    ) -> None:
        self.bins = int(bins)
        self.radii = tuple(int(r) for r in radii)
        self.azimuth = np.linspace(
            -1.0,
            1.0,
            self.bins,
            endpoint=False,
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
    ) -> np.ndarray:
        """Return luminance in [0, 1] for angular bins around the fly.

        Multiple radii are averaged along each ray. This is an optical
        downsampling step only; it does not identify shapes or objects.
        """
        origin = QPoint(int(round(x)), int(round(y)))
        screen = self._screen_at(origin)
        if screen is None:
            return np.full(
                self.bins,
                0.9,
                dtype=np.float32,
            )

        geom = screen.geometry()
        pixmap = screen.grabWindow(0)
        image = pixmap.toImage().convertToFormat(
            QImage.Format.Format_RGB32
        )

        local_x = x - geom.left()
        local_y = y - geom.top()

        values = np.empty(
            self.bins,
            dtype=np.float32,
        )

        for i, az in enumerate(self.azimuth):
            angle = heading + float(az) * math.pi
            cs = math.cos(angle)
            sn = math.sin(angle)
            total = 0.0
            count = 0

            for radius in self.radii:
                px = int(round(local_x + cs * radius))
                py = int(round(local_y + sn * radius))
                if (
                    0 <= px < image.width()
                    and 0 <= py < image.height()
                ):
                    total += qGray(image.pixel(px, py)) / 255.0
                    count += 1

            values[i] = (
                total / count
                if count
                else 0.9
            )

        # Windows screen capture normally omits the hardware cursor. Add its
        # retinal silhouette as a sensory image, not as a behaviour command.
        if cursor is not None:
            dx = float(cursor.x()) - x
            dy = float(cursor.y()) - y
            distance = max(
                1.0,
                math.hypot(dx, dy),
            )
            bearing = math.atan2(dy, dx)
            relative = (
                bearing - heading + math.pi
            ) % (2.0 * math.pi) - math.pi
            center = relative / math.pi
            half_width = min(
                0.40,
                max(
                    0.006,
                    math.atan2(14.0, distance)
                    / math.pi,
                ),
            )
            delta = np.abs(
                (
                    self.azimuth - center + 1.0
                ) % 2.0 - 1.0
            )
            values[delta <= half_width] *= 0.08

        return np.clip(
            values,
            0.0,
            1.0,
        ).astype(np.float32)
