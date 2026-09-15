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

from .sensory import DesktopMotionModel, SensoryDynamics


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
        self.motion = DesktopMotionModel(cursor_radius=14.0)

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
        local_x = x - geom.left()
        local_y = y - geom.top()

        # Capture only the retinal neighbourhood instead of the full desktop.
        # At 50 Hz a full-screen copy is unnecessarily expensive; the retina
        # never samples beyond max(self.radii).
        pad = max(self.radii) + 4
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
                px = int(round(sample_x + cs * radius))
                py = int(round(sample_y + sn * radius))
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

        # Windows screen capture normally omits the hardware cursor. Add its
        # retinal silhouette as a sensory image, not as a behaviour command.
        if cursor is not None:
            overlay_object(
                float(cursor.x()),
                float(cursor.y()),
                14.0,
                0.07,
            )

        # Flybit's sugar drop is another world object. We explicitly render its
        # retinal silhouette so visibility does not depend on whether the OS
        # includes our transparent overlay window in screen capture.
        if food is not None:
            fx, fy, fr = food
            overlay_object(
                float(fx),
                float(fy),
                max(2.0, float(fr)),
                0.38,
            )

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
        """Capture retina and temporal motion cues from the same observation."""
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
            timestamp=timestamp,
        )
        return luminance, dynamics
