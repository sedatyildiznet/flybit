"""Professional Windows desktop shell for the Flybit organism."""
from __future__ import annotations

from collections import deque
import math
import time

from PySide6.QtCore import (
    QObject,
    QPoint,
    QRectF,
    QThread,
    QTimer,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .arousal import ThreatArousalModel
from .benchmark import EthogramRecorder
from .care import CareModel
from .circadian import CircadianModel
from .ethology import EthologyModel, EthologySnapshot
from .icon import flybit_icon
from .life import LifeModel, LifeSnapshot
from .perception import DesktopSemanticScanner, PerceivedObject
from .motion import (
    FlyBodyState,
    FlyKinematics,
    MotorActivity,
)
from .neural import FlybitNeuralCore, NeuralSnapshot
from .olfaction import FoodOdorModel
from .phenotype import phenotype_from_identity, seed_from_identity
from .state import (
    elapsed_since_last_simulation,
    load_state,
    mark_simulated_now,
    normalize_display_name,
    save_state,
)
from .vision import DesktopRetinaSampler
from .world import WindowSurfaceScanner, boundary_cue


class FlyOverlay(QWidget):
    """Small anatomically styled fly driven by live biomechanics."""

    clicked = Signal()
    context_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.heading = 0.0
        self.airborne = True
        self.drive = 0.0
        self.gait_phase = 0.0
        self.altitude = 0.0
        self.behavior = "idle"
        self.legs = ()
        self.body_bob = 0.0
        self.head_yaw = 0.0
        self.proboscis_extension = 0.0
        self.groom_target = ""
        self.micro_action = ""
        self.leg_extension = 0.0
        self.body_scale = 1.0
        self._wing_phase = 0.0

        # Intentionally small: visible enough to read as a fly, but not large
        # enough to obscure text or become irritating during normal desktop use.
        self.setFixedSize(36, 30)
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )
        self.setToolTip(
            "Flybit · click for neural control panel"
        )

    def set_pose(
        self,
        *,
        heading: float,
        airborne: bool,
        drive: float,
        gait_phase: float = 0.0,
        altitude: float = 0.0,
        behavior: str = "idle",
        legs=(),
        body_bob: float = 0.0,
        head_yaw: float = 0.0,
        proboscis_extension: float = 0.0,
        groom_target: str = "",
        micro_action: str = "",
        leg_extension: float = 0.0,
        body_scale: float = 1.0,
    ) -> None:
        self.heading = heading
        self.airborne = airborne
        self.drive = max(0.0, min(1.0, drive))
        self.gait_phase = float(gait_phase) % 1.0
        self.altitude = max(0.0, float(altitude))
        self.behavior = str(behavior or "idle")
        self.legs = tuple(legs or ())
        self.body_bob = float(body_bob)
        self.head_yaw = max(-0.4, min(0.4, float(head_yaw)))
        self.proboscis_extension = max(
            0.0,
            min(1.0, float(proboscis_extension)),
        )
        self.groom_target = str(groom_target or "")
        self.micro_action = str(micro_action or "")
        self.leg_extension = max(0.0, min(1.0, float(leg_extension)))
        self.body_scale = max(0.90, min(1.08, float(body_scale)))

        wing_rate = 0.72 if airborne else 0.10
        if self.micro_action == "wing_flick":
            wing_rate += 0.38
        self._wing_phase = (
            self._wing_phase
            + wing_rate
            + self.drive * 0.30
        ) % (2.0 * math.pi)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(math.degrees(self.heading))
        painter.translate(0.0, self.body_bob * 0.40)
        painter.scale(self.body_scale, self.body_scale)

        shadow_alpha = max(
            10,
            min(48, int(48 - self.altitude * 0.32)),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, shadow_alpha)))
        painter.drawEllipse(QRectF(-8.5, -2.7, 18.0, 6.6))

        # Jointed legs are rendered from the six-leg gait model.
        leg_pen = QPen(QColor(39, 29, 21, 225), 0.78)
        leg_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(leg_pen)
        if self.legs:
            for leg in self.legs:
                root_x = float(leg.root_x)
                root_y = float(leg.root_y)
                foot_x = float(leg.foot_x)
                foot_y = float(leg.foot_y)
                side = -1.0 if foot_y < 0.0 else 1.0
                knee_x = root_x * 0.45 + foot_x * 0.55
                knee_y = (
                    root_y * 0.38
                    + foot_y * 0.62
                    + side * float(leg.lift) * 1.2
                )
                painter.drawLine(
                    int(round(root_x)),
                    int(round(root_y)),
                    int(round(knee_x)),
                    int(round(knee_y)),
                )
                painter.drawLine(
                    int(round(knee_x)),
                    int(round(knee_y)),
                    int(round(foot_x)),
                    int(round(foot_y)),
                )
        else:
            for root_x, root_y, foot_x, foot_y in (
                (4, -3, 9, -8),
                (0, -4, 0, -10),
                (-4, -3, -9, -8),
                (4, 3, 9, 8),
                (0, 4, 0, 10),
                (-4, 3, -9, 8),
            ):
                painter.drawLine(root_x, root_y, foot_x, foot_y)

        # Transparent wings sit behind the thorax. Airborne wingbeat is shown as
        # a tiny blur rather than huge flapping shapes.
        flap = math.sin(self._wing_phase)
        wing_shift = (1.5 * flap if self.airborne else 0.0)
        if self.micro_action == "wing_flick":
            wing_shift += 1.4 * flap

        painter.setPen(QPen(QColor(86, 76, 65, 95), 0.55))
        painter.setBrush(QBrush(QColor(214, 221, 216, 78)))
        painter.drawEllipse(
            QRectF(-4.8, -7.1 - wing_shift, 10.8, 5.3)
        )
        painter.drawEllipse(
            QRectF(-4.8, 1.8 + wing_shift, 10.8, 5.3)
        )
        painter.setPen(QPen(QColor(92, 81, 69, 80), 0.45))
        painter.drawLine(-3, -5, 4, -3)
        painter.drawLine(-3, 5, 4, 3)

        # Abdomen: elongated, dark-banded, tapered posterior.
        painter.setPen(QPen(QColor(49, 37, 25, 220), 0.65))
        painter.setBrush(QBrush(QColor(151, 113, 66, 245)))
        painter.drawEllipse(QRectF(-8.3, -3.1, 12.8, 6.2))
        painter.setPen(QPen(QColor(58, 42, 27, 190), 0.70))
        for x in (-5.2, -2.9, -0.6):
            painter.drawLine(int(round(x)), -3, int(round(x)), 3)

        # Thorax: compact golden-brown central mass.
        painter.setPen(QPen(QColor(58, 41, 27, 225), 0.65))
        painter.setBrush(QBrush(QColor(131, 91, 50, 250)))
        painter.drawEllipse(QRectF(1.0, -4.2, 8.5, 8.4))
        painter.setPen(QPen(QColor(191, 150, 92, 95), 0.45))
        painter.drawLine(4, -3, 5, 3)

        # Head shifts subtly with active head/antenna movements.
        head_offset = self.head_yaw * 1.7
        painter.setPen(QPen(QColor(58, 39, 26, 225), 0.60))
        painter.setBrush(QBrush(QColor(128, 84, 46, 250)))
        painter.drawEllipse(
            QRectF(8.0, -3.5 + head_offset, 6.0, 7.0)
        )

        # Red compound eyes dominate the head silhouette at tiny scale.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(151, 39, 31, 245)))
        painter.drawEllipse(
            QRectF(10.7, -3.0 + head_offset, 2.5, 2.8)
        )
        painter.drawEllipse(
            QRectF(10.7, 0.2 + head_offset, 2.5, 2.8)
        )
        painter.setBrush(QBrush(QColor(217, 92, 69, 150)))
        painter.drawEllipse(
            QRectF(11.4, -2.5 + head_offset, 0.7, 0.7)
        )
        painter.drawEllipse(
            QRectF(11.4, 0.7 + head_offset, 0.7, 0.7)
        )

        antenna_phase = math.sin(self._wing_phase * 0.55)
        if self.micro_action == "antenna_sweep":
            antenna_phase *= 2.0
        painter.setPen(QPen(QColor(56, 38, 25, 230), 0.60))
        painter.drawLine(
            13,
            int(round(-1.6 + head_offset)),
            16,
            int(round(-3.4 + antenna_phase)),
        )
        painter.drawLine(
            13,
            int(round(1.6 + head_offset)),
            16,
            int(round(3.4 - antenna_phase)),
        )

        if self.proboscis_extension > 0.03:
            painter.setPen(QPen(QColor(72, 45, 30, 235), 0.75))
            length = 1.0 + 3.3 * self.proboscis_extension
            painter.drawLine(
                13,
                int(round(head_offset)),
                int(round(13 + length)),
                int(round(head_offset + 0.7)),
            )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.context_requested.emit(
                event.globalPosition().toPoint()
            )
            event.accept()
            return
        super().mousePressEvent(event)


class BrainMapWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._layout: tuple[
            tuple[float, float], ...
        ] = ()
        self._active: tuple[
            tuple[float, float], ...
        ] = ()
        self.setMinimumHeight(230)

    def set_layout(self, points) -> None:
        self._layout = tuple(points)
        self.update()

    def set_active(self, points) -> None:
        self._active = tuple(points)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )
        area = self.rect().adjusted(
            8,
            8,
            -8,
            -8,
        )

        painter.setPen(
            QPen(QColor(48, 54, 63), 1)
        )
        painter.setBrush(
            QBrush(QColor(12, 15, 20))
        )
        painter.drawRoundedRect(
            area,
            12,
            12,
        )

        inset = area.adjusted(
            18,
            14,
            -18,
            -14,
        )
        painter.setPen(Qt.PenStyle.NoPen)

        painter.setBrush(
            QBrush(QColor(101, 114, 133, 62))
        )
        for x, y in self._layout:
            px = inset.left() + x * inset.width()
            py = inset.top() + y * inset.height()
            painter.drawEllipse(
                QPoint(int(px), int(py)),
                1,
                1,
            )

        painter.setBrush(
            QBrush(QColor(106, 226, 255, 220))
        )
        for x, y in self._active:
            px = inset.left() + x * inset.width()
            py = inset.top() + y * inset.height()
            painter.drawEllipse(
                QPoint(int(px), int(py)),
                2,
                2,
            )

        painter.setPen(
            QPen(QColor(122, 132, 146))
        )
        painter.setFont(
            QFont("Segoe UI", 8)
        )
        painter.drawText(
            area.adjusted(
                14,
                8,
                -14,
                -8,
            ),
            Qt.AlignmentFlag.AlignTop
            | Qt.AlignmentFlag.AlignRight,
            "MaleCNS · live activity",
        )


class FoodOverlay(QWidget):
    """Small physical sugar drop rendered on the desktop plane."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(38, 38)
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )

    def set_center(self, x: float, y: float) -> None:
        self.move(
            int(x - self.width() / 2),
            int(y - self.height() / 2),
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)

        p.setBrush(QColor(0, 0, 0, 50))
        p.drawEllipse(QRectF(7, 18, 26, 12))

        p.setBrush(QColor(245, 198, 78, 235))
        p.drawEllipse(QRectF(8, 8, 22, 22))
        p.setBrush(QColor(255, 232, 153, 220))
        p.drawEllipse(QRectF(12, 10, 7, 7))

        p.setBrush(QColor(255, 255, 255, 215))
        for x, y in ((5, 14), (29, 11), (31, 24), (8, 28)):
            p.drawEllipse(QRectF(x, y, 3, 3))


class FoodPlacementOverlay(QWidget):
    """Temporary full-desktop click target for placing food."""

    placed = Signal(object)
    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def begin(
        self,
        bounds: tuple[float, float, float, float],
    ) -> None:
        left, top, right, bottom = bounds
        self.setGeometry(
            int(left),
            int(top),
            max(1, int(right - left)),
            max(1, int(bottom - top)),
        )
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(5, 8, 12, 26))

        box = QRectF(
            self.width() / 2 - 175,
            28,
            350,
            46,
        )
        p.setPen(QPen(QColor(74, 91, 109), 1))
        p.setBrush(QColor(13, 17, 23, 235))
        p.drawRoundedRect(box, 12, 12)
        p.setPen(QColor(230, 237, 243))
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        p.drawText(
            box,
            Qt.AlignmentFlag.AlignCenter,
            "Click anywhere to place sugar  ·  Esc to cancel",
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            point = event.globalPosition().toPoint()
            self.hide()
            self.placed.emit(point)
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.hide()
            self.cancelled.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            self.cancelled.emit()
            event.accept()
            return
        super().keyPressEvent(event)



