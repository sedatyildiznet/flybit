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

from .care import CareModel
from .circadian import CircadianModel
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
from .state import load_state, normalize_display_name, save_state
from .vision import DesktopRetinaSampler


class BrainWorker(QObject):
    ready = Signal(object)
    layout = Signal(object)
    snapshot = Signal(object)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._target_center: float | None = None
        self._target_width = 0.035
        self._scene_luminance = None
        self._scene_azimuth = None
        self._sensory_dynamics = None
        self._timer: QTimer | None = None
        self._core: FlybitNeuralCore | None = None
        self._hunger_drive = 0.0
        self._vitality = 1.0
        self._activity_trait = 0.5
        self._boldness_trait = 0.5
        self._curiosity_trait = 0.5
        self._rest_drive = 0.0

    @Slot()
    def start(self) -> None:
        try:
            self._core = FlybitNeuralCore(
                device="auto",
                persistent_state=True,
            )
            self.ready.emit(
                {
                    "device": self._core.device,
                    "neurons": self._core.neuron_count,
                    "graded": self._core.graded_cell_count,
                    "looming_cells": self._core.looming_cell_count,
                    "restored": self._core.restored_state,
                }
            )
            self.layout.emit(
                self._core.brain_layout()
            )

            self._timer = QTimer(self)
            self._timer.setTimerType(
                Qt.TimerType.PreciseTimer
            )
            self._timer.setInterval(20)
            self._timer.timeout.connect(self._step)
            self._timer.start()
        except Exception as exc:
            self.failed.emit(
                f"{type(exc).__name__}: {exc}"
            )

    @Slot(float, float)
    def set_target(
        self,
        center: float,
        width: float,
    ) -> None:
        self._target_center = max(
            -1.0,
            min(1.0, float(center)),
        )
        self._target_width = max(
            0.008,
            min(0.75, float(width)),
        )

    @Slot(object, object, object)
    def set_scene(
        self,
        luminance,
        azimuth,
        dynamics,
    ) -> None:
        self._scene_luminance = luminance
        self._scene_azimuth = azimuth
        self._sensory_dynamics = dynamics

    @Slot(float, float, float, float, float, float)
    def set_homeostasis(
        self,
        hunger_drive: float,
        vitality: float,
        activity_trait: float,
        boldness_trait: float,
        curiosity_trait: float,
        rest_drive: float,
    ) -> None:
        self._hunger_drive = float(hunger_drive)
        self._vitality = float(vitality)
        self._activity_trait = float(activity_trait)
        self._boldness_trait = float(boldness_trait)
        self._curiosity_trait = float(curiosity_trait)
        self._rest_drive = float(rest_drive)

    @Slot()
    def persist(self) -> None:
        if self._core is not None:
            self._core.save_persistent_state()

    @Slot()
    def _step(self) -> None:
        if self._core is None:
            return
        try:
            self._core.set_homeostasis(
                self._hunger_drive,
                self._vitality,
                self._activity_trait,
                self._boldness_trait,
                self._curiosity_trait,
                self._rest_drive,
            )
            if self._sensory_dynamics is not None:
                self._core.set_visual_motion(
                    self._sensory_dynamics.retinal_loom_left,
                    self._sensory_dynamics.retinal_loom_right,
                )
            else:
                self._core.set_visual_motion(0.0, 0.0)
            if (
                self._scene_luminance is not None
                and self._scene_azimuth is not None
            ):
                snap = self._core.step_visual_luminance(
                    self._scene_luminance,
                    self._scene_azimuth,
                )
            else:
                snap = self._core.step_visual_target(
                    self._target_center,
                    self._target_width,
                )
            if snap.step % 250 == 0:
                self._core.save_persistent_state()
            self.snapshot.emit(snap)
        except Exception as exc:
            if self._timer:
                self._timer.stop()
            self.failed.emit(
                f"{type(exc).__name__}: {exc}"
            )


class FlyOverlay(QWidget):
    """Tiny always-on-top organism body.

    Drawing/wing animation is presentation only. Position and orientation come
    from the neural motor bridge.
    """

    clicked = Signal()
    context_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.heading = 0.0
        self.airborne = True
        self.drive = 0.0
        self.gait_phase = 0.0
        self.altitude = 0.0
        self._wing_phase = 0.0

        self.setFixedSize(48, 40)
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
    ) -> None:
        self.heading = heading
        self.airborne = airborne
        self.drive = max(0.0, min(1.0, drive))
        self.gait_phase = float(gait_phase) % 1.0
        self.altitude = max(0.0, float(altitude))
        self._wing_phase = (
            self._wing_phase
            + (0.55 if airborne else 0.08)
            + self.drive * 0.35
        ) % (2.0 * math.pi)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )
        painter.translate(
            self.width() / 2,
            self.height() / 2,
        )
        painter.rotate(
            math.degrees(self.heading)
        )

        # Shadow fades with virtual altitude while remaining on the same screen
        # plane. Altitude is depth, never monitor-Y gravity.
        shadow_alpha = max(
            18,
            min(65, int(65 - self.altitude * 0.40)),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(
            QBrush(QColor(0, 0, 0, shadow_alpha))
        )
        painter.drawEllipse(
            QRectF(-13, -5, 30, 15)
        )

        # Six legs use an alternating tripod gait while grounded. This is only
        # body rendering derived from biomechanics; it never selects movement.
        leg_pen = QPen(
            QColor(35, 28, 22, 235),
            1.35,
        )
        leg_pen.setCapStyle(
            Qt.PenCapStyle.RoundCap
        )
        painter.setPen(leg_pen)
        stride = (
            math.sin(self.gait_phase * 2.0 * math.pi)
            * 3.2
            * self.drive
            if not self.airborne
            else 0.0
        )
        for root_x, root_y, end_x, end_y, phase_sign in (
            (-5, -4, -15, -12, 1.0),
            (1, -5, -3, -16, -1.0),
            (7, -4, 17, -11, 1.0),
            (-5, 4, -15, 12, -1.0),
            (1, 5, -3, 16, 1.0),
            (7, 4, 17, 11, -1.0),
        ):
            painter.drawLine(
                root_x,
                root_y,
                int(round(end_x + phase_sign * stride)),
                end_y,
            )

        # Wings flutter visually while airborne. They do not move the body.
        flap = (
            math.sin(self._wing_phase) * 2.2
            if self.airborne
            else 0.0
        )
        painter.setPen(
            QPen(
                QColor(70, 65, 58, 115),
                0.9,
            )
        )
        painter.setBrush(
            QBrush(
                QColor(205, 213, 209, 92)
            )
        )
        painter.drawEllipse(
            QRectF(-7, -13 - flap, 18, 11)
        )
        painter.drawEllipse(
            QRectF(-7, 2 + flap, 18, 11)
        )

        # Abdomen.
        painter.setPen(
            QPen(QColor(27, 23, 19), 1)
        )
        painter.setBrush(
            QBrush(QColor(48, 39, 30))
        )
        painter.drawEllipse(
            QRectF(-10, -4.8, 18, 9.6)
        )

        # Thorax.
        painter.setBrush(
            QBrush(QColor(62, 49, 35))
        )
        painter.drawEllipse(
            QRectF(2, -6.4, 12.5, 12.8)
        )

        # Head and compound eyes.
        painter.setBrush(
            QBrush(QColor(55, 43, 31))
        )
        painter.drawEllipse(
            QRectF(11, -5.0, 8.5, 10.0)
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(
            QBrush(QColor(105, 42, 32))
        )
        painter.drawEllipse(
            QRectF(15.2, -4.0, 3.2, 3.7)
        )
        painter.drawEllipse(
            QRectF(15.2, 0.3, 3.2, 3.7)
        )

        # Antennae.
        painter.setPen(
            QPen(
                QColor(35, 28, 22),
                0.9,
            )
        )
        painter.drawLine(18, -2, 22, -5)
        painter.drawLine(18, 2, 22, 5)

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


class ControlPanel(QWidget):
    """Resizable native Windows control panel."""

    closed = Signal()
    feed_requested = Signal()
    name_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._logs = deque(maxlen=80)

        self.setWindowTitle("Flybit Neural Control")
        self.setWindowIcon(flybit_icon())
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setMinimumSize(560, 600)
        self.resize(720, 760)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        self.title = QLabel("FLYBIT")
        self.title.setObjectName("title")
        subtitle = QLabel("MaleCNS desktop organism")
        subtitle.setObjectName("muted")
        title_box.addWidget(self.title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.status = QLabel("● INITIALIZING")
        self.status.setObjectName("status")
        header.addWidget(self.status)
        outer.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        outer.addWidget(self.tabs, 1)

        # Overview
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        overview_layout.setContentsMargins(8, 12, 8, 8)
        overview_layout.setSpacing(12)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(8)
        metrics.setVerticalSpacing(8)
        self.neurons = self._metric_card("NEURONS", "—")
        self.graded = self._metric_card("GRADED VISUAL", "—")
        self.device = self._metric_card("DEVICE", "—")
        self.step = self._metric_card("NEURAL STEP", "—")
        metrics.addWidget(self.neurons, 0, 0)
        metrics.addWidget(self.graded, 0, 1)
        metrics.addWidget(self.device, 0, 2)
        metrics.addWidget(self.step, 0, 3)
        overview_layout.addLayout(metrics)

        self.neural_detail = QLabel(
            "retina —   lamina —   visual projection —"
        )
        self.neural_detail.setObjectName("muted")
        overview_layout.addWidget(self.neural_detail)

        motor_header = QHBoxLayout()
        motor_label = QLabel("DESCENDING MOTOR OUTPUT")
        motor_label.setObjectName("section")
        motor_header.addWidget(motor_label)
        motor_header.addStretch()
        self.body_state = QLabel("GROUND")
        self.body_state.setObjectName("pill")
        motor_header.addWidget(self.body_state)
        overview_layout.addLayout(motor_header)

        self.forward = self._motor_row(
            overview_layout, "Walking · DNg100 + DNa"
        )
        self.steer = self._motor_row(
            overview_layout, "Steering · DNa01/02"
        )
        self.escape = self._motor_row(
            overview_layout, "Escape · DNp01"
        )
        self.flight = self._motor_row(
            overview_layout, "Flight thrust · DNg02"
        )
        self.backward = self._motor_row(
            overview_layout, "Backward · MDN"
        )

        overview_note = QLabel(
            "Motor bars are short-window neural firing-rate decoders. "
            "No scripted wandering or mouse-near escape command is used."
        )
        overview_note.setWordWrap(True)
        overview_note.setObjectName("foot")
        overview_layout.addWidget(overview_note)
        overview_layout.addStretch()
        self.tabs.addTab(overview, "Overview")

        # Brain
        brain_tab = QWidget()
        brain_layout = QVBoxLayout(brain_tab)
        brain_layout.setContentsMargins(8, 12, 8, 8)
        brain_head = QHBoxLayout()
        label = QLabel("LIVE MALECNS MAP")
        label.setObjectName("section")
        brain_head.addWidget(label)
        brain_head.addStretch()
        self.vision_mode = QLabel("RAW DESKTOP VISION · 384×6")
        self.vision_mode.setObjectName("pill")
        brain_head.addWidget(self.vision_mode)
        brain_layout.addLayout(brain_head)
        self.brain_map = BrainMapWidget()
        brain_layout.addWidget(self.brain_map, 1)
        brain_note = QLabel(
            "Gray points are sampled EM neuron positions; cyan points are "
            "currently spiking neurons. Retina input comes from desktop "
            "luminance plus the hardware-cursor silhouette."
        )
        brain_note.setWordWrap(True)
        brain_note.setObjectName("foot")
        brain_layout.addWidget(brain_note)
        self.tabs.addTab(brain_tab, "Brain")

        # Care
        care_tab = QWidget()
        care_layout = QVBoxLayout(care_tab)
        care_layout.setContentsMargins(8, 12, 8, 8)
        care_layout.setSpacing(12)

        care_head = QHBoxLayout()
        care_title = QLabel("CARE & FEEDING")
        care_title.setObjectName("section")
        care_head.addWidget(care_title)
        care_head.addStretch()
        self.care_status = QLabel("STABLE")
        self.care_status.setObjectName("pill")
        care_head.addWidget(self.care_status)
        care_layout.addLayout(care_head)

        hunger_label = QLabel("HUNGER")
        hunger_label.setObjectName("muted")
        care_layout.addWidget(hunger_label)
        self.hunger = QProgressBar()
        self.hunger.setRange(0, 100)
        self.hunger.setValue(35)
        self.hunger.setFormat("%p%")
        care_layout.addWidget(self.hunger)

        self.feedings_label = QLabel("Feedings · 0")
        self.feedings_label.setObjectName("muted")
        care_layout.addWidget(self.feedings_label)

        self.last_feed_label = QLabel("Last feed · never")
        self.last_feed_label.setObjectName("muted")
        care_layout.addWidget(self.last_feed_label)

        self.odor_status = QLabel(
            "Odor field · no source · neural coupling disabled"
        )
        self.odor_status.setWordWrap(True)
        self.odor_status.setObjectName("muted")
        care_layout.addWidget(self.odor_status)

        self.feed_button = QPushButton("Place sugar…")
        self.feed_button.setObjectName("primary")
        self.feed_button.clicked.connect(
            lambda _checked=False: self.feed_requested.emit()
        )
        care_layout.addWidget(self.feed_button)

        self.taste_status = QLabel(
            "Taste mapping · nutrition/contact active; exact sweet-GRN "
            "neural injection disabled until receptor identity is present "
            "in the bundled MaleCNS metadata."
        )
        self.taste_status.setWordWrap(True)
        self.taste_status.setObjectName("foot")
        care_layout.addWidget(self.taste_status)

        care_note = QLabel(
            "The sugar drop is a real desktop-world object. Flybit must make "
            "physical contact with it to consume it. Hunger is persistent "
            "physiology and does not directly select movement."
        )
        care_note.setWordWrap(True)
        care_note.setObjectName("foot")
        care_layout.addWidget(care_note)
        care_layout.addStretch()
        self.tabs.addTab(care_tab, "Care")

        # Events
        events_tab = QWidget()
        events_layout = QVBoxLayout(events_tab)
        events_layout.setContentsMargins(8, 12, 8, 8)
        log_label = QLabel("RECENT NEURAL / BODY EVENTS")
        log_label.setObjectName("section")
        events_layout.addWidget(log_label)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(80)
        self.log.setObjectName("log")
        events_layout.addWidget(self.log, 1)
        self.tabs.addTab(events_tab, "Events")

        # Perception
        perception_tab = QWidget()
        perception_layout = QVBoxLayout(perception_tab)
        perception_layout.setContentsMargins(8, 12, 8, 8)
        perception_title = QLabel("SEMANTIC DESKTOP PERCEPTION")
        perception_title.setObjectName("section")
        perception_layout.addWidget(perception_title)
        self.perception_summary = QLabel(
            "Scanning cursor, windows, applications and native buttons…"
        )
        self.perception_summary.setWordWrap(True)
        self.perception_summary.setObjectName("muted")
        perception_layout.addWidget(self.perception_summary)
        self.sensory_summary = QLabel(
            "Temporal vision · waiting for retinal motion samples…"
        )
        self.sensory_summary.setWordWrap(True)
        self.sensory_summary.setObjectName("muted")
        perception_layout.addWidget(self.sensory_summary)
        self.perception_log = QPlainTextEdit()
        self.perception_log.setReadOnly(True)
        self.perception_log.setObjectName("log")
        perception_layout.addWidget(self.perception_log, 1)
        perception_note = QLabel(
            "Semantic labels are sensory context only. Chrome/button/window/"
            "cursor recognition never maps directly to a movement command."
        )
        perception_note.setWordWrap(True)
        perception_note.setObjectName("foot")
        perception_layout.addWidget(perception_note)
        self.tabs.addTab(perception_tab, "Perception")

        # Model boundary / provenance
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        model_layout.setContentsMargins(8, 12, 8, 8)
        model_title = QLabel("MODEL BOUNDARY & PROVENANCE")
        model_title.setObjectName("section")
        model_layout.addWidget(model_title)
        model_text = QPlainTextEdit()
        model_text.setReadOnly(True)
        model_text.setObjectName("log")
        model_text.setPlainText(
            "MEASURED / DATA-DRIVEN\n"
            "  • MaleCNS neuron identities and connection graph\n"
            "  • connection direction / synapse-derived weights\n"
            "  • transmitter-derived connection sign\n"
            "  • mapped photoreceptor identities and azimuths\n"
            "  • identified DN and LPLC2 cell types\n\n"
            "MODELED\n"
            "  • graded membrane constants / transfer functions\n"
            "  • raw-luminance temporal looming → LPLC2 transduction\n"
            "  • DN firing-rate → 2.5-D body decoder\n"
            "  • altitude/lift/gravity and tripod gait rendering\n"
            "  • hunger/metabolic/circadian/phenotype modulation\n\n"
            "SYNTHETIC WORLD / TELEMETRY\n"
            "  • desktop sugar object and nutrition collision\n"
            "  • semantic Chrome/window/button labels\n"
            "  • TTC/threat-salience/near-field observer metrics\n\n"
            "NOT CLAIMED\n"
            "  • complete biological brain emulation\n"
            "  • receptor-accurate taste/odor/mechanosensation\n"
            "  • complete muscles, hormones or synaptic plasticity"
        )
        model_layout.addWidget(model_text, 1)
        model_note = QLabel(
            "No modeled observer value may bypass the neural/body loop and "
            "become a direct movement command."
        )
        model_note.setWordWrap(True)
        model_note.setObjectName("foot")
        model_layout.addWidget(model_note)
        self.tabs.addTab(model_tab, "Model")

        # Life
        life_tab = QWidget()
        life_layout = QVBoxLayout(life_tab)
        life_layout.setContentsMargins(8, 12, 8, 8)
        life_layout.setSpacing(12)
        life_title = QLabel("LIFE HISTORY & PHENOTYPE")
        life_title.setObjectName("section")
        life_layout.addWidget(life_title)

        name_label = QLabel("ORGANISM NAME")
        name_label.setObjectName("muted")
        life_layout.addWidget(name_label)
        self.name_edit = QLineEdit()
        self.name_edit.setMaxLength(32)
        self.name_edit.setPlaceholderText("Flybit")
        self.name_edit.setToolTip(
            "The organism can only be named from this control panel."
        )
        self.name_edit.editingFinished.connect(
            lambda: self.name_changed.emit(self.name_edit.text())
        )
        life_layout.addWidget(self.name_edit)

        self.life_age = QLabel("Age · —")
        self.life_span = QLabel("Expected lifespan · —")
        self.life_sex = QLabel("Sex · Male")
        for widget in (self.life_age, self.life_span, self.life_sex):
            widget.setObjectName("muted")
            life_layout.addWidget(widget)
        self.life_progress = QProgressBar()
        self.life_progress.setRange(0, 1000)
        self.life_progress.setFormat("Life progress · %p%")
        life_layout.addWidget(self.life_progress)
        self.life_vitality = QProgressBar()
        self.life_vitality.setRange(0, 100)
        self.life_vitality.setFormat("Vitality · %p%")
        life_layout.addWidget(self.life_vitality)
        self.life_energy = QProgressBar()
        self.life_energy.setRange(0, 100)
        self.life_energy.setFormat("Metabolic energy · %p%")
        life_layout.addWidget(self.life_energy)
        self.life_traits = QLabel("Activity — · Boldness — · Curiosity —")
        self.life_traits.setWordWrap(True)
        self.life_traits.setObjectName("muted")
        life_layout.addWidget(self.life_traits)
        self.circadian_status = QLabel(
            "Circadian wake — · sleep pressure — · rest drive —"
        )
        self.circadian_status.setWordWrap(True)
        self.circadian_status.setObjectName("muted")
        life_layout.addWidget(self.circadian_status)
        self.biomechanics = QLabel(
            "Speed — · acceleration — · gait — · wingbeat —"
        )
        self.biomechanics.setWordWrap(True)
        self.biomechanics.setObjectName("muted")
        life_layout.addWidget(self.biomechanics)
        life_note = QLabel(
            "This is a persistent digital organism with a wall-clock birth, "
            "finite modeled lifespan, metabolism and individual phenotype. "
            "The MaleCNS connectome is biological data; the full organism is "
            "still a computational life model rather than a literal animal."
        )
        life_note.setWordWrap(True)
        life_note.setObjectName("foot")
        life_layout.addWidget(life_note)
        life_layout.addStretch()
        self.tabs.addTab(life_tab, "Life")

        self.setStyleSheet(
            """
            QWidget {
                background: #0d1117;
                color: #e8edf2;
                font-family: "Segoe UI";
            }
            QLabel { background: transparent; }
            #title {
                font-size: 24px;
                font-weight: 700;
                letter-spacing: 2px;
            }
            #muted, #foot {
                color: #8490a0;
                font-size: 10px;
            }
            #status {
                color: #72e6b1;
                font-size: 10px;
                font-weight: 700;
            }
            #section {
                color: #b8c3d1;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            #pill {
                color: #80dfff;
                background: #17212a;
                border: 1px solid #263746;
                border-radius: 8px;
                padding: 3px 8px;
                font-size: 9px;
                font-weight: 700;
            }
            QFrame#metric {
                background: #111821;
                border: 1px solid #28323e;
                border-radius: 10px;
            }
            QLabel#metricValue {
                color: #f5f8fb;
                font-size: 14px;
                font-weight: 700;
            }
            QLabel#metricName {
                color: #738092;
                font-size: 8px;
                font-weight: 700;
            }
            QTabWidget::pane {
                border: 1px solid #26303b;
                border-radius: 10px;
                background: #0d1117;
            }
            QTabBar::tab {
                background: #111821;
                color: #8996a6;
                padding: 8px 16px;
                margin-right: 3px;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
            }
            QTabBar::tab:selected {
                color: #eef5fa;
                background: #1a2430;
            }
            QProgressBar {
                background: #111821;
                border: 1px solid #28323e;
                border-radius: 5px;
                min-height: 10px;
                text-align: center;
                color: #dce7ef;
                font-size: 9px;
            }
            QProgressBar::chunk {
                background: #64d6ef;
                border-radius: 4px;
            }
            QLineEdit {
                background: #0b1016;
                border: 1px solid #2a3542;
                border-radius: 7px;
                color: #eef5fa;
                padding: 7px 9px;
                selection-background-color: #1f8fb0;
            }
            QLineEdit:focus {
                border-color: #4ab6d2;
            }
            #log {
                background: #090c10;
                border: 1px solid #252f3a;
                border-radius: 9px;
                color: #aeb9c6;
                font-family: "Consolas";
                font-size: 9px;
                padding: 8px;
            }
            QPushButton#primary {
                background: #1f8fb0;
                color: white;
                border: 0;
                border-radius: 8px;
                padding: 10px 14px;
                font-weight: 700;
            }
            QPushButton#primary:hover {
                background: #25a4c8;
            }
            """
        )

    @staticmethod
    def _metric_card(name: str, value: str) -> QFrame:
        card = QFrame()
        card.setObjectName("metric")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        name_label = QLabel(name)
        name_label.setObjectName("metricName")
        layout.addWidget(value_label)
        layout.addWidget(name_label)
        card.value_label = value_label
        return card

    @staticmethod
    def _motor_row(
        root: QVBoxLayout,
        label: str,
    ) -> QProgressBar:
        row = QHBoxLayout()
        text = QLabel(label)
        text.setFixedWidth(165)
        text.setObjectName("muted")
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        row.addWidget(text)
        row.addWidget(bar)
        root.addLayout(row)
        return bar

    def set_name(self, name: str) -> None:
        clean = normalize_display_name(name)
        self.title.setText(clean.upper())
        self.setWindowTitle(f"{clean} · Flybit Neural Control")
        if not self.name_edit.hasFocus():
            self.name_edit.setText(clean)

    def set_ready(self, info: dict) -> None:
        self.status.setText("● MALECNS ONLINE")
        self.neurons.value_label.setText(f"{int(info['neurons']):,}")
        self.graded.value_label.setText(f"{int(info['graded']):,}")
        self.device.value_label.setText(str(info["device"]).upper())

    def set_failed(self, message: str) -> None:
        self.status.setText("● NEURAL CORE ERROR")
        self.append_log(message)

    def update_snapshot(
        self,
        snap: NeuralSnapshot,
        *,
        airborne: bool,
    ) -> None:
        self.step.value_label.setText(f"{snap.step:,}")
        self.body_state.setText("FLIGHT" if airborne else "GROUND")
        self.neural_detail.setText(
            f"retina rms {snap.photoreceptor_rms:.3f}   "
            f"lamina rms {snap.lamina_rms:.3f}   "
            f"visual projection {snap.visual_projection_spikes:,} spikes"
        )
        self.brain_map.set_active(snap.active_brain_points)
        self.forward.setValue(min(100, int(snap.motor.forward * 100)))
        self.steer.setValue(
            min(100, int(abs(snap.motor.steering) * 100))
        )
        self.escape.setValue(min(100, int(snap.motor.escape * 100)))
        self.flight.setValue(min(100, int(snap.motor.flight * 100)))
        self.backward.setValue(min(100, int(snap.motor.backward * 100)))

    def update_care(
        self,
        *,
        hunger: float,
        feedings: int,
        last_feed: str | None,
        food_active: bool,
        odor=None,
    ) -> None:
        pct = int(max(0.0, min(1.0, hunger)) * 100)
        self.hunger.setValue(pct)
        if pct >= 80:
            self.care_status.setText("HUNGRY")
        elif pct >= 50:
            self.care_status.setText("READY TO EAT")
        else:
            self.care_status.setText("FED")
        self.feedings_label.setText(f"Feedings · {int(feedings)}")
        self.last_feed_label.setText(
            f"Last feed · {last_feed or 'never'}"
        )
        self.feed_button.setText(
            "Move sugar…"
            if food_active
            else "Place sugar…"
        )
        if odor is not None and odor.food_distance is not None:
            self.odor_status.setText(
                f"Modeled odor · L {odor.left:.3f} / R {odor.right:.3f} · "
                f"gradient {odor.gradient:+.3f} · salience {odor.salience:.3f} · "
                f"distance {odor.food_distance:.0f}px · neural coupling disabled"
            )
        else:
            self.odor_status.setText(
                "Odor field · no source · neural coupling disabled"
            )

    def update_sensory(self, dynamics) -> None:
        if dynamics is None:
            return
        ttc = (
            f"{dynamics.time_to_collision * 1000.0:.0f} ms"
            if dynamics.time_to_collision is not None
            else "—"
        )
        self.sensory_summary.setText(
            f"Cursor {dynamics.cursor_distance:.0f}px · "
            f"speed {dynamics.cursor_speed:.0f}px/s · "
            f"closing {dynamics.closing_speed:.0f}px/s · "
            f"loom {dynamics.looming_rate:.3f}rad/s · "
            f"TTC {ttc} · optic flow {dynamics.optic_flow:+.3f}rev/s · "
            f"loom L/R {dynamics.retinal_loom_left:.2f}/"
            f"{dynamics.retinal_loom_right:.2f} · "
            f"near-field {dynamics.mechanosensory_disturbance:.2f} · "
            f"observer salience {dynamics.threat_salience:.2f}"
        )

    def update_perception(
        self,
        objects: tuple[PerceivedObject, ...],
    ) -> None:
        counts: dict[str, int] = {}
        for obj in objects:
            counts[obj.kind] = counts.get(obj.kind, 0) + 1
        self.perception_summary.setText(
            " · ".join(
                f"{kind} {count}"
                for kind, count in sorted(counts.items())
            ) or "No semantic objects detected"
        )
        lines = []
        for obj in objects:
            lines.append(
                f"{obj.kind.upper():11s}  {obj.label[:42]:42s}  "
                f"{obj.confidence * 100:5.1f}%"
            )
        self.perception_log.setPlainText("\n".join(lines))

    def update_life(
        self,
        life: LifeSnapshot,
        biomechanics,
        circadian=None,
    ) -> None:
        if life.age_days >= 1.0:
            age_text = f"{life.age_days:.2f} days"
        else:
            age_text = f"{life.age_seconds / 3600.0:.2f} hours"
        self.life_age.setText(f"Age · {age_text}")
        self.life_span.setText(
            f"Expected lifespan · {life.lifespan_days:.1f} days · "
            f"remaining {life.remaining_days:.1f} days"
        )
        self.life_sex.setText(
            f"Sex · {life.sex} · {'ALIVE' if life.alive else 'LIFE ENDED'}"
        )
        self.life_progress.setValue(int(life.life_progress * 1000))
        self.life_vitality.setValue(int(life.vitality * 100))
        self.life_energy.setValue(int(life.energy * 100))
        self.life_traits.setText(
            f"Activity {life.activity:.2f} · Boldness {life.boldness:.2f} · "
            f"Curiosity {life.curiosity:.2f}"
        )
        self.biomechanics.setText(
            f"Speed {biomechanics.speed:.1f}px/s · "
            f"acceleration {biomechanics.acceleration:.1f}px/s² · "
            f"turn {biomechanics.turn_rate:.2f}rad/s · "
            f"gait {biomechanics.gait_phase:.2f} · "
            f"wingbeat {biomechanics.wingbeat_hz:.0f}Hz · "
            f"altitude {biomechanics.altitude:.1f} · "
            f"vertical {biomechanics.vertical_speed:+.1f}"
        )
        if circadian is not None:
            self.circadian_status.setText(
                f"Circadian wake {circadian.wake_drive:.2f} · "
                f"sleep pressure {circadian.sleep_pressure:.2f} · "
                f"rest drive {circadian.rest_drive:.2f} · "
                f"ambient {circadian.ambient_luminance:.2f}"
            )

    def append_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self._logs.append(f"{stamp}  {message}")
        self.log.setPlainText("\n".join(self._logs))
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event) -> None:  # noqa: N802
        event.ignore()
        self.hide()

    def hideEvent(self, event) -> None:  # noqa: N802
        self.closed.emit()
        super().hideEvent(event)


class FlybitWindow(QObject):
    """Application controller; only the organism is visible by default."""

    scene_changed = Signal(object, object, object)
    homeostasis_changed = Signal(float, float, float, float, float, float)
    persist_neural = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.app = QApplication.instance()
        self.state = load_state()
        self.care = CareModel(self.state)
        self.life = LifeModel(self.state)
        self.circadian = CircadianModel(self.state)
        self.olfaction = FoodOdorModel()
        self.semantic_scanner = DesktopSemanticScanner()
        self.nearby_objects: tuple[PerceivedObject, ...] = ()

        self.fly = FlyOverlay()
        self.fly.setWindowIcon(flybit_icon())
        self.panel = ControlPanel()
        self.panel.set_name(self.state.display_name)
        self.fly.setToolTip(
            f"{self.state.display_name} · click for neural control panel"
        )
        if (
            self.state.panel_w is not None
            and self.state.panel_h is not None
        ):
            self.panel.resize(
                max(560, int(self.state.panel_w)),
                max(600, int(self.state.panel_h)),
            )
        if (
            self.state.panel_x is not None
            and self.state.panel_y is not None
        ):
            self.panel.move(
                int(self.state.panel_x),
                int(self.state.panel_y),
            )
        self.food_overlay = FoodOverlay()
        self.food_placement = FoodPlacementOverlay()
        self.vision = DesktopRetinaSampler()
        self.surfaces = []

        bounds = self._desktop_bounds()
        left, top, right, bottom = bounds

        start_x = (
            float(self.state.x)
            if self.state.x is not None
            else left + (right - left) * 0.55
        )
        start_y = (
            float(self.state.y)
            if self.state.y is not None
            else top + (bottom - top) * 0.45
        )
        start_x = min(right - 70.0, max(left + 70.0, start_x))
        start_y = min(bottom - 70.0, max(top + 70.0, start_y))

        self.kinematics = FlyKinematics(
            FlyBodyState(
                x=start_x,
                y=start_y,
                heading=float(self.state.heading),
            )
        )
        self.latest_motor = MotorActivity()
        self.latest_snapshot: NeuralSnapshot | None = None
        self.latest_sensory = None
        self._last_log: dict[str, float] = {}

        self._brain_thread = QThread(self)
        self._worker = BrainWorker()
        self._worker.moveToThread(self._brain_thread)
        self._brain_thread.started.connect(self._worker.start)
        self.scene_changed.connect(self._worker.set_scene)
        self.homeostasis_changed.connect(self._worker.set_homeostasis)
        self.persist_neural.connect(self._worker.persist)
        self._worker.ready.connect(self._on_ready)
        self._worker.layout.connect(self.panel.brain_map.set_layout)
        self._worker.snapshot.connect(self._on_snapshot)
        self._worker.failed.connect(self._on_failed)
        self._brain_thread.start()

        self.fly.clicked.connect(self.toggle_panel)
        self.fly.context_requested.connect(self._context_menu)
        self.panel.feed_requested.connect(
            self._begin_food_placement
        )
        self.panel.name_changed.connect(
            self._rename_organism
        )
        self.food_placement.placed.connect(
            self._place_food_at
        )

        self._physics_timer = QTimer(self)
        self._physics_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._physics_timer.setInterval(20)
        self._physics_timer.timeout.connect(self._tick)
        self._physics_timer.start()

        self._vision_timer = QTimer(self)
        self._vision_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._vision_timer.setInterval(20)
        self._vision_timer.timeout.connect(self._capture_scene)
        self._vision_timer.start()
        self._capture_scene()

        self._care_timer = QTimer(self)
        self._care_timer.setInterval(1000)
        self._care_timer.timeout.connect(self._refresh_care)
        self._care_timer.start()

        self._perception_timer = QTimer(self)
        self._perception_timer.setInterval(500)
        self._perception_timer.timeout.connect(self._refresh_perception)
        self._perception_timer.start()

        self._save_timer = QTimer(self)
        self._save_timer.setInterval(2000)
        self._save_timer.timeout.connect(self._save_position)
        self._save_timer.start()

        if self.app is not None:
            self.app.aboutToQuit.connect(self.shutdown)

    def show(self) -> None:
        self._position_overlay()
        self.fly.show()
        self._refresh_care()

    def _desktop_bounds(
        self,
    ) -> tuple[float, float, float, float]:
        screens = QApplication.screens()
        if not screens:
            return (0.0, 0.0, 1920.0, 1080.0)
        geoms = [screen.geometry() for screen in screens]
        left = min(rect.left() for rect in geoms)
        top = min(rect.top() for rect in geoms)
        right = max(rect.left() + rect.width() for rect in geoms)
        bottom = max(rect.top() + rect.height() for rect in geoms)
        return float(left), float(top), float(right), float(bottom)

    @Slot()
    def _capture_scene(self) -> None:
        body = self.kinematics.state
        food = self.care.food
        food_point = (
            (food.x, food.y, food.radius)
            if food is not None
            else None
        )
        cursor = QCursor.pos()
        luminance, dynamics = self.vision.sample_with_dynamics(
            x=body.x,
            y=body.y,
            heading=body.heading,
            cursor=cursor,
            food=food_point,
        )
        self.latest_sensory = dynamics
        if self.panel.isVisible() and dynamics is not None:
            self.panel.update_sensory(dynamics)
        self.scene_changed.emit(
            luminance,
            self.vision.azimuth,
            dynamics,
        )

    @Slot()
    def _tick(self) -> None:
        self.care.tick(0.020)
        life = self.life.snapshot()
        motor_load = self.kinematics.biomechanics().locomotor_load
        self.life.tick(
            0.020,
            motor_load=motor_load,
            hunger=self.state.hunger,
        )
        life = self.life.snapshot()
        ambient = (
            self.latest_sensory.ambient_luminance
            if self.latest_sensory is not None
            else 0.5
        )
        self.circadian.tick(
            0.020,
            motor_load=motor_load,
            ambient_luminance=ambient,
        )
        circadian = self.circadian.snapshot()
        self.homeostasis_changed.emit(
            self.care.homeostatic_drive,
            life.vitality,
            life.activity,
            life.boldness,
            life.curiosity,
            circadian.rest_drive,
        )
        bounds = self._desktop_bounds()
        physiology_gain = life.vitality * (0.78 + 0.30 * life.activity)
        if not life.alive:
            physiology_gain = 0.0
        events = self.kinematics.update(
            self.latest_motor if life.alive else MotorActivity(),
            self.surfaces,
            bounds,
            dt=0.020,
            physiology_gain=physiology_gain,
        )

        for event in events:
            if event.kind == "land":
                self.panel.append_log(
                    f"flight settled · {event.detail}"
                )
            elif event.kind == "takeoff":
                self.panel.append_log(
                    "DNp01 → planar flight burst"
                )

        body = self.kinematics.state
        bio = self.kinematics.biomechanics()
        self.fly.set_pose(
            heading=body.heading,
            airborne=body.airborne,
            drive=max(
                self.latest_motor.forward,
                self.latest_motor.escape,
                self.latest_motor.flight,
            ),
            gait_phase=bio.gait_phase,
            altitude=bio.altitude,
        )
        self._position_overlay()

        if self.care.contact(body.x, body.y):
            self.life.feed()
            self.food_overlay.hide()
            self.panel.append_log(
                "sugar contact → consumed · nutrition state updated"
            )
            save_state(self.state)
            self._refresh_care()

    def _position_overlay(self) -> None:
        body = self.kinematics.state
        self.fly.move(
            int(body.x - self.fly.width() / 2),
            int(body.y - self.fly.height() / 2),
        )

    @Slot()
    def _begin_food_placement(self) -> None:
        self.panel.hide()
        self.food_placement.begin(
            self._desktop_bounds()
        )

    @Slot(object)
    def _place_food_at(self, point: QPoint) -> None:
        bounds = self._desktop_bounds()
        x = min(
            bounds[2] - 24.0,
            max(bounds[0] + 24.0, float(point.x())),
        )
        y = min(
            bounds[3] - 24.0,
            max(bounds[1] + 24.0, float(point.y())),
        )
        food = self.care.place_food(x, y)
        self.food_overlay.set_center(food.x, food.y)
        self.food_overlay.show()
        self.food_overlay.raise_()
        self.panel.append_log(
            f"sugar placed · ({int(food.x)}, {int(food.y)})"
        )
        self._refresh_care()
        self._capture_scene()

    @Slot(str)
    def _rename_organism(self, name: str) -> None:
        clean = normalize_display_name(name)
        if clean == self.state.display_name:
            self.panel.set_name(clean)
            return
        self.state.display_name = clean
        self.panel.set_name(clean)
        self.fly.setToolTip(
            f"{clean} · click for neural control panel"
        )
        self.panel.append_log(f"organism renamed · {clean}")
        save_state(self.state)

    @Slot()
    def _refresh_care(self) -> None:
        body = self.kinematics.state
        food = self.care.food
        odor_food = (
            (food.x, food.y, food.amount)
            if food is not None
            else None
        )
        odor = self.olfaction.sample(
            x=body.x,
            y=body.y,
            heading=body.heading,
            food=odor_food,
            hunger_drive=self.care.homeostatic_drive,
        )
        self.panel.update_care(
            hunger=self.state.hunger,
            feedings=self.state.feedings,
            last_feed=self.state.last_feed_at,
            food_active=food is not None,
            odor=odor,
        )
        self.panel.update_life(
            self.life.snapshot(),
            self.kinematics.biomechanics(),
            self.circadian.snapshot(),
        )

    @Slot()
    def _refresh_perception(self) -> None:
        cursor = QCursor.pos()
        objects = self.semantic_scanner.scan(
            (float(cursor.x()), float(cursor.y()))
        )
        body = self.kinematics.state
        self.nearby_objects = self.semantic_scanner.nearest(
            objects,
            body.x,
            body.y,
            limit=12,
        )
        self.panel.update_perception(self.nearby_objects)

    @Slot(object)
    def _on_ready(self, info: dict) -> None:
        self.panel.set_ready(info)
        self.panel.append_log("MaleCNS neural core online")
        if bool(info.get("restored")):
            self.panel.append_log(
                "persistent neural membrane/adaptation state restored"
            )
        self.panel.append_log(
            "R1–R8 / L1–L3 graded vision active"
        )
        self.panel.append_log(
            "raw desktop panorama online · 384 angular bins"
        )
        self.panel.append_log(
            "modeled retinal looming → LPLC2 · "
            f"{int(info.get('looming_cells', 0))} cells"
        )
        self._refresh_care()

    @Slot(object)
    def _on_snapshot(self, snap: NeuralSnapshot) -> None:
        self.latest_snapshot = snap
        self.latest_motor = snap.motor

        if self.panel.isVisible():
            self.panel.update_snapshot(
                snap,
                airborne=self.kinematics.state.airborne,
            )

        now = time.monotonic()
        motor = snap.motor
        if (
            motor.escape > 0.0
            and now - self._last_log.get("escape", 0.0) > 0.35
        ):
            self.panel.append_log(
                "DNp01 escape · "
                f"L {motor.escape_left:.2f} R {motor.escape_right:.2f}"
            )
            self._last_log["escape"] = now

        if (
            abs(motor.steering) > 0.0
            and now - self._last_log.get("steer", 0.0) > 0.55
        ):
            direction = "right" if motor.steering > 0.0 else "left"
            self.panel.append_log(
                f"DNa steering · {direction} "
                f"{abs(motor.steering):.2f}"
            )
            self._last_log["steer"] = now

        if (
            motor.forward > 0.0
            and now - self._last_log.get("forward", 0.0) > 0.8
        ):
            self.panel.append_log(
                f"walking drive · {motor.forward:.2f}"
            )
            self._last_log["forward"] = now

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self.panel.set_failed(message)
        self.panel.show()
        self.panel.raise_()

    @Slot()
    def toggle_panel(self) -> None:
        if self.panel.isVisible():
            self.panel.hide()
            return

        if self.latest_snapshot is not None:
            self.panel.update_snapshot(
                self.latest_snapshot,
                airborne=self.kinematics.state.airborne,
            )
        if self.latest_sensory is not None:
            self.panel.update_sensory(self.latest_sensory)
        self.panel.set_name(self.state.display_name)
        self._refresh_care()

        bounds = self._desktop_bounds()
        body = self.kinematics.state
        if (
            self.state.panel_x is None
            or self.state.panel_y is None
        ):
            x = min(
                bounds[2] - self.panel.width() - 18,
                max(bounds[0] + 18, body.x + 35),
            )
            y = min(
                bounds[3] - self.panel.height() - 18,
                max(bounds[1] + 18, body.y - 80),
            )
            self.panel.move(int(x), int(y))
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()

    @Slot(object)
    def _context_menu(self, point: QPoint) -> None:
        menu = QMenu()
        menu.setWindowIcon(flybit_icon())
        open_action = menu.addAction("Open neural control panel")
        feed_action = menu.addAction("Place sugar…")
        menu.addSeparator()
        exit_action = menu.addAction("Exit Flybit")
        chosen = menu.exec(point)
        if chosen is open_action:
            if not self.panel.isVisible():
                self.toggle_panel()
        elif chosen is feed_action:
            self._begin_food_placement()
        elif chosen is exit_action:
            QApplication.quit()

    @Slot()
    def _save_position(self) -> None:
        body = self.kinematics.state
        self.state.x = float(body.x)
        self.state.y = float(body.y)
        self.state.heading = float(body.heading)
        geom = self.panel.geometry()
        self.state.panel_x = int(geom.x())
        self.state.panel_y = int(geom.y())
        self.state.panel_w = int(geom.width())
        self.state.panel_h = int(geom.height())
        save_state(self.state)

    @Slot()
    def shutdown(self) -> None:
        self._save_position()
        self._physics_timer.stop()
        self._vision_timer.stop()
        self._care_timer.stop()
        self._perception_timer.stop()
        self._save_timer.stop()
        self.food_overlay.hide()
        self.food_placement.hide()
        self.persist_neural.emit()
        if self._brain_thread.isRunning():
            self._brain_thread.quit()
            self._brain_thread.wait(2500)
