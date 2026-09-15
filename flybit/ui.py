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
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .motion import (
    FlyBodyState,
    FlyKinematics,
    MotorActivity,
)
from .neural import FlybitNeuralCore, NeuralSnapshot
from .state import load_state, save_state
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
        self._timer: QTimer | None = None
        self._core: FlybitNeuralCore | None = None

    @Slot()
    def start(self) -> None:
        try:
            self._core = FlybitNeuralCore(
                device="auto"
            )
            self.ready.emit(
                {
                    "device": self._core.device,
                    "neurons": self._core.neuron_count,
                    "graded": self._core.graded_cell_count,
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

    @Slot(object, object)
    def set_scene(
        self,
        luminance,
        azimuth,
    ) -> None:
        self._scene_luminance = luminance
        self._scene_azimuth = azimuth

    @Slot()
    def _step(self) -> None:
        if self._core is None:
            return
        try:
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
    ) -> None:
        self.heading = heading
        self.airborne = airborne
        self.drive = max(0.0, min(1.0, drive))
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

        # Soft shadow gives the 40 px body separation from bright windows.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(
            QBrush(QColor(0, 0, 0, 65))
        )
        painter.drawEllipse(
            QRectF(-13, -5, 30, 15)
        )

        # Six legs.
        leg_pen = QPen(
            QColor(35, 28, 22, 235),
            1.35,
        )
        leg_pen.setCapStyle(
            Qt.PenCapStyle.RoundCap
        )
        painter.setPen(leg_pen)
        for root_x, root_y, end_x, end_y in (
            (-5, -4, -15, -12),
            (1, -5, -3, -16),
            (7, -4, 17, -11),
            (-5, 4, -15, 12),
            (1, 5, -3, 16),
            (7, 4, 17, 11),
        ):
            painter.drawLine(
                root_x,
                root_y,
                end_x,
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


class ControlPanel(QWidget):
    closed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._logs = deque(maxlen=40)
        self._drag_offset: QPoint | None = None

        self.setWindowTitle(
            "Flybit Neural Control"
        )
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setMinimumSize(500, 520)
        self.resize(650, 760)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        shell = QFrame()
        shell.setObjectName("shell")
        outer.addWidget(shell)

        root = QVBoxLayout(shell)
        root.setContentsMargins(
            22,
            18,
            22,
            18,
        )
        root.setSpacing(12)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("FLYBIT")
        title.setObjectName("title")
        subtitle = QLabel(
            "MaleCNS desktop organism"
        )
        subtitle.setObjectName("muted")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()

        self.status = QLabel("● INITIALIZING")
        self.status.setObjectName("status")
        header.addWidget(self.status)

        close_button = QPushButton("×")
        close_button.setObjectName("close")
        close_button.setFixedSize(34, 34)
        close_button.clicked.connect(self.hide)
        header.addWidget(close_button)
        root.addLayout(header)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(8)
        metrics.setVerticalSpacing(8)

        self.neurons = self._metric_card(
            "NEURONS",
            "—",
        )
        self.graded = self._metric_card(
            "GRADED VISUAL",
            "—",
        )
        self.device = self._metric_card(
            "DEVICE",
            "—",
        )
        self.step = self._metric_card(
            "NEURAL STEP",
            "—",
        )
        metrics.addWidget(self.neurons, 0, 0)
        metrics.addWidget(self.graded, 0, 1)
        metrics.addWidget(self.device, 0, 2)
        metrics.addWidget(self.step, 0, 3)
        root.addLayout(metrics)

        section = QLabel("LIVE BRAIN MAP")
        section.setObjectName("section")
        root.addWidget(section)

        self.brain_map = BrainMapWidget()
        root.addWidget(self.brain_map)

        self.neural_detail = QLabel(
            "retina —   lamina —   visual projection —"
        )
        self.neural_detail.setObjectName("muted")
        root.addWidget(self.neural_detail)

        motor_header = QHBoxLayout()
        motor_label = QLabel("DESCENDING MOTOR OUTPUT")
        motor_label.setObjectName("section")
        motor_header.addWidget(motor_label)
        motor_header.addStretch()
        self.body_state = QLabel("AIRBORNE")
        self.body_state.setObjectName("pill")
        motor_header.addWidget(self.body_state)
        root.addLayout(motor_header)

        self.forward = self._motor_row(
            root,
            "DNg100 · forward",
        )
        self.steer = self._motor_row(
            root,
            "DNa02 · steering",
        )
        self.escape = self._motor_row(
            root,
            "DNp01 · escape",
        )
        self.backward = self._motor_row(
            root,
            "MDN · backward",
        )

        log_header = QHBoxLayout()
        log_label = QLabel("RECENT NEURAL / BODY EVENTS")
        log_label.setObjectName("section")
        log_header.addWidget(log_label)
        log_header.addStretch()
        root.addLayout(log_header)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(40)
        self.log.setFixedHeight(118)
        self.log.setObjectName("log")
        root.addWidget(self.log)

        self.footer = QLabel(
            "Visual input is sampled from raw desktop pixels plus the cursor "
            "silhouette. Movement is decoded from identified MaleCNS descending "
            "neurons on a flat 2-D desktop plane; no mouse-distance rule selects "
            "an action."
        )
        self.footer.setWordWrap(True)
        self.footer.setObjectName("foot")
        root.addWidget(self.footer)

        self.setStyleSheet(
            """
            #shell {
                background: rgba(12, 15, 20, 248);
                border: 1px solid #303642;
                border-radius: 18px;
            }
            QLabel {
                color: #e9edf2;
                font-family: "Segoe UI";
            }
            #title {
                font-size: 24px;
                font-weight: 700;
                letter-spacing: 2px;
            }
            #muted, #foot {
                color: #7f8998;
                font-size: 10px;
            }
            #status {
                color: #72e6b1;
                font-size: 10px;
                font-weight: 700;
            }
            #section {
                color: #aeb8c6;
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
                background: #11161d;
                border: 1px solid #252c36;
                border-radius: 10px;
            }
            QLabel#metricValue {
                color: #f4f7fa;
                font-size: 14px;
                font-weight: 700;
            }
            QLabel#metricName {
                color: #6f7a89;
                font-size: 8px;
                font-weight: 700;
            }
            QProgressBar {
                background: #11161d;
                border: 1px solid #252c36;
                border-radius: 5px;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #6ad7f2;
                border-radius: 4px;
            }
            #log {
                background: #0b0e12;
                border: 1px solid #252c36;
                border-radius: 10px;
                color: #aeb8c6;
                font-family: "Consolas";
                font-size: 9px;
                padding: 8px;
            }
            #close {
                color: #8d97a5;
                background: transparent;
                border: 0;
                font-size: 22px;
            }
            #close:hover {
                color: white;
                background: #20262f;
                border-radius: 8px;
            }
            """
        )

    @staticmethod
    def _metric_card(
        name: str,
        value: str,
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("metric")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        value_label.setProperty(
            "metric_value",
            True,
        )
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
        text.setFixedWidth(135)
        text.setObjectName("muted")
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        row.addWidget(text)
        row.addWidget(bar)
        root.addLayout(row)
        return bar

    def set_ready(self, info: dict) -> None:
        self.status.setText("● MALECNS ONLINE")
        self.neurons.value_label.setText(
            f"{int(info['neurons']):,}"
        )
        self.graded.value_label.setText(
            f"{int(info['graded']):,}"
        )
        self.device.value_label.setText(
            str(info["device"]).upper()
        )

    def set_failed(self, message: str) -> None:
        self.status.setText("● NEURAL CORE ERROR")
        self.append_log(message)

    def update_snapshot(
        self,
        snap: NeuralSnapshot,
        *,
        airborne: bool,
    ) -> None:
        self.step.value_label.setText(
            f"{snap.step:,}"
        )
        self.body_state.setText(
            "AIRBORNE"
            if airborne
            else "LANDED"
        )
        self.neural_detail.setText(
            f"retina rms {snap.photoreceptor_rms:.3f}   "
            f"lamina rms {snap.lamina_rms:.3f}   "
            f"visual projection {snap.visual_projection_spikes:,} spikes"
        )
        self.brain_map.set_active(
            snap.active_brain_points
        )

        self.forward.setValue(
            min(
                100,
                int(snap.motor.forward * 100),
            )
        )
        self.steer.setValue(
            min(
                100,
                int(
                    abs(
                        snap.motor.steering
                    )
                    * 100
                ),
            )
        )
        self.escape.setValue(
            min(
                100,
                int(snap.motor.escape * 100),
            )
        )
        self.backward.setValue(
            min(
                100,
                int(snap.motor.backward * 100),
            )
        )

    def append_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"{stamp}  {message}"
        self._logs.append(line)
        self.log.setPlainText(
            "\n".join(self._logs)
        )
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(
            scrollbar.maximum()
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.position().y() <= 62
        ):
            self._drag_offset = (
                event.globalPosition().toPoint()
                - self.frameGeometry().topLeft()
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if (
            self._drag_offset is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(
                event.globalPosition().toPoint()
                - self._drag_offset
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        self.closed.emit()
        super().hideEvent(event)


class FlybitWindow(QObject):
    """Application controller; only the fly is visible by default."""

    scene_changed = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()
        self.app = QApplication.instance()
        self.state = load_state()

        self.fly = FlyOverlay()
        self.panel = ControlPanel()
        self.vision = DesktopRetinaSampler()
        self.surfaces = []

        bounds = self._desktop_bounds()
        left, top, right, bottom = bounds

        start_x = (
            float(self.state.x)
            if self.state.x is not None
            else left + (right - left) * 0.58
        )
        start_y = (
            float(self.state.y)
            if self.state.y is not None
            else top + 105.0
        )

        self.kinematics = FlyKinematics(
            FlyBodyState(
                x=start_x,
                y=start_y,
                heading=float(
                    self.state.heading
                ),
            )
        )
        self.latest_motor = MotorActivity()
        self.latest_snapshot: (
            NeuralSnapshot | None
        ) = None
        self._last_log: dict[str, float] = {}

        self._brain_thread = QThread(self)
        self._worker = BrainWorker()
        self._worker.moveToThread(
            self._brain_thread
        )
        self._brain_thread.started.connect(
            self._worker.start
        )
        self.scene_changed.connect(
            self._worker.set_scene
        )
        self._worker.ready.connect(
            self._on_ready
        )
        self._worker.layout.connect(
            self.panel.brain_map.set_layout
        )
        self._worker.snapshot.connect(
            self._on_snapshot
        )
        self._worker.failed.connect(
            self._on_failed
        )
        self._brain_thread.start()

        self.fly.clicked.connect(
            self.toggle_panel
        )
        self.fly.context_requested.connect(
            self._context_menu
        )

        self._physics_timer = QTimer(self)
        self._physics_timer.setTimerType(
            Qt.TimerType.PreciseTimer
        )
        self._physics_timer.setInterval(20)
        self._physics_timer.timeout.connect(
            self._tick
        )
        self._physics_timer.start()

        self._vision_timer = QTimer(self)
        self._vision_timer.setInterval(80)
        self._vision_timer.timeout.connect(
            self._capture_scene
        )
        self._vision_timer.start()
        self._capture_scene()

        self._save_timer = QTimer(self)
        self._save_timer.setInterval(2000)
        self._save_timer.timeout.connect(
            self._save_position
        )
        self._save_timer.start()

        if self.app is not None:
            self.app.aboutToQuit.connect(
                self.shutdown
            )

    def show(self) -> None:
        self._position_overlay()
        self.fly.show()

    def _desktop_bounds(
        self,
    ) -> tuple[float, float, float, float]:
        screens = QApplication.screens()
        if not screens:
            return (0.0, 0.0, 1920.0, 1080.0)

        geoms = [
            screen.geometry()
            for screen in screens
        ]
        left = min(rect.left() for rect in geoms)
        top = min(rect.top() for rect in geoms)
        right = max(
            rect.left() + rect.width()
            for rect in geoms
        )
        bottom = max(
            rect.top() + rect.height()
            for rect in geoms
        )
        return (
            float(left),
            float(top),
            float(right),
            float(bottom),
        )

    @Slot()
    def _capture_scene(self) -> None:
        body = self.kinematics.state
        luminance = self.vision.sample(
            x=body.x,
            y=body.y,
            heading=body.heading,
            cursor=QCursor.pos(),
        )
        self.scene_changed.emit(
            luminance,
            self.vision.azimuth,
        )

    @Slot()
    def _tick(self) -> None:
        bounds = self._desktop_bounds()
        events = self.kinematics.update(
            self.latest_motor,
            self.surfaces,
            bounds,
            dt=0.020,
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
            elif event.kind == "fall":
                self.panel.append_log(
                    f"support lost · {event.detail}"
                )

        body = self.kinematics.state
        self.fly.set_pose(
            heading=body.heading,
            airborne=body.airborne,
            drive=max(
                self.latest_motor.forward,
                self.latest_motor.escape,
            ),
        )
        self._position_overlay()

    def _position_overlay(self) -> None:
        body = self.kinematics.state
        self.fly.move(
            int(
                body.x
                - self.fly.width() / 2
            ),
            int(
                body.y
                - self.fly.height() / 2
            ),
        )

    @Slot(object)
    def _on_ready(self, info: dict) -> None:
        self.panel.set_ready(info)
        self.panel.append_log(
            "MaleCNS neural core online"
        )
        self.panel.append_log(
            "R1–R8 / L1–L3 graded vision active"
        )
        self.panel.append_log(
            "raw desktop panorama online · 384 angular bins"
        )

    @Slot(object)
    def _on_snapshot(
        self,
        snap: NeuralSnapshot,
    ) -> None:
        self.latest_snapshot = snap
        self.latest_motor = snap.motor

        if self.panel.isVisible():
            self.panel.update_snapshot(
                snap,
                airborne=(
                    self.kinematics.state.airborne
                ),
            )

        now = time.monotonic()
        motor = snap.motor

        if (
            motor.escape > 0.0
            and now
            - self._last_log.get(
                "escape",
                0.0,
            )
            > 0.35
        ):
            self.panel.append_log(
                "DNp01 escape firing · "
                f"L {motor.escape_left:.2f} "
                f"R {motor.escape_right:.2f}"
            )
            self._last_log["escape"] = now

        if (
            abs(motor.steering) > 0.0
            and now
            - self._last_log.get(
                "steer",
                0.0,
            )
            > 0.55
        ):
            direction = (
                "right"
                if motor.steering > 0.0
                else "left"
            )
            self.panel.append_log(
                "DNa02 steering · "
                f"{direction} "
                f"{abs(motor.steering):.2f}"
            )
            self._last_log["steer"] = now

        if (
            motor.forward > 0.0
            and now
            - self._last_log.get(
                "forward",
                0.0,
            )
            > 0.8
        ):
            self.panel.append_log(
                "DNg100 locomotor output · "
                f"{motor.forward:.2f}"
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
                airborne=(
                    self.kinematics.state.airborne
                ),
            )

        bounds = self._desktop_bounds()
        _left, top, right, bottom = bounds
        body = self.kinematics.state

        x = min(
            right - self.panel.width() - 18,
            max(
                bounds[0] + 18,
                body.x + 35,
            ),
        )
        y = min(
            bottom - self.panel.height() - 18,
            max(
                top + 18,
                body.y - 80,
            ),
        )
        self.panel.move(
            int(x),
            int(y),
        )
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()

    @Slot(object)
    def _context_menu(self, point: QPoint) -> None:
        menu = QMenu()
        open_action = menu.addAction(
            "Open neural control panel"
        )
        menu.addSeparator()
        exit_action = menu.addAction(
            "Exit Flybit"
        )
        chosen = menu.exec(point)
        if chosen is open_action:
            if not self.panel.isVisible():
                self.toggle_panel()
        elif chosen is exit_action:
            QApplication.quit()

    @Slot()
    def _save_position(self) -> None:
        body = self.kinematics.state
        self.state.x = float(body.x)
        self.state.y = float(body.y)
        self.state.heading = float(
            body.heading
        )
        save_state(self.state)

    @Slot()
    def shutdown(self) -> None:
        self._save_position()
        self._physics_timer.stop()
        self._vision_timer.stop()
        self._save_timer.stop()

        if self._brain_thread.isRunning():
            self._brain_thread.quit()
            self._brain_thread.wait(2500)
