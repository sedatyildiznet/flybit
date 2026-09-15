"""Windows desktop shell for the Flybit neural core."""
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QCursor, QPainter, QPen, QBrush, QColor
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from .neural import FlybitNeuralCore, NeuralSnapshot
from .state import load_state


class BrainWorker(QObject):
    ready = Signal(str, int)
    snapshot = Signal(object)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._target: float | None = None
        self._timer: QTimer | None = None
        self._core: FlybitNeuralCore | None = None

    @Slot()
    def start(self) -> None:
        try:
            self._core = FlybitNeuralCore(device="auto")
            self._timer = QTimer(self)
            self._timer.setTimerType(Qt.TimerType.PreciseTimer)
            self._timer.setInterval(20)
            self._timer.timeout.connect(self._step)
            self._timer.start()
            self.ready.emit(self._core.device, self._core.neuron_count)
        except Exception as exc:  # first-run download/build errors must be visible
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    @Slot(float)
    def set_target(self, target: float) -> None:
        self._target = max(-1.0, min(1.0, float(target)))

    @Slot()
    def _step(self) -> None:
        if self._core is None:
            return
        try:
            self.snapshot.emit(self._core.step_visual_target(self._target))
        except Exception as exc:
            if self._timer:
                self._timer.stop()
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class FlyGlyph(QWidget):
    """A neutral visual marker only; it does not animate or choose behaviour."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(70, 52)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(205, 205, 205), 1.2))
        painter.setBrush(QBrush(QColor(30, 30, 30)))
        painter.drawEllipse(26, 16, 18, 24)
        painter.setBrush(QBrush(QColor(170, 170, 170, 80)))
        painter.drawEllipse(5, 6, 30, 24)
        painter.drawEllipse(35, 6, 30, 24)
        painter.drawEllipse(29, 9, 12, 11)


class FlybitWindow(QWidget):
    target_changed = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.state = load_state()
        self.setWindowTitle("Flybit")
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(330, 185)

        self.status = QLabel("Loading MaleCNS…")
        self.status.setStyleSheet("font-size: 14px; font-weight: 600;")
        self.metrics = QLabel("First launch may download ~260 MB of connectome data.")
        self.metrics.setWordWrap(True)
        self.metrics.setStyleSheet("font-family: Consolas; font-size: 11px;")
        self.note = QLabel("No scripted behaviour · raw photoreceptor input")
        self.note.setStyleSheet("font-size: 10px; color: #aaaaaa;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.addWidget(FlyGlyph(), alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.status)
        layout.addWidget(self.metrics)
        layout.addWidget(self.note)

        self._brain_thread = QThread(self)
        self._worker = BrainWorker()
        self._worker.moveToThread(self._brain_thread)
        self._brain_thread.started.connect(self._worker.start)
        self.target_changed.connect(self._worker.set_target)
        self._worker.ready.connect(self._on_ready)
        self._worker.snapshot.connect(self._on_snapshot)
        self._worker.failed.connect(self._on_failed)
        self._brain_thread.start()

        self._input_timer = QTimer(self)
        self._input_timer.setInterval(20)
        self._input_timer.timeout.connect(self._sample_cursor)
        self._input_timer.start()

        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width() - 20, screen.bottom() - self.height() - 20)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(70, 70, 70, 210), 1))
        painter.setBrush(QBrush(QColor(16, 16, 16, 225)))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 12, 12)

    @Slot()
    def _sample_cursor(self) -> None:
        screen = QApplication.primaryScreen().geometry()
        x = QCursor.pos().x()
        center = ((x - screen.left()) / max(1, screen.width() - 1)) * 2.0 - 1.0
        self.target_changed.emit(center)

    @Slot(str, int)
    def _on_ready(self, device: str, neurons: int) -> None:
        self.status.setText(f"MaleCNS online · {device.upper()}")
        self.metrics.setText(f"neurons {neurons:,}\nwaiting for neural activity…")

    @Slot(object)
    def _on_snapshot(self, snap: NeuralSnapshot) -> None:
        self.metrics.setText(
            f"step {snap.step:,}   spikes {snap.total_spikes:,}\n"
            f"descending spikes {snap.descending_spikes:,}   active {snap.descending_active:,}"
        )

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self.status.setText("Neural core failed")
        self.metrics.setText(message)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._input_timer.stop()
        self._brain_thread.quit()
        self._brain_thread.wait(2000)
        super().closeEvent(event)
