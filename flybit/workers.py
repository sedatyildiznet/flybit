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
        self._threat_arousal = 0.0

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

    @Slot(float, float, float, float, float, float, float)
    def set_homeostasis(
        self,
        hunger_drive: float,
        vitality: float,
        activity_trait: float,
        boldness_trait: float,
        curiosity_trait: float,
        rest_drive: float,
        threat_arousal: float,
    ) -> None:
        self._hunger_drive = float(hunger_drive)
        self._vitality = float(vitality)
        self._activity_trait = float(activity_trait)
        self._boldness_trait = float(boldness_trait)
        self._curiosity_trait = float(curiosity_trait)
        self._rest_drive = float(rest_drive)
        self._threat_arousal = float(threat_arousal)

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
                self._threat_arousal,
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



