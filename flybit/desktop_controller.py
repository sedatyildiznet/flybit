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
from .simulation import FlybitSimulation
from .controller import TimedNeuralOutput
from .state import state_dir


from .workers import BrainWorker
from .rendering import FlyOverlay, FoodOverlay, FoodPlacementOverlay
from .control_panel import ControlPanel

class FlybitWindow(QObject):
    """Application controller; only the organism is visible by default."""

    scene_changed = Signal(object, object, object)
    homeostasis_changed = Signal(
        float, float, float, float, float, float, float
    )
    persist_neural = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.app = QApplication.instance()
        self.state = load_state()
        offline_elapsed = elapsed_since_last_simulation(self.state)
        self.care = CareModel(self.state)
        self.life = LifeModel(self.state)
        self.circadian = CircadianModel(self.state)
        self.arousal = ThreatArousalModel()
        self.phenotype = phenotype_from_identity(self.state.created_at)
        self.ethology = EthologyModel(
            seed=seed_from_identity(self.state.created_at),
            grooming_need=float(getattr(self.state, "grooming_need", 0.18)),
            threat_memory=float(getattr(self.state, "threat_memory", 0.0)),
            phenotype=self.phenotype,
        )
        self.ethogram = EthogramRecorder()
        self.olfaction = FoodOdorModel()
        if offline_elapsed > 1.0:
            self.care.tick(offline_elapsed)
            self.life.elapse(offline_elapsed, hunger=self.state.hunger)
            self.circadian.elapse(offline_elapsed)
        self.semantic_scanner = DesktopSemanticScanner()
        self.surface_scanner = WindowSurfaceScanner()
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
        self.surfaces = self.surface_scanner.scan()

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
            ),
            phenotype=self.phenotype,
        )
        self.simulation = FlybitSimulation(
            self.state,
            self.kinematics.state,
            seed=seed_from_identity(self.state.created_at),
            memory_path=state_dir() / "memory.json",
        )
        # The Qt layer renders and reports these models; ordered updates are
        # owned by FlybitSimulation.
        self.care = self.simulation.care
        self.life = self.simulation.life
        self.circadian = self.simulation.circadian
        self.arousal = self.simulation.arousal
        self.ethology = self.simulation.ethology
        self.kinematics = self.simulation.kinematics
        self.olfaction = self.simulation.olfaction
        self.latest_neural_motor = MotorActivity()
        self.latest_motor = MotorActivity()
        self.latest_snapshot: NeuralSnapshot | None = None
        self.latest_sensory = None
        self.latest_odor = None
        self.latest_ethology: EthologySnapshot | None = None
        self._last_log: dict[str, float] = {}
        self._last_tick_time = time.monotonic()

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

        # Vision is captured from the same world tick as body physics. The
        # neural worker remains asynchronous, but all desktop observations now
        # carry the current body state instead of running on a competing timer.
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
        self._refresh_perception()
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
        now = time.monotonic()
        dt = max(0.005, min(0.100, now - self._last_tick_time))
        self._last_tick_time = now

        # One world clock: sample the scene immediately before physiology,
        # ethology and body integration.
        self._capture_scene()
        bounds = self._desktop_bounds()
        snap = self.simulation.tick(
            dt, sensory=self.latest_sensory, surfaces=self.surfaces,
            bounds=bounds, timestamp=now,
        )
        life = self.life.snapshot()
        circadian = snap.circadian
        arousal = self.arousal.snapshot()
        ethology = snap.ethology
        self.latest_ethology = ethology
        self.latest_motor = ethology.motor if life.alive else MotorActivity()
        self.state.grooming_need = ethology.grooming_need
        self.state.threat_memory = ethology.alertness

        self.homeostasis_changed.emit(
            self.care.homeostatic_drive,
            life.vitality,
            life.activity,
            life.boldness,
            life.curiosity,
            circadian.rest_drive,
            arousal.threat_arousal,
        )
        events = snap.events

        for event in events:
            if event.kind == "land":
                self.panel.append_log(
                    f"flight settled · {event.detail}"
                )
            elif event.kind == "takeoff_prepare":
                self.panel.append_log(
                    "escape preload · legs compressing"
                )
            elif event.kind == "takeoff":
                self.panel.append_log(
                    f"{ethology.mode} → jump + wing takeoff"
                )
            elif event.kind == "surface_contact":
                self.panel.append_log(
                    f"substrate contact · {event.detail}"
                )
            elif event.kind == "food_contact":
                self.food_overlay.hide()
                self.panel.append_log(
                    "sugar contact → feeding bout · nutrition state updated"
                )
                save_state(self.state)

        body = self.kinematics.state
        bio = self.kinematics.biomechanics()
        self.ethogram.update(
            dt,
            mode=ethology.mode,
            speed=bio.speed,
            turn_rate=bio.turn_rate,
        )
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
            behavior=ethology.mode,
            legs=bio.legs,
            body_bob=bio.body_bob,
            head_yaw=ethology.head_yaw,
            proboscis_extension=ethology.proboscis_extension,
            groom_target=ethology.groom_target,
            micro_action=ethology.micro_action,
            leg_extension=bio.leg_extension,
            body_scale=self.phenotype.body_scale,
        )
        self._position_overlay()

        if self.panel.isVisible():
            self.panel.update_life(
                life,
                bio,
                circadian,
                arousal,
                ethology,
            )

        if self.care.contact(body.x, body.y):
            self.life.feed()
            self.ethology.begin_feeding(1.5)
            self.food_overlay.hide()
            self.panel.append_log(
                "sugar contact → feeding bout · nutrition state updated"
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

    def _sample_odor(self):
        body = self.kinematics.state
        food = self.care.food
        odor_food = (
            (food.x, food.y, food.amount)
            if food is not None
            else None
        )
        self.latest_odor = self.olfaction.sample(
            x=body.x,
            y=body.y,
            heading=body.heading,
            food=odor_food,
            hunger_drive=self.care.homeostatic_drive,
        )
        return self.latest_odor

    @Slot()
    def _refresh_care(self) -> None:
        food = self.care.food
        odor = self._sample_odor()
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
            self.arousal.snapshot(),
            self.latest_ethology,
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
        self.surfaces = self.surface_scanner.scan()
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
        self.latest_neural_motor = snap.motor
        self.simulation.set_neural_output(
            TimedNeuralOutput(snap.motor, time.monotonic(), self.simulation.step_count)
        )

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
        self.state.grooming_need = float(self.ethology.grooming_need)
        self.state.threat_memory = float(self.ethology.alertness)
        mark_simulated_now(self.state)
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
        self._care_timer.stop()
        self._perception_timer.stop()
        self._save_timer.stop()
        self.food_overlay.hide()
        self.food_placement.hide()
        self.persist_neural.emit()
        if self._brain_thread.isRunning():
            self._brain_thread.quit()
            self._brain_thread.wait(2500)
