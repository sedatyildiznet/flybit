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


from .rendering import BrainMapWidget

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
        self.arousal_status = QLabel("Threat arousal · —")
        self.arousal_status.setObjectName("muted")
        life_layout.addWidget(self.arousal_status)
        self.behavior_status = QLabel("Ethology · —")
        self.behavior_status.setWordWrap(True)
        self.behavior_status.setObjectName("muted")
        life_layout.addWidget(self.behavior_status)
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
            f"visual projection {snap.visual_projection_spikes:,} spikes   "
            f"LPLC2 loom {snap.looming_spikes:,}"
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
            f"habituation {dynamics.loom_habituation:.2f} · "
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
        arousal=None,
        ethology: EthologySnapshot | None = None,
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
            f"stride {biomechanics.stride_hz:.1f}Hz · "
            f"stance {biomechanics.stance_count}/6 · "
            f"tripod {biomechanics.tripod_coherence:.2f} · "
            f"wingbeat {biomechanics.wingbeat_hz:.0f}Hz · "
            f"altitude {biomechanics.altitude:.1f} · "
            f"legs {biomechanics.leg_extension:.2f} · "
            f"substrate {biomechanics.support_title[:28]}"
        )
        if circadian is not None:
            self.circadian_status.setText(
                f"Circadian wake {circadian.wake_drive:.2f} · "
                f"sleep pressure {circadian.sleep_pressure:.2f} · "
                f"rest drive {circadian.rest_drive:.2f} · "
                f"ambient {circadian.ambient_luminance:.2f}"
            )
        if arousal is not None:
            self.arousal_status.setText(
                f"Threat arousal · {arousal.threat_arousal:.2f}"
            )
        if ethology is not None:
            detail = (
                f"sleep {ethology.sleep_stage}"
                if ethology.asleep
                else (
                    f"groom {ethology.groom_target}"
                    if ethology.grooming
                    else (
                        f"micro {ethology.micro_action}"
                        if ethology.micro_action
                        else "active"
                    )
                )
            )
            self.behavior_status.setText(
                f"Ethology · {ethology.mode.upper()} · {detail} · "
                f"alert {ethology.alertness:.2f} · "
                f"threat {ethology.threat_drive:.2f} · "
                f"food {ethology.food_drive:.2f} · "
                f"boundary {ethology.boundary_drive:.2f} · "
                f"landing {ethology.landing_drive:.2f}"
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



