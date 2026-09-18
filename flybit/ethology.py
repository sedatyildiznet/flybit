"""Modeled ethology/VNC layer for lifelike desktop behaviour.

The controller consumes neural motor activity, non-semantic sensory quantities,
internal physiology and geometric boundary cues. It never inspects application
names or maps cursor identity directly to actions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .motion import MotorActivity
from .phenotype import IndividualPhenotype


_NEUTRAL_PHENOTYPE = IndividualPhenotype(
    stride_scale=1.0,
    turn_bias=0.0,
    pause_scale=1.0,
    grooming_bias=1.0,
    startle_bias=0.0,
    flight_saccade_scale=1.0,
    micro_activity=1.0,
    body_scale=1.0,
    handedness=0.0,
)


@dataclass(frozen=True)
class EthologySnapshot:
    mode: str
    motor: MotorActivity
    asleep: bool
    grooming: bool
    sleep_stage: str
    groom_target: str
    micro_action: str
    proboscis_extension: float
    head_yaw: float
    landing_drive: float
    flight_saccade: float
    boundary_drive: float
    alertness: float
    threat_drive: float
    food_drive: float
    grooming_need: float


class EthologyModel:
    """Competing semi-Markov motor programs with persistent individuality."""

    MODES = {
        "idle",
        "walk",
        "turn",
        "boundary",
        "forage",
        "feed",
        "groom",
        "sleep",
        "escape",
        "flight",
        "landing",
    }

    _GROOM_PRIORITY = {
        "eyes": 1.32,
        "antennae": 1.23,
        "proboscis": 1.12,
        "abdomen": 0.98,
        "wings": 0.88,
        "thorax": 0.80,
    }

    def __init__(
        self,
        *,
        seed: int = 64,
        grooming_need: float = 0.18,
        threat_memory: float = 0.0,
        phenotype: IndividualPhenotype | None = None,
    ) -> None:
        self.rng = random.Random(int(seed))
        self.phenotype = phenotype or _NEUTRAL_PHENOTYPE
        self.mode = "idle"
        self.mode_elapsed = 0.0
        self.bout_remaining = 0.35
        self.alertness = self._clamp01(threat_memory)
        self.grooming_need = self._clamp01(grooming_need)
        self._wander_bias = 0.0
        self._search_phase = self.rng.random() * math.tau
        self._groom_target = ""
        self._micro_action = ""
        self._micro_remaining = 0.0
        self._flight_elapsed = 0.0
        self._saccade_remaining = 0.0
        self._saccade_sign = 0.0
        self._saccade_cooldown = self.rng.uniform(0.35, 1.2)

        base = self.grooming_need
        self._groom_load = {
            "eyes": self._clamp01(base * 0.92),
            "antennae": self._clamp01(base * 0.84),
            "proboscis": self._clamp01(base * 0.48),
            "abdomen": self._clamp01(base * 0.62),
            "wings": self._clamp01(base * 0.55),
            "thorax": self._clamp01(base * 0.50),
        }

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _wrap(value: float) -> float:
        return (float(value) + math.pi) % math.tau - math.pi

    def _set_mode(self, mode: str, seconds: float) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown ethology mode: {mode}")
        changed = mode != self.mode
        self.mode = mode
        if changed:
            self.mode_elapsed = 0.0
            self._micro_action = ""
            self._micro_remaining = 0.0
        self.bout_remaining = max(0.04, float(seconds))
        if changed and mode in {"walk", "turn"}:
            self._wander_bias = max(
                -1.0,
                min(
                    1.0,
                    self.rng.uniform(-1.0, 1.0)
                    + 0.34 * self.phenotype.turn_bias,
                ),
            )
        if changed and mode == "groom":
            self._groom_target = self._choose_groom_target()

    def begin_feeding(self, seconds: float = 1.4) -> None:
        self._set_mode("feed", max(0.35, float(seconds)))

    @staticmethod
    def _max_motor(a: MotorActivity, b: MotorActivity) -> MotorActivity:
        return MotorActivity(
            forward_left=max(a.forward_left, b.forward_left),
            forward_right=max(a.forward_right, b.forward_right),
            steer_left=max(a.steer_left, b.steer_left),
            steer_right=max(a.steer_right, b.steer_right),
            escape_left=max(a.escape_left, b.escape_left),
            escape_right=max(a.escape_right, b.escape_right),
            backward_left=max(a.backward_left, b.backward_left),
            backward_right=max(a.backward_right, b.backward_right),
            flight_left=max(a.flight_left, b.flight_left),
            flight_right=max(a.flight_right, b.flight_right),
        )

    def _choose_groom_target(self) -> str:
        best = "eyes"
        best_score = -1.0
        for target, load in self._groom_load.items():
            priority = self._GROOM_PRIORITY[target]
            score = (
                load * priority
                + self.rng.uniform(-0.045, 0.045)
            )
            if score > best_score:
                best_score = score
                best = target
        return best

    def _update_groom_load(self, dt: float, movement_load: float) -> None:
        deposition = dt * (0.0025 + 0.011 * movement_load)
        factors = {
            "eyes": 1.08,
            "antennae": 1.12,
            "proboscis": 0.62,
            "abdomen": 0.82,
            "wings": 0.90,
            "thorax": 0.72,
        }
        for target in self._groom_load:
            self._groom_load[target] = self._clamp01(
                self._groom_load[target] + deposition * factors[target]
            )
        self.grooming_need = max(self._groom_load.values())

    def _choose_ground_mode(
        self,
        *,
        hunger: float,
        food_drive: float,
        rest: float,
        activity: float,
        curiosity: float,
        boundary_drive: float,
    ) -> None:
        """Choose among competing drives using noisy activation, not hard order."""
        groom = self.grooming_need * self.phenotype.grooming_bias
        scores = {
            "sleep": 0.10 + 1.65 * rest - 0.45 * food_drive,
            "forage": 0.06 + 1.25 * food_drive + 0.45 * hunger,
            "groom": 0.18 + 1.25 * groom,
            "boundary": 0.12 + 1.10 * boundary_drive,
            "walk": 0.46 + 0.50 * activity + 0.20 * curiosity,
            "turn": 0.28 + 0.20 * curiosity + 0.10 * abs(self.phenotype.turn_bias),
            "idle": 0.48 + 0.28 * (1.0 - activity),
        }

        if rest < 0.42:
            scores["sleep"] -= 0.55
        if food_drive < 0.12:
            scores["forage"] -= 0.52
        if groom < 0.48:
            scores["groom"] -= 0.52
        if boundary_drive < 0.28:
            scores["boundary"] -= 0.60

        mode = max(
            scores,
            key=lambda key: scores[key] + self.rng.uniform(-0.16, 0.16),
        )

        pause_scale = self.phenotype.pause_scale
        if mode == "sleep":
            self._set_mode("sleep", self.rng.uniform(5.0, 16.0) * pause_scale)
        elif mode == "forage":
            self._set_mode("forage", self.rng.uniform(0.7, 2.5))
        elif mode == "groom":
            self._set_mode("groom", self.rng.uniform(0.55, 1.65))
        elif mode == "boundary":
            self._set_mode("boundary", self.rng.uniform(0.45, 2.2))
        elif mode == "walk":
            self._set_mode("walk", self.rng.uniform(0.28, 1.55))
        elif mode == "turn":
            self._set_mode("turn", self.rng.uniform(0.10, 0.42))
        else:
            self._set_mode(
                "idle",
                self.rng.uniform(0.16, 0.95) * pause_scale,
            )

    def _sleep_stage(self) -> str:
        if self.mode != "sleep":
            return "awake"
        if self.mode_elapsed < 1.8:
            return "drowsy"
        if self.mode_elapsed < 7.0:
            return "light"
        return "deep"

    def _update_micro_action(self, dt: float) -> tuple[str, float, float]:
        if self.mode != "idle":
            self._micro_action = ""
            self._micro_remaining = 0.0
            return "", 0.0, 0.0

        self._micro_remaining -= dt
        if self._micro_remaining <= 0.0:
            choices = [
                "antenna_sweep",
                "head_turn",
                "leg_adjust",
                "wing_flick",
                "still",
                "still",
                "proboscis",
            ]
            self._micro_action = self.rng.choice(choices)
            self._micro_remaining = self.rng.uniform(0.18, 0.72) / max(
                0.65,
                self.phenotype.micro_activity,
            )

        phase = math.sin(self.mode_elapsed * 8.5 + self.phenotype.handedness)
        head_yaw = 0.0
        proboscis = 0.0
        if self._micro_action in {"antenna_sweep", "head_turn"}:
            head_yaw = 0.24 * phase
        if self._micro_action == "proboscis":
            proboscis = 0.35 + 0.20 * (0.5 + 0.5 * phase)
        return self._micro_action, head_yaw, proboscis

    def _trigger_saccade(self, sign: float, strength: float = 1.0) -> None:
        self._saccade_sign = -1.0 if sign < 0.0 else 1.0
        self._saccade_remaining = self.rng.uniform(0.045, 0.095)
        self._saccade_cooldown = self.rng.uniform(0.55, 1.55) / max(
            0.65,
            self.phenotype.flight_saccade_scale,
        )
        if strength > 0.75:
            self._saccade_remaining *= 1.15

    def tick(
        self,
        dt: float,
        *,
        neural: MotorActivity,
        sensory=None,
        odor=None,
        boundary=None,
        heading: float = 0.0,
        hunger_drive: float = 0.0,
        rest_drive: float = 0.0,
        activity: float = 0.5,
        boldness: float = 0.5,
        curiosity: float = 0.5,
        threat_arousal: float = 0.0,
        airborne: bool = False,
    ) -> EthologySnapshot:
        dt = max(0.001, min(0.20, float(dt)))
        hunger = self._clamp01(hunger_drive)
        rest = self._clamp01(rest_drive)
        activity = self._clamp01(activity)
        boldness = self._clamp01(boldness)
        curiosity = self._clamp01(curiosity)
        arousal = self._clamp01(threat_arousal)

        loom_left = self._clamp01(
            getattr(sensory, "retinal_loom_left", 0.0)
        )
        loom_right = self._clamp01(
            getattr(sensory, "retinal_loom_right", 0.0)
        )
        disturbance = self._clamp01(
            getattr(sensory, "mechanosensory_disturbance", 0.0)
        )
        optic_flow = float(getattr(sensory, "optic_flow", 0.0))
        loom = max(loom_left, loom_right)
        threat = self._clamp01(
            max(neural.escape, 0.88 * loom + 0.24 * disturbance)
        )

        odor_salience = self._clamp01(getattr(odor, "salience", 0.0))
        odor_mean = self._clamp01(getattr(odor, "mean", 0.0))
        odor_gradient = max(
            -1.0,
            min(1.0, float(getattr(odor, "gradient", 0.0))),
        )
        food_drive = self._clamp01(
            odor_salience * (0.30 + 0.70 * hunger)
        )

        boundary_drive = self._clamp01(getattr(boundary, "strength", 0.0))

        self.alertness *= math.exp(-dt / 12.0)
        if threat > 0.0:
            self.alertness = max(
                self.alertness,
                self._clamp01(0.12 + 0.82 * threat),
            )
        self.alertness = max(self.alertness, 0.55 * arousal)

        movement_load = max(
            neural.forward,
            neural.backward,
            neural.flight,
            neural.escape,
        )
        self._update_groom_load(dt, movement_load)

        self.mode_elapsed += dt
        self.bout_remaining -= dt

        landing_drive = 0.0
        flight_saccade = 0.0

        if airborne:
            self._flight_elapsed += dt
            self._saccade_cooldown -= dt

            if self.mode not in {"flight", "landing", "escape"}:
                self._set_mode("flight", self.rng.uniform(0.45, 1.8))

            # In flight, coherent expansion can prepare landing instead of
            # blindly reusing the grounded escape program.
            if (
                self._flight_elapsed > 0.24
                and 0.11 <= loom <= 0.72
                and self.mode != "escape"
            ):
                self._set_mode("landing", self.rng.uniform(0.28, 0.75))

            if self.mode == "landing":
                landing_drive = self._clamp01(
                    0.32 + 1.20 * loom + 0.15 * min(1.0, abs(optic_flow))
                )
                if loom < 0.025 and self.bout_remaining <= 0.0:
                    self._set_mode("flight", self.rng.uniform(0.4, 1.2))

            # Strong/asymmetric expansion or spontaneous free-flight timing can
            # evoke a short yaw saccade.
            if self._saccade_remaining <= 0.0:
                if loom > 0.48 and self._saccade_cooldown <= 0.0:
                    away = loom_left - loom_right
                    if abs(away) < 0.05:
                        away = self.phenotype.handedness or 1.0
                    self._trigger_saccade(away, loom)
                elif (
                    self._saccade_cooldown <= 0.0
                    and self.rng.random()
                    < 0.018 * self.phenotype.flight_saccade_scale
                ):
                    self._trigger_saccade(
                        self.rng.choice((-1.0, 1.0)),
                        0.35,
                    )

            if self._saccade_remaining > 0.0:
                self._saccade_remaining -= dt
                flight_saccade = self._saccade_sign

        else:
            self._flight_elapsed = 0.0
            self._saccade_remaining = 0.0

            escape_threshold = max(
                0.14,
                min(
                    0.64,
                    0.42
                    + 0.18 * boldness
                    - 0.22 * self.alertness
                    + self.phenotype.startle_bias,
                ),
            )
            neural_escape = neural.escape > 0.0

            if neural_escape or threat >= escape_threshold:
                self._set_mode("escape", self.rng.uniform(0.15, 0.38))
            elif self.mode == "sleep":
                stage = self._sleep_stage()
                stage_threshold = {
                    "drowsy": 0.16,
                    "light": 0.24,
                    "deep": 0.36,
                }[stage]
                wake_threshold = stage_threshold + 0.08 * rest
                if threat >= wake_threshold or neural.forward > 0.16:
                    self._set_mode(
                        "escape" if threat >= wake_threshold else "idle",
                        0.22,
                    )
                elif self.bout_remaining <= 0.0:
                    if rest > 0.50:
                        self._set_mode(
                            "sleep",
                            self.rng.uniform(4.5, 15.0)
                            * self.phenotype.pause_scale,
                        )
                    else:
                        self._choose_ground_mode(
                            hunger=hunger,
                            food_drive=food_drive,
                            rest=rest,
                            activity=activity,
                            curiosity=curiosity,
                            boundary_drive=boundary_drive,
                        )
            elif self.mode == "boundary" and boundary_drive < 0.08:
                self.bout_remaining = 0.0
            elif self.bout_remaining <= 0.0:
                self._choose_ground_mode(
                    hunger=hunger,
                    food_drive=food_drive,
                    rest=rest,
                    activity=activity,
                    curiosity=curiosity,
                    boundary_drive=boundary_drive,
                )

        intent = MotorActivity()

        if self.mode == "escape":
            away = max(-1.0, min(1.0, loom_left - loom_right))
            if abs(away) < 0.08 and neural.escape > 0.0:
                away = max(
                    -1.0,
                    min(1.0, neural.escape_left - neural.escape_right),
                )
            base = 0.68 + 0.30 * max(threat, neural.escape)
            intent = MotorActivity(
                forward_left=base,
                forward_right=base,
                steer_left=max(0.0, -away) * 0.95,
                steer_right=max(0.0, away) * 0.95,
                escape_left=max(neural.escape_left, loom_right),
                escape_right=max(neural.escape_right, loom_left),
                flight_left=max(neural.flight_left, 0.32),
                flight_right=max(neural.flight_right, 0.32),
            )

        elif self.mode in {"flight", "landing"}:
            saccade = flight_saccade * 0.82
            thrust = 0.16 if self.mode == "flight" else 0.045
            forward = 0.18 if self.mode == "flight" else 0.08
            intent = MotorActivity(
                forward_left=forward,
                forward_right=forward,
                steer_left=max(0.0, -saccade),
                steer_right=max(0.0, saccade),
                flight_left=thrust,
                flight_right=thrust,
            )

        elif self.mode == "forage":
            self._search_phase = (self._search_phase + dt * 3.0) % math.tau
            cast = math.sin(self._search_phase) * (
                1.0 - min(1.0, odor_mean * 3.0)
            )
            steer = max(
                -1.0,
                min(
                    1.0,
                    odor_gradient * 2.6
                    + 0.34 * cast
                    + 0.10 * self.phenotype.turn_bias,
                ),
            )
            speed = 0.18 + 0.34 * hunger + 0.20 * odor_mean
            intent = MotorActivity(
                forward_left=speed,
                forward_right=speed,
                steer_left=max(0.0, -steer) * 0.58,
                steer_right=max(0.0, steer) * 0.58,
            )

        elif self.mode == "boundary" and boundary is not None:
            tangent = float(getattr(boundary, "tangent_heading", heading))
            inward = float(getattr(boundary, "inward_heading", heading))
            # Stay roughly parallel while adding a small inward correction at
            # very high edge proximity.
            target = tangent
            if boundary_drive > 0.82:
                blend = (boundary_drive - 0.82) / 0.18
                tx = (
                    (1.0 - blend) * math.cos(tangent)
                    + blend * math.cos(inward)
                )
                ty = (
                    (1.0 - blend) * math.sin(tangent)
                    + blend * math.sin(inward)
                )
                target = math.atan2(ty, tx)
            error = self._wrap(target - float(heading))
            steer = max(-1.0, min(1.0, error / 0.85))
            intent = MotorActivity(
                forward_left=0.22,
                forward_right=0.22,
                steer_left=max(0.0, -steer) * 0.62,
                steer_right=max(0.0, steer) * 0.62,
            )

        elif self.mode == "walk":
            speed = 0.12 + 0.25 * activity + 0.08 * curiosity
            steer = (
                self._wander_bias * (0.10 + 0.18 * curiosity)
                + 0.06 * self.phenotype.turn_bias
            )
            intent = MotorActivity(
                forward_left=speed,
                forward_right=speed,
                steer_left=max(0.0, -steer),
                steer_right=max(0.0, steer),
            )

        elif self.mode == "turn":
            steer = self._wander_bias or (
                self.phenotype.handedness or 1.0
            )
            intent = MotorActivity(
                forward_left=0.07,
                forward_right=0.07,
                steer_left=max(0.0, -steer) * 0.50,
                steer_right=max(0.0, steer) * 0.50,
            )

        elif self.mode == "groom":
            target = self._groom_target or self._choose_groom_target()
            self._groom_target = target
            self._groom_load[target] = self._clamp01(
                self._groom_load[target] - dt * 0.48
            )
            self.grooming_need = max(self._groom_load.values())

        micro_action, head_yaw, proboscis = self._update_micro_action(dt)
        if self.mode == "feed":
            proboscis = 1.0
            head_yaw = 0.04 * math.sin(self.mode_elapsed * 5.0)
        elif self.mode == "groom":
            head_yaw = 0.10 * math.sin(self.mode_elapsed * 10.0)

        motor = self._max_motor(neural, intent)
        if (
            self.mode in {"sleep", "groom", "feed"}
            and threat < 0.12
            and neural.escape <= 0.0
        ):
            motor = MotorActivity(
                forward_left=(
                    neural.forward_left if neural.forward_left > 0.28 else 0.0
                ),
                forward_right=(
                    neural.forward_right if neural.forward_right > 0.28 else 0.0
                ),
                steer_left=(
                    neural.steer_left if neural.steer_left > 0.32 else 0.0
                ),
                steer_right=(
                    neural.steer_right if neural.steer_right > 0.32 else 0.0
                ),
                escape_left=neural.escape_left,
                escape_right=neural.escape_right,
                backward_left=(
                    neural.backward_left if neural.backward_left > 0.28 else 0.0
                ),
                backward_right=(
                    neural.backward_right if neural.backward_right > 0.28 else 0.0
                ),
                flight_left=(
                    neural.flight_left if neural.flight_left > 0.30 else 0.0
                ),
                flight_right=(
                    neural.flight_right if neural.flight_right > 0.30 else 0.0
                ),
            )

        return EthologySnapshot(
            mode=self.mode,
            motor=motor,
            asleep=self.mode == "sleep",
            grooming=self.mode == "groom",
            sleep_stage=self._sleep_stage(),
            groom_target=self._groom_target if self.mode == "groom" else "",
            micro_action=micro_action,
            proboscis_extension=float(proboscis),
            head_yaw=float(head_yaw),
            landing_drive=float(landing_drive),
            flight_saccade=float(flight_saccade),
            boundary_drive=float(boundary_drive),
            alertness=float(self.alertness),
            threat_drive=float(threat),
            food_drive=float(food_drive),
            grooming_need=float(self.grooming_need),
        )
