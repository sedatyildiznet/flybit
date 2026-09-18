"""Modeled ethology layer for lifelike desktop behaviour.

This module sits between neural descending output and body mechanics. It does
not inspect semantic desktop labels or cursor identity. Instead it combines
neural motor activity with sensory quantities (looming, near-field disturbance,
food odor), internal drives and stochastic bout timing to model behavioural
states that are missing from the simplified whole-CNS neuron model.

The layer is deliberately labelled MODELED: it is an ethological/VNC bridge,
not a claim that MaleCNS alone currently reproduces every behaviour below.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .motion import MotorActivity


@dataclass(frozen=True)
class EthologySnapshot:
    mode: str
    motor: MotorActivity
    asleep: bool
    grooming: bool
    alertness: float
    threat_drive: float
    food_drive: float
    grooming_need: float


class EthologyModel:
    """Competing semi-Markov motor programmes for a desktop fly.

    The controller never receives object names. Visual threat comes from raw
    retinal expansion; food approach comes from the synthetic bilateral odor
    field. Neural descending activity always survives into the final motor
    command and can override quieter autonomous bouts.
    """

    MODES = {
        "idle",
        "walk",
        "turn",
        "forage",
        "feed",
        "groom",
        "sleep",
        "escape",
        "flight",
    }

    def __init__(
        self,
        *,
        seed: int = 64,
        grooming_need: float = 0.18,
        threat_memory: float = 0.0,
    ) -> None:
        self.rng = random.Random(int(seed))
        self.mode = "idle"
        self.mode_elapsed = 0.0
        self.bout_remaining = 0.35
        self.alertness = max(0.0, min(1.0, float(threat_memory)))
        self.grooming_need = max(0.0, min(1.0, float(grooming_need)))
        self._wander_bias = 0.0
        self._search_phase = self.rng.random() * math.tau

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _set_mode(self, mode: str, seconds: float) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown ethology mode: {mode}")
        self.mode = mode
        self.mode_elapsed = 0.0
        self.bout_remaining = max(0.04, float(seconds))
        if mode in {"walk", "turn"}:
            self._wander_bias = self.rng.uniform(-1.0, 1.0)

    def begin_feeding(self, seconds: float = 1.4) -> None:
        """Enter a stationary feeding bout after physical food contact."""
        self._set_mode("feed", max(0.35, float(seconds)))

    @staticmethod
    def _max_motor(a: MotorActivity, b: MotorActivity) -> MotorActivity:
        """Preserve neural output while allowing modeled programmes to add drive."""
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

    def _choose_quiet_mode(
        self,
        *,
        hunger_drive: float,
        food_drive: float,
        rest_drive: float,
        activity: float,
        curiosity: float,
    ) -> None:
        if rest_drive > 0.64 and food_drive < 0.18:
            self._set_mode("sleep", self.rng.uniform(5.0, 18.0))
            return

        if food_drive > 0.16 and hunger_drive > 0.18:
            self._set_mode("forage", self.rng.uniform(0.8, 2.4))
            return

        groom_threshold = 0.64 - 0.10 * (1.0 - activity)
        if self.grooming_need >= groom_threshold:
            self._set_mode("groom", self.rng.uniform(0.7, 2.0))
            return

        walk_probability = 0.34 + 0.34 * activity + 0.18 * curiosity
        roll = self.rng.random()
        if roll < walk_probability:
            self._set_mode("walk", self.rng.uniform(0.35, 1.65))
        elif roll < walk_probability + 0.23:
            self._set_mode("turn", self.rng.uniform(0.12, 0.48))
        else:
            self._set_mode("idle", self.rng.uniform(0.18, 1.10))

    def tick(
        self,
        dt: float,
        *,
        neural: MotorActivity,
        sensory=None,
        odor=None,
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
        self.grooming_need = self._clamp01(
            self.grooming_need
            + dt * (0.004 + 0.012 * movement_load)
        )

        self.mode_elapsed += dt
        self.bout_remaining -= dt

        escape_threshold = max(
            0.16,
            min(0.62, 0.42 + 0.18 * boldness - 0.22 * self.alertness),
        )
        neural_escape = neural.escape > 0.0
        if neural_escape or threat >= escape_threshold:
            self._set_mode("escape", self.rng.uniform(0.16, 0.42))
        elif airborne:
            if self.mode != "escape":
                self._set_mode("flight", max(0.25, self.bout_remaining))
        elif self.mode == "sleep":
            wake_threshold = 0.18 + 0.24 * rest
            if threat >= wake_threshold or neural.forward > 0.12:
                self._set_mode(
                    "escape" if threat >= wake_threshold else "idle",
                    0.24,
                )
            elif self.bout_remaining <= 0.0:
                if rest > 0.52:
                    self._set_mode("sleep", self.rng.uniform(4.0, 15.0))
                else:
                    self._choose_quiet_mode(
                        hunger_drive=hunger,
                        food_drive=food_drive,
                        rest_drive=rest,
                        activity=activity,
                        curiosity=curiosity,
                    )
        elif self.bout_remaining <= 0.0:
            self._choose_quiet_mode(
                hunger_drive=hunger,
                food_drive=food_drive,
                rest_drive=rest,
                activity=activity,
                curiosity=curiosity,
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

        elif self.mode == "flight":
            intent = MotorActivity(
                forward_left=0.18,
                forward_right=0.18,
                flight_left=0.16,
                flight_right=0.16,
            )

        elif self.mode == "forage":
            self._search_phase = (self._search_phase + dt * 3.0) % math.tau
            cast = math.sin(self._search_phase) * (
                1.0 - min(1.0, odor_mean * 3.0)
            )
            steer = max(
                -1.0,
                min(1.0, odor_gradient * 2.6 + 0.34 * cast),
            )
            speed = 0.18 + 0.34 * hunger + 0.20 * odor_mean
            intent = MotorActivity(
                forward_left=speed,
                forward_right=speed,
                steer_left=max(0.0, -steer) * 0.58,
                steer_right=max(0.0, steer) * 0.58,
            )

        elif self.mode == "walk":
            speed = 0.12 + 0.25 * activity + 0.08 * curiosity
            steer = self._wander_bias * (0.10 + 0.18 * curiosity)
            intent = MotorActivity(
                forward_left=speed,
                forward_right=speed,
                steer_left=max(0.0, -steer),
                steer_right=max(0.0, steer),
            )

        elif self.mode == "turn":
            steer = self._wander_bias or 1.0
            intent = MotorActivity(
                forward_left=0.08,
                forward_right=0.08,
                steer_left=max(0.0, -steer) * 0.46,
                steer_right=max(0.0, steer) * 0.46,
            )

        elif self.mode == "groom":
            self.grooming_need = self._clamp01(
                self.grooming_need - dt * 0.28
            )

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
            alertness=float(self.alertness),
            threat_drive=float(threat),
            food_drive=float(food_drive),
            grooming_need=float(self.grooming_need),
        )
