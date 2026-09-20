"""Headless orchestration for the sensory-to-body Flybit loop.

The caller supplies measured/modeled sensory and the latest asynchronous
MaleCNS output.  This controller never consumes desktop semantics.
"""
from __future__ import annotations

import time
from pathlib import Path

from .arousal import ThreatArousalModel
from .care import CareModel
from .circadian import CircadianModel
from .controller import SimulationSnapshot, TimedNeuralOutput
from .ethology import EthologyModel
from .life import LifeModel
from .motion import FlyBodyState, FlyKinematics, MotorActivity, MotionEvent
from .memory import AssociativeMemory, sensory_fingerprint
from .olfaction import FoodOdorModel
from .phenotype import phenotype_from_identity
from .state import FlybitState
from .world import Surface, boundary_cue


class FlybitSimulation:
    """Own physiology, ethology, motor integration and monotonic ordering."""

    STALE_NEURAL_SECONDS = 0.160

    def __init__(self, state: FlybitState, body: FlyBodyState, *, seed: int = 64,
                 memory_path: Path | None = None) -> None:
        self.state = state
        self.step_count = 0
        phenotype = phenotype_from_identity(state.created_at)
        self.care = CareModel(state)
        self.life = LifeModel(state)
        self.circadian = CircadianModel(state)
        self.arousal = ThreatArousalModel()
        self.arousal._value = max(0.0, min(1.0, state.threat_memory))
        self.ethology = EthologyModel(seed=seed, grooming_need=state.grooming_need,
                                     threat_memory=state.threat_memory, phenotype=phenotype)
        self.kinematics = FlyKinematics(body, phenotype=phenotype)
        self.olfaction = FoodOdorModel()
        self._neural = TimedNeuralOutput(MotorActivity(), float("-inf"), 0)
        self.memory = AssociativeMemory(memory_path)
        self._memory_save_elapsed = 0.0

    def set_neural_output(self, output: TimedNeuralOutput) -> None:
        if output.step >= self._neural.step and output.timestamp >= self._neural.timestamp:
            self._neural = output

    def tick(self, dt: float, *, sensory=None, surfaces: list[Surface] | None = None,
             bounds: tuple[float, float, float, float] = (0, 0, 1920, 1080),
             timestamp: float | None = None) -> SimulationSnapshot:
        now = time.monotonic() if timestamp is None else float(timestamp)
        dt = max(0.001, min(0.100, float(dt)))
        self.step_count += 1
        surfaces = surfaces or []
        body = self.kinematics.state
        self.care.tick(dt)
        prior_bio = self.kinematics.biomechanics()
        self.life.tick(dt, motor_load=prior_bio.locomotor_load, hunger=self.state.hunger)
        life = self.life.snapshot()
        ambient = float(getattr(sensory, "ambient_luminance", 0.5))
        self.circadian.tick(dt, motor_load=prior_bio.locomotor_load, ambient_luminance=ambient)
        circadian = self.circadian.snapshot()
        threat = max(float(getattr(sensory, "retinal_loom_left", 0.0)),
                     float(getattr(sensory, "retinal_loom_right", 0.0)))
        self.arousal.tick(dt, escape_drive=self._neural.motor.escape, sensory_threat=threat)
        food = self.care.food
        odor = self.olfaction.sample(x=body.x, y=body.y, heading=body.heading,
            food=(food.x, food.y, food.amount) if food else None,
            hunger_drive=self.care.homeostatic_drive)
        edge = boundary_cue(surfaces, body.x, body.y, body.heading, bounds)
        fingerprint = sensory_fingerprint(sensory=sensory, odor=odor,
            geometry=prior_bio.proprioception.geometry)
        memory_now = time.time()
        memory_bias = self.memory.recall(fingerprint, now=memory_now)
        ethology = self.ethology.tick(dt, neural=self._neural.motor if life.alive else MotorActivity(),
            sensory=sensory, odor=odor, boundary=edge, heading=body.heading,
            hunger_drive=self.care.homeostatic_drive, rest_drive=circadian.rest_drive,
            activity=life.activity, boldness=life.boldness, curiosity=life.curiosity,
            threat_arousal=self.arousal.snapshot().threat_arousal, airborne=body.airborne,
            proprioception=prior_bio.proprioception, memory_bias=memory_bias)
        events = self.kinematics.update(ethology.motor if life.alive else MotorActivity(),
            surfaces, bounds, dt=dt, physiology_gain=life.vitality,
            landing_drive=ethology.landing_drive, groom_target=ethology.groom_target,
            micro_action=ethology.micro_action,
            optic_flow=float(getattr(sensory, "optic_flow", 0.0)))
        latency = max(0.0, now - self._neural.timestamp)
        food_contact = self.care.contact(body.x, body.y)
        if food_contact:
            self.life.feed()
            self.ethology.begin_feeding(1.5)
            events.append(MotionEvent("food_contact", "physical sugar contact"))
        self.memory.observe(fingerprint, food_reward=1.0 if food_contact else 0.0,
            threat=threat, neutral=1.0 if threat < .04 and not food_contact else 0.0,
            safe_rest=1.0 if ethology.mode == "sleep" and threat < .04 else 0.0,
            dt=dt, now=memory_now)
        self._memory_save_elapsed += dt
        if self._memory_save_elapsed >= 30.0:
            self.memory.save()
            self._memory_save_elapsed = 0.0
        return SimulationSnapshot(now, self.step_count, body, ethology,
            self.kinematics.biomechanics(), circadian, tuple(events),
            self._neural.timestamp, self._neural.step, latency * 1000.0,
            latency > self.STALE_NEURAL_SECONDS)
