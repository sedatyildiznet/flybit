"""UI-independent simulation contracts and timing telemetry."""
from __future__ import annotations

from dataclasses import dataclass

from .circadian import CircadianSnapshot
from .ethology import EthologySnapshot
from .motion import BiomechanicsSnapshot, FlyBodyState, MotorActivity, MotionEvent


@dataclass(frozen=True)
class TimedNeuralOutput:
    motor: MotorActivity
    timestamp: float
    step: int


@dataclass(frozen=True)
class SimulationSnapshot:
    timestamp: float
    step: int
    body: FlyBodyState
    ethology: EthologySnapshot
    biomechanics: BiomechanicsSnapshot
    circadian: CircadianSnapshot
    events: tuple[MotionEvent, ...]
    neural_timestamp: float
    neural_step: int
    neural_latency_ms: float
    stale_neural_output: bool

