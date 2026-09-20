"""Deterministic headless behavioral assays for regression testing."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from .controller import TimedNeuralOutput
from .memory import AssociativeMemory, sensory_fingerprint
from .motion import FlyBodyState, MotorActivity
from .simulation import FlybitSimulation
from .state import FlybitState
from .world import Surface


MODES = ("idle", "walk", "turn", "boundary", "forage", "feed", "groom", "sleep", "escape", "flight", "landing")


@dataclass(frozen=True)
class AssayReport:
    name: str
    seed: int
    duration_seconds: float
    mode_ratios: dict[str, float]
    bout_seconds: dict[str, dict[str, float]]
    startle_latency_ms: float | None
    landing_success: float
    walking_speed: float
    turn_rate: float
    stride_frequency: float
    slip_frequency: float
    food_finding_success: float


def _sensory(**updates):
    values = dict(retinal_loom_left=0.0, retinal_loom_right=0.0,
                  mechanosensory_disturbance=0.0, optic_flow=0.0, ambient_luminance=.55)
    values.update(updates)
    return SimpleNamespace(**values)


def run_assay(name: str, *, seed: int = 64, duration: float = 30.0, dt: float = .02) -> AssayReport:
    state = FlybitState(created_at=f"2099-01-01T00:00:{seed % 60:02d}+00:00", hunger=.72,
                        sleep_pressure=.32)
    body = FlyBodyState(400, 300)
    sim = FlybitSimulation(state, body, seed=seed)
    surfaces = [Surface(1, 100, 700, 80, 520, z_order=0)]
    if name == "food": sim.care.place_food(470, 300)
    if name == "edge": body.x = 108
    if name == "sleep_startle": state.sleep_pressure = .98
    counts = Counter(); bouts: dict[str, list[float]] = defaultdict(list)
    active_mode = None; bout = 0.0
    speeds: list[float] = []; turns: list[float] = []; strides: list[float] = []
    slips = 0; land_attempts = 0; lands = 0; startle_at = None; latency = None
    memory = AssociativeMemory()
    total_steps = max(1, int(duration / dt))
    for step in range(total_steps):
        t = step * dt
        sensory = _sensory()
        motor = MotorActivity()
        if name == "looming" and 5 <= t < 5.5:
            sensory = _sensory(retinal_loom_left=.9, mechanosensory_disturbance=.3)
            if startle_at is None: startle_at = t
        elif name == "flight_landing" and t < .3:
            motor = MotorActivity(escape_left=1, escape_right=1)
        elif name == "flight_landing" and 1.0 < t < 2.5:
            sensory = _sensory(retinal_loom_left=.25, retinal_loom_right=.25)
        elif name == "sleep_startle" and 15 <= t < 15.4:
            sensory = _sensory(retinal_loom_right=.95)
            if startle_at is None: startle_at = t
        sim.set_neural_output(TimedNeuralOutput(motor, t, step + 1))
        snap = sim.tick(dt, sensory=sensory, surfaces=surfaces, bounds=(0, 0, 800, 600), timestamp=t)
        mode = snap.ethology.mode; counts[mode] += 1
        if mode != active_mode:
            if active_mode is not None: bouts[active_mode].append(bout)
            active_mode, bout = mode, 0.0
        bout += dt
        if startle_at is not None and latency is None and mode == "escape": latency = max(0.0, t - startle_at) * 1000
        speeds.append(snap.biomechanics.speed); turns.append(abs(snap.biomechanics.turn_rate)); strides.append(snap.biomechanics.stride_hz)
        slips += snap.biomechanics.proprioception.slip_mean > .25
        for event in snap.events:
            land_attempts += event.kind in {"land", "landing_abort"}
            lands += event.kind == "land"
        if name == "learning":
            fp = sensory_fingerprint(sensory=sensory, odor=sim.olfaction.sample(x=body.x, y=body.y,
                heading=body.heading, food=None, hunger_drive=.5), geometry=snap.biomechanics.proprioception.geometry)
            memory.observe(fp, neutral=1, dt=dt, now=t)
    if active_mode is not None: bouts[active_mode].append(bout)
    food_success = float(name == "food" and sim.care.food is None)
    return AssayReport(name, seed, total_steps * dt,
        {mode: counts[mode] / total_steps for mode in MODES},
        {mode: {"mean": sum(values) / len(values), "min": min(values), "max": max(values), "count": len(values)} for mode, values in bouts.items()},
        latency, lands / max(1, land_attempts), sum(speeds) / len(speeds), sum(turns) / len(turns),
        sum(strides) / len(strides), slips / (total_steps * dt), food_success)


def run_behavior_assays(*, seed: int = 64, duration: float = 30.0) -> dict[str, object]:
    names = ("baseline", "looming", "food", "edge", "flight_landing", "sleep_startle", "learning")
    return {"schema": 1, "seed": seed, "assays": {name: asdict(run_assay(name, seed=seed, duration=duration)) for name in names}}


def write_report(path: Path, *, seed: int = 64, duration: float = 30.0) -> dict[str, object]:
    report = run_behavior_assays(seed=seed, duration=duration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
