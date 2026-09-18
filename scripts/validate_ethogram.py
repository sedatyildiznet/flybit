"""Deterministic behavioural calibration checks for Flybit v0.4."""
from __future__ import annotations

from types import SimpleNamespace

from flybit.benchmark import EthogramRecorder, wtsb_category
from flybit.ethology import EthologyModel
from flybit.motion import MotorActivity
from flybit.phenotype import phenotype_from_identity


def sensory(left=0.0, right=0.0, disturbance=0.0, optic_flow=0.0):
    return SimpleNamespace(
        retinal_loom_left=left,
        retinal_loom_right=right,
        mechanosensory_disturbance=disturbance,
        optic_flow=optic_flow,
    )


def odor(salience=0.0, mean=0.0, gradient=0.0):
    return SimpleNamespace(
        salience=salience,
        mean=mean,
        gradient=gradient,
    )


def boundary(strength=0.0, tangent_heading=0.0, inward_heading=1.57):
    return SimpleNamespace(
        strength=strength,
        tangent_heading=tangent_heading,
        inward_heading=inward_heading,
    )


def main() -> None:
    model = EthologyModel(
        seed=440,
        phenotype=phenotype_from_identity("validation-fly"),
    )
    rec = EthogramRecorder()
    seen = set()

    dt = 0.020
    for step in range(4000):
        t = step * dt
        food = (
            odor(0.70, 0.35, 0.25)
            if 22.0 < t < 34.0
            else odor()
        )
        edge = (
            boundary(0.78, tangent_heading=0.0)
            if 42.0 < t < 55.0
            else boundary()
        )
        snap = model.tick(
            dt,
            neural=MotorActivity(),
            sensory=sensory(),
            odor=food,
            boundary=edge,
            heading=0.15,
            hunger_drive=0.75 if t > 18.0 else 0.25,
            rest_drive=0.72 if 60.0 < t < 76.0 else 0.20,
            activity=0.62,
            boldness=0.48,
            curiosity=0.67,
            airborne=False,
        )
        seen.add(snap.mode)
        pseudo_speed = {
            "walk": 35.0,
            "forage": 42.0,
            "boundary": 28.0,
            "turn": 10.0,
        }.get(snap.mode, 0.0)
        pseudo_turn = 2.2 if snap.mode == "turn" else 0.25
        rec.update(
            dt,
            mode=snap.mode,
            speed=pseudo_speed,
            turn_rate=pseudo_turn,
        )

    required = {"walk", "turn", "forage", "boundary", "sleep"}
    missing = required - seen
    if missing:
        raise SystemExit(f"missing neutral ethogram modes: {sorted(missing)}")

    # Ground looming must produce an immediate escape motor program.
    escape = model.tick(
        dt,
        neural=MotorActivity(),
        sensory=sensory(left=0.95),
        odor=odor(),
        boundary=boundary(),
        airborne=False,
        boldness=0.2,
    )
    if escape.mode != "escape" or escape.motor.escape <= 0.0:
        raise SystemExit("ground startle/escape calibration failed")

    # Sustained coherent looming in flight must become landing preparation.
    flight = EthologyModel(
        seed=441,
        phenotype=phenotype_from_identity("flight-validation"),
    )
    landing_seen = False
    for _ in range(30):
        snap = flight.tick(
            dt,
            neural=MotorActivity(),
            sensory=sensory(left=0.28, right=0.24, optic_flow=0.2),
            odor=odor(),
            boundary=boundary(),
            airborne=True,
        )
        landing_seen |= snap.mode == "landing" and snap.landing_drive > 0.0
    if not landing_seen:
        raise SystemExit("airborne looming -> landing calibration failed")

    summary = rec.snapshot()
    categories = {wtsb_category(mode) for mode in seen}
    expected_categories = {"stop", "curved_walk", "sharp_turn", "boundary"}
    if not expected_categories.issubset(categories):
        raise SystemExit(
            f"incomplete WTSB coverage: {sorted(categories)}"
        )

    if summary.transitions < 12:
        raise SystemExit("ethogram transition diversity too low")
    if summary.speed_cv <= 0.10:
        raise SystemExit("locomotor variability too low")

    print(
        "Flybit ethogram calibration OK · "
        f"{summary.transitions} transitions · "
        f"modes {sorted(seen)}"
    )


if __name__ == "__main__":
    main()
