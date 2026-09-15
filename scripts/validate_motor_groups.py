"""Validate Flybit motor read-outs against the bundled MaleCNS data."""
from __future__ import annotations

import numpy as np

from flybit.neural import FlybitNeuralCore


REQUIRED = (
    "forward_L",
    "forward_R",
    "steer_L",
    "steer_R",
    "escape_L",
    "escape_R",
    "backward_L",
    "backward_R",
    "flight_L",
    "flight_R",
)


def main() -> int:
    core = FlybitNeuralCore(
        device="cpu",
        seed=64,
    )

    print("MaleCNS motor groups")
    for name, group in core.motor_groups.items():
        print(f"  {name:12s} {len(group):3d}")

    for side, group in core.loom_groups.items():
        print(f"  LPLC2_{side:1s}     {len(group):3d}")

    missing = [
        name
        for name in REQUIRED
        if len(core.motor_groups[name]) == 0
    ]
    missing += [
        f"LPLC2_{side}"
        for side, group in core.loom_groups.items()
        if len(group) == 0
    ]
    if missing:
        raise SystemExit(
            "FAIL: required motor groups are empty: "
            + ", ".join(missing)
        )

    # Giant Fiber physiology: one DNp01 spike must survive the decoder as a
    # takeoff command rather than being averaged below an arbitrary Hz gate.
    left_gf = core.motor_groups["escape_L"][0]
    motor = core._motor_activity(  # validation of private decoder on purpose
        np.asarray([left_gf], dtype=np.int64)
    )
    print("single DNp01 spike -> escape_left", motor.escape_left)
    if motor.escape_left < 0.99:
        raise SystemExit(
            "FAIL: single DNp01 spike was lost by motor decoder"
        )

    # Repeated bilateral DNa02 activity should produce some locomotor drive as
    # well as steering-capable activity; it must not be a rotate-in-place-only
    # channel.
    left_dna = core.motor_groups["steer_L"][0]
    right_dna = core.motor_groups["steer_R"][0]
    motor = None
    for _ in range(12):
        motor = core._motor_activity(
            np.asarray(
                [left_dna, right_dna],
                dtype=np.int64,
            )
        )
    assert motor is not None
    print("bilateral DNa activity -> forward", motor.forward)
    if motor.forward <= 0.02:
        raise SystemExit(
            "FAIL: bilateral DNa activity produces no walking drive"
        )

    # DNg02 is the identified flight-thrust population. Repeated activity must
    # survive the rate decoder.
    left_flight = core.motor_groups["flight_L"][0]
    right_flight = core.motor_groups["flight_R"][0]
    motor = None
    for _ in range(12):
        motor = core._motor_activity(
            np.asarray(
                [left_flight, right_flight],
                dtype=np.int64,
            )
        )
    assert motor is not None
    print("bilateral DNg02 activity -> flight", motor.flight)
    if motor.flight <= 0.02:
        raise SystemExit(
            "FAIL: DNg02 activity produces no flight-thrust drive"
        )

    # End-to-end raw-retina challenge. This uses the same graded eye route as
    # the desktop app and deliberately avoids FeatureDetectors. A moving,
    # expanding dark object must produce some translational motor output;
    # otherwise Flybit can rotate in place while never actually moving.
    max_forward = 0.0
    max_backward = 0.0
    max_escape = 0.0
    max_flight = 0.0
    max_steer = 0.0

    # Adapt to blank background.
    for _ in range(20):
        snap = core.step_visual_target(None)
        max_forward = max(max_forward, snap.motor.forward)

    for i in range(220):
        phase = i / 219.0
        center = -0.72 + 1.44 * phase
        # Slow expansion over the sweep creates natural retinal looming while
        # remaining pure image geometry.
        half_width = 0.025 + 0.30 * phase
        snap = core.step_visual_target(
            center,
            half_width,
        )
        max_forward = max(max_forward, snap.motor.forward)
        max_backward = max(max_backward, snap.motor.backward)
        max_escape = max(max_escape, snap.motor.escape)
        max_flight = max(max_flight, snap.motor.flight)
        max_steer = max(
            max_steer,
            abs(snap.motor.steering),
        )

    print(
        "raw-retina max motor:",
        f"forward={max_forward:.4f}",
        f"backward={max_backward:.4f}",
        f"escape={max_escape:.4f}",
        f"flight={max_flight:.4f}",
        f"steer={max_steer:.4f}",
    )

    translational = max(
        max_forward,
        max_backward,
        max_escape,
        max_flight,
    )
    if translational <= 0.005:
        raise SystemExit(
            "FAIL: raw retina reaches no translational motor output"
        )

    print("PASS: MaleCNS motor groups and raw-retina decoder are usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
