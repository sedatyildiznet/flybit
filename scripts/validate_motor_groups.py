"""Validate Flybit motor read-outs against the bundled MaleCNS data."""
from __future__ import annotations

import numpy as np

from flybit.neural import FlybitNeuralCore
from flybit.sensory import DesktopMotionModel


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

    # Controlled temporal-looming assay. First verify that an actually
    # expanding raw luminance silhouette carries more sustained loom than a
    # stationary silhouette. No cursor identity or semantic label is used.
    source_azimuth = np.linspace(
        -1.0,
        1.0,
        384,
        endpoint=False,
        dtype=np.float32,
    )

    def panorama(half_width: float) -> np.ndarray:
        lum = np.full(
            len(source_azimuth),
            0.90,
            dtype=np.float32,
        )
        lum[np.abs(source_azimuth) <= half_width] = 0.06
        return lum

    static_motion = DesktopMotionModel()
    static_sum = 0.0
    t = 0.0
    for i in range(90):
        t += 0.020
        dyn = static_motion.update(
            body_x=0.0,
            body_y=0.0,
            heading=0.0,
            cursor_x=5000.0,
            cursor_y=5000.0,
            luminance=panorama(0.08),
            timestamp=t,
        )
        if i > 3:
            static_sum += (
                dyn.retinal_loom_left
                + dyn.retinal_loom_right
            )

    expanding_motion = DesktopMotionModel()
    expanding_frames: list[
        tuple[np.ndarray, float, float]
    ] = []
    expanding_sum = 0.0
    t = 0.0
    for i in range(90):
        phase = i / 89.0
        width = 0.018 + 0.31 * phase
        lum = panorama(width)
        t += 0.020
        dyn = expanding_motion.update(
            body_x=0.0,
            body_y=0.0,
            heading=0.0,
            cursor_x=5000.0,
            cursor_y=5000.0,
            luminance=lum,
            timestamp=t,
        )
        expanding_sum += (
            dyn.retinal_loom_left
            + dyn.retinal_loom_right
        )
        expanding_frames.append(
            (
                lum,
                dyn.retinal_loom_left,
                dyn.retinal_loom_right,
            )
        )

    print(
        "temporal loom signal:",
        f"static={static_sum:.4f}",
        f"expanding={expanding_sum:.4f}",
    )
    if expanding_sum <= static_sum + 0.05:
        raise SystemExit(
            "FAIL: expanding raw silhouette is not distinguished from static"
        )

    # Run those same raw-luminance frames through a fresh real MaleCNS core.
    # The modeled boundary may stimulate only LPLC2; escape remains a downstream
    # property of the measured network plus the existing DN decoder.
    del core
    loom_core = FlybitNeuralCore(
        device="cpu",
        seed=71,
    )
    blank = np.full(
        len(source_azimuth),
        0.90,
        dtype=np.float32,
    )

    baseline_loom_spikes = 0
    for _ in range(30):
        loom_core.set_visual_motion(0.0, 0.0)
        snap = loom_core.step_visual_luminance(
            blank,
            source_azimuth,
        )
        baseline_loom_spikes += snap.looming_spikes

    expanding_loom_spikes = 0
    expanding_escape = 0.0
    for lum, loom_left, loom_right in expanding_frames:
        loom_core.set_visual_motion(
            loom_left,
            loom_right,
        )
        snap = loom_core.step_visual_luminance(
            lum,
            source_azimuth,
        )
        expanding_loom_spikes += snap.looming_spikes
        expanding_escape = max(
            expanding_escape,
            snap.motor.escape,
        )

    baseline_rate = baseline_loom_spikes / 30.0
    expanding_rate = expanding_loom_spikes / len(expanding_frames)
    print(
        "real MaleCNS looming assay:",
        f"LPLC2 baseline/frame={baseline_rate:.3f}",
        f"expanding/frame={expanding_rate:.3f}",
        f"escape_max={expanding_escape:.4f}",
    )
    if expanding_rate <= baseline_rate:
        raise SystemExit(
            "FAIL: expanding raw retina does not increase LPLC2 activity"
        )
    if expanding_escape <= 0.0:
        raise SystemExit(
            "FAIL: expanding raw retina produces no DNp01 escape output"
        )

    print(
        "PASS: MaleCNS motor groups, retinal looming and raw-retina decoder "
        "are usable"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
