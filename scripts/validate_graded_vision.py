"""Validate graded vision against the real MaleCNS data.

This is intentionally not a behavioural benchmark. It checks the specific
failure mode Flybit is fixing: raw photoreceptor contrast must create continuous
state in L1/L2/L3 and that graded state must produce non-zero synaptic current
into downstream non-graded neurons through the measured connectome.
"""
from __future__ import annotations

import numpy as np

from flybrain import FlyBrain
from flybrain.eyes import Blob, Eyes


def rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values.astype(np.float64)))))


def main() -> int:
    brain = FlyBrain(
        device="cpu",
        seed=64,
        graded_visual=True,
    )
    eyes = Eyes(brain.azimuth)

    # Establish the adapted background first.
    brain.step(
        eye_drive=eyes.contrast_drive(
            [],
            dt=brain.dt,
        )
    )

    max_retina = 0.0
    max_lamina = 0.0
    max_downstream_analog = 0.0
    visual_projection_spikes = 0

    visual_projection = brain.cells(
        ["visual_projection"]
    )
    vp_mask = np.zeros(brain.n, dtype=np.bool_)
    vp_mask[visual_projection] = True

    # Sweep a dark target across the visual field. This is raw scene geometry,
    # not a looming/target classifier.
    for center in np.linspace(-0.8, 0.8, 60):
        drive = eyes.contrast_drive(
            [
                Blob(
                    center=float(center),
                    half_width=0.05,
                    darkness=1.0,
                )
            ],
            dt=brain.dt,
        )
        fired = np.asarray(
            brain.step(eye_drive=drive),
            dtype=np.int64,
        )

        max_retina = max(
            max_retina,
            rms(
                brain.membrane_values(
                    ["R1-6", "R7", "R8"]
                )
            ),
        )
        max_lamina = max(
            max_lamina,
            rms(
                brain.membrane_values(
                    ["L1", "L2", "L3"]
                )
            ),
        )

        analog = brain.graded_synaptic_input()
        if brain.xp is not np:
            analog = analog.get()
        analog = np.asarray(analog)

        if len(brain.spiking):
            max_downstream_analog = max(
                max_downstream_analog,
                float(
                    np.max(
                        np.abs(
                            analog[
                                brain.spiking
                            ]
                        )
                    )
                ),
            )

        if fired.size:
            visual_projection_spikes += int(
                vp_mask[fired].sum()
            )

    print(
        "graded cells:",
        len(brain.graded),
    )
    print(
        "max photoreceptor RMS:",
        f"{max_retina:.6f}",
    )
    print(
        "max L1/L2/L3 RMS:",
        f"{max_lamina:.6f}",
    )
    print(
        "max graded current into spiking cells:",
        f"{max_downstream_analog:.6f}",
    )
    print(
        "visual projection spikes:",
        visual_projection_spikes,
    )

    if max_retina <= 1e-3:
        raise SystemExit(
            "FAIL: raw scene did not drive photoreceptors"
        )
    if max_lamina <= 1e-4:
        raise SystemExit(
            "FAIL: graded signal did not reach L1/L2/L3"
        )
    if max_downstream_analog <= 1e-5:
        raise SystemExit(
            "FAIL: L1/L2/L3 analog state did not leave the first visual relay"
        )

    print(
        "PASS: photoreceptor -> lamina -> downstream analog relay is active"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
