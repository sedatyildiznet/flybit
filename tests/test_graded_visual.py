from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from flybrain import FlyBrain
from flybrain.eyes import Blob, Eyes


class GradedVisualTest(unittest.TestCase):
    def make_brain(self) -> FlyBrain:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        data = Path(tmp.name)

        # Three-cell miniature of the early ON route:
        # photoreceptor (histamine, inhibitory) -> L1
        # L1 (glutamate, inhibitory) -> downstream spiking neuron.
        #
        # A light increment depolarizes the photoreceptor, hyperpolarizes L1,
        # then reduced/inverted L1 output disinhibits the downstream cell.
        W = sparse.csr_matrix(
            (
                np.array([-1.0, -1.0], np.float32),
                (
                    np.array([1, 2]),
                    np.array([0, 1]),
                ),
            ),
            shape=(3, 3),
            dtype=np.float32,
        )
        sparse.save_npz(data / "weights.npz", W)

        np.savez(
            data / "brain.npz",
            ids=np.array([1, 2, 3], np.int64),
            visual=np.array([0], np.int32),
            azimuth=np.array([0.0], np.float32),
            cell_type=np.array(["R1-6", "L1", "DNtest"]),
            side=np.array(["L", "L", "L"]),
            positions=np.full((3, 3), np.nan, np.float32),
            superclass=np.array(
                ["sensory", "visual_interneuron", "descending_neuron"]
            ),
        )

        brain = FlyBrain(
            data=data,
            device="cpu",
            seed=1,
            graded_visual=True,
        )
        brain.tonic = 0.0
        brain.noise_hz = 0.0
        brain.eye_gain = 1.0
        return brain

    def test_graded_relay_reaches_spiking_downstream(self):
        brain = self.make_brain()

        fired1 = brain.step(
            eye_drive=np.array([1.0], np.float32)
        )
        fired2 = brain.step(
            eye_drive=np.array([1.0], np.float32)
        )

        # L1 must be hyperpolarized by light via the negative histamine edge.
        self.assertLess(float(brain.v[1, 0]), 0.0)

        fired3 = brain.step(
            eye_drive=np.array([1.0], np.float32)
        )

        self.assertNotIn(0, fired1)
        self.assertNotIn(1, fired1)
        self.assertNotIn(0, fired2)
        self.assertNotIn(1, fired2)
        self.assertIn(2, fired3)

    def test_dark_contrast_depolarizes_l1(self):
        brain = self.make_brain()

        brain.step(
            eye_drive=np.array([-1.0], np.float32)
        )
        brain.step(
            eye_drive=np.array([-1.0], np.float32)
        )

        self.assertGreater(float(brain.v[1, 0]), 0.0)

    def test_contrast_transducer_is_signed_and_adapting(self):
        eyes = Eyes(np.array([-0.5, 0.5], np.float32))

        first = eyes.contrast_drive([])
        self.assertTrue(np.allclose(first, 0.0))

        dark = eyes.contrast_drive(
            [Blob(center=-0.5, half_width=0.1, darkness=1.0)]
        )
        self.assertLess(float(dark[0]), 0.0)
        self.assertAlmostEqual(float(dark[1]), 0.0, places=6)

        # Repeated identical input should adapt toward zero magnitude.
        latest = dark
        for _ in range(20):
            latest = eyes.contrast_drive(
                [Blob(center=-0.5, half_width=0.1, darkness=1.0)]
            )
        self.assertLess(abs(float(latest[0])), abs(float(dark[0])))


if __name__ == "__main__":
    unittest.main()
