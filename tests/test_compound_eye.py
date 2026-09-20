from __future__ import annotations
import time
import unittest
import numpy as np

from flybit.compound_eye import CompoundEyeModel


class CompoundEyeTest(unittest.TestCase):
    def test_bilateral_fields_preserve_legacy_panorama(self):
        model = CompoundEyeModel(bins=128)
        snap = model.process(np.ones((4, 3, 128), dtype=np.float32) * .7)
        self.assertEqual(snap.left.luminance.shape, (4, 64))
        self.assertEqual(snap.right.luminance.shape, (4, 64))
        self.assertEqual(snap.male_cns_panorama.shape, (128,))
        self.assertAlmostEqual(float(snap.male_cns_panorama.mean()), .7, places=4)

    def test_vertical_and_radial_coherence_contribute_to_loom(self):
        model = CompoundEyeModel(bins=128)
        base = np.ones((4, 3, 128), dtype=np.float32) * .9
        model.process(base)
        expanded = base.copy(); expanded[:, :, 30:50] = .1
        snap = model.process(expanded)
        self.assertGreater(snap.motion.expansion, 0)
        self.assertGreater(snap.motion.radial_coherence, 0)

    def test_processing_budget(self):
        model = CompoundEyeModel()
        data = np.random.default_rng(2).random((4, 6, 384), dtype=np.float32)
        started = time.perf_counter()
        for _ in range(50): model.process(data)
        self.assertLess((time.perf_counter() - started) / 50, .02)

if __name__ == "__main__": unittest.main()
