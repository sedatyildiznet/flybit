from __future__ import annotations

import unittest

from flybit.benchmark import EthogramRecorder, wtsb_category


class EthogramRecorderTest(unittest.TestCase):
    def test_records_mode_fraction_and_bouts(self):
        rec = EthogramRecorder()
        for _ in range(10):
            rec.update(0.1, mode="walk", speed=20.0, turn_rate=0.2)
        for _ in range(5):
            rec.update(0.1, mode="turn", speed=8.0, turn_rate=2.0)
        snap = rec.snapshot()

        self.assertGreater(snap.transitions, 0)
        self.assertAlmostEqual(
            snap.mode_fraction["walk"],
            2.0 / 3.0,
            delta=0.02,
        )
        self.assertGreater(snap.mean_speed, 0.0)
        self.assertGreater(snap.mean_abs_turn_rate, 0.0)

    def test_maps_to_wtsb_ethogram_categories(self):
        self.assertEqual(wtsb_category("idle"), "stop")
        self.assertEqual(wtsb_category("walk"), "curved_walk")
        self.assertEqual(wtsb_category("turn"), "sharp_turn")
        self.assertEqual(wtsb_category("boundary"), "boundary")


if __name__ == "__main__":
    unittest.main()
