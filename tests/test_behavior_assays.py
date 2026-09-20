from __future__ import annotations
import unittest

from flybit.assays import MODES, run_assay, run_behavior_assays


class BehaviorAssayTest(unittest.TestCase):
    def test_all_required_assays_and_metrics_are_reported(self):
        report = run_behavior_assays(seed=7, duration=2)
        self.assertEqual(set(report["assays"]), {"baseline", "looming", "food", "edge", "flight_landing", "sleep_startle", "learning"})
        for assay in report["assays"].values():
            self.assertEqual(set(assay["mode_ratios"]), set(MODES))
            self.assertAlmostEqual(sum(assay["mode_ratios"].values()), 1, places=5)

    def test_looming_startle_latency_is_bounded(self):
        report = run_assay("looming", seed=4, duration=8)
        self.assertIsNotNone(report.startle_latency_ms)
        self.assertLess(report.startle_latency_ms, 250)

    def test_deterministic_regression_tolerance(self):
        a = run_assay("baseline", seed=13, duration=4)
        b = run_assay("baseline", seed=13, duration=4)
        self.assertEqual(a.mode_ratios, b.mode_ratios)
        self.assertAlmostEqual(a.walking_speed, b.walking_speed, delta=.01)

if __name__ == "__main__": unittest.main()
