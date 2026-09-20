from __future__ import annotations
import unittest

from flybit.sleep import SleepEpisodeModel


class SleepEpisodeTest(unittest.TestCase):
    def test_episode_deepens_and_reduces_responsiveness(self):
        model = SleepEpisodeModel()
        early = model.tick(1, sleeping=True)
        deep = model.tick(8, sleeping=True)
        self.assertEqual(early.stage, "drowsy")
        self.assertEqual(deep.stage, "deep")
        self.assertLess(deep.responsiveness, early.responsiveness)

    def test_threat_wakes_deep_sleep(self):
        model = SleepEpisodeModel(); model.tick(10, sleeping=True)
        snap = model.tick(.02, sleeping=True, threat=.8)
        self.assertTrue(snap.woke)

    def test_long_episode_has_brief_micro_awake_stage(self):
        model = SleepEpisodeModel()
        seen = False
        for _ in range(4000):
            seen |= model.tick(.02, sleeping=True).stage == "micro-awake"
        self.assertTrue(seen)

    def test_offline_progress_is_bounded_and_recovers(self):
        model = SleepEpisodeModel()
        self.assertLess(model.offline_progress(8 * 3600, .9), .9)
        self.assertGreaterEqual(model.offline_progress(10**9, 1), 0)

if __name__ == "__main__": unittest.main()
