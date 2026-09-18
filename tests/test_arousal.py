from __future__ import annotations

import unittest

from flybit.arousal import ThreatArousalModel


class ThreatArousalModelTest(unittest.TestCase):
    def test_escape_activity_creates_sensitization(self):
        model = ThreatArousalModel(decay_seconds=10.0)
        model.tick(0.020, escape_drive=1.0)
        self.assertGreater(
            model.snapshot().threat_arousal,
            0.9,
        )

    def test_sensory_threat_raises_arousal_before_escape(self):
        model = ThreatArousalModel(decay_seconds=10.0)
        model.tick(
            0.020,
            escape_drive=0.0,
            sensory_threat=0.8,
        )
        self.assertGreater(
            model.snapshot().threat_arousal,
            0.4,
        )

    def test_arousal_decays_without_new_escape(self):
        model = ThreatArousalModel(decay_seconds=10.0)
        model.tick(0.020, escape_drive=1.0)
        before = model.snapshot().threat_arousal
        model.tick(10.0, escape_drive=0.0)
        after = model.snapshot().threat_arousal
        self.assertLess(after, before)
        self.assertGreater(after, 0.0)


if __name__ == "__main__":
    unittest.main()
