from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest

from flybit.memory import AssociativeMemory, SensoryFingerprint


class MemoryTest(unittest.TestCase):
    def fp(self, x=.2): return SensoryFingerprint(x, .1, 0, .4, .3, .2, 1)

    def test_reward_is_recalled_as_bounded_bias(self):
        memory = AssociativeMemory()
        for i in range(20): memory.observe(self.fp(), food_reward=1, now=100+i)
        bias = memory.recall(self.fp(), now=121)
        self.assertGreater(bias.food_bias, 0)
        self.assertLessEqual(bias.food_bias, .2)

    def test_atomic_persistence_migration_and_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            path.write_text(json.dumps({"schema": 1, "memories": [{"fingerprint": list(self.fp().vector()), "food": .5, "threat": 0, "safe_rest": 0, "updated_at": 1}]}))
            memory = AssociativeMemory(path, max_associations=8)
            self.assertEqual(len(memory.associations), 1)
            for i in range(30): memory.observe(self.fp(i / 10), neutral=1, now=10+i)
            memory.save()
            self.assertLessEqual(len(memory.associations), 8)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_corruption_is_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            path.write_text("not-json")
            memory = AssociativeMemory(path)
            self.assertEqual(memory.associations, [])
            self.assertTrue(path.with_suffix(".json.corrupt").exists())

if __name__ == "__main__": unittest.main()
