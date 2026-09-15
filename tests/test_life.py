from __future__ import annotations

import unittest

from flybit.life import LifeModel
from flybit.state import FlybitState


class LifeModelTest(unittest.TestCase):
    def make_state(self) -> FlybitState:
        return FlybitState(
            created_at="2026-09-15T00:00:00+00:00",
            hunger=0.4,
        )

    def test_traits_and_lifespan_are_stable(self):
        state = self.make_state()
        first = LifeModel(state).snapshot()
        second = LifeModel(state).snapshot()

        self.assertEqual(first.lifespan_days, second.lifespan_days)
        self.assertEqual(first.activity, second.activity)
        self.assertGreaterEqual(first.lifespan_days, 35.0)
        self.assertLessEqual(first.lifespan_days, 55.0)

    def test_metabolic_load_drains_energy(self):
        state = self.make_state()
        life = LifeModel(state)
        before = life.snapshot().energy
        life.tick(600.0, motor_load=1.0, hunger=1.0)
        self.assertLess(life.snapshot().energy, before)

    def test_feeding_restores_energy(self):
        state = self.make_state()
        life = LifeModel(state)
        life.tick(1200.0, motor_load=1.0, hunger=1.0)
        before = life.snapshot().energy
        life.feed()
        self.assertGreater(life.snapshot().energy, before)


if __name__ == "__main__":
    unittest.main()
