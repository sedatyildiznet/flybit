from __future__ import annotations

import unittest

from flybit.care import CareModel
from flybit.state import FlybitState


class CareModelTest(unittest.TestCase):
    def make_state(self) -> FlybitState:
        return FlybitState(
            created_at="2026-09-15T00:00:00+00:00",
            hunger=0.75,
        )

    def test_hunger_increases_without_selecting_behaviour(self):
        state = self.make_state()
        care = CareModel(state)
        before = state.hunger

        care.tick(3600.0)

        self.assertGreater(state.hunger, before)
        self.assertLessEqual(state.hunger, 1.0)

    def test_hunger_exposes_internal_homeostatic_drive(self):
        state = self.make_state()
        care = CareModel(state)
        drive = care.homeostatic_drive
        self.assertGreater(drive, 0.0)
        state.hunger = 0.1
        self.assertEqual(care.homeostatic_drive, 0.0)

    def test_food_requires_physical_contact(self):
        state = self.make_state()
        care = CareModel(state)
        care.place_food(200.0, 200.0)

        self.assertFalse(care.contact(20.0, 20.0))
        self.assertIsNotNone(care.food)
        self.assertEqual(state.feedings, 0)

    def test_food_contact_consumes_drop_and_reduces_hunger(self):
        state = self.make_state()
        care = CareModel(state)
        care.place_food(200.0, 200.0)

        consumed = care.contact(203.0, 202.0)

        self.assertTrue(consumed)
        self.assertIsNone(care.food)
        self.assertLess(state.hunger, 0.75)
        self.assertEqual(state.feedings, 1)
        self.assertIsNotNone(state.last_feed_at)
        self.assertEqual(care.feeding_signal, 1.0)


if __name__ == "__main__":
    unittest.main()
