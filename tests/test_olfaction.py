from __future__ import annotations

import unittest

from flybit.olfaction import FoodOdorModel


class FoodOdorModelTest(unittest.TestCase):
    def test_no_food_has_zero_odor(self):
        model = FoodOdorModel()
        snap = model.sample(
            x=0.0,
            y=0.0,
            heading=0.0,
            food=None,
            hunger_drive=1.0,
        )
        self.assertEqual(snap.mean, 0.0)
        self.assertIsNone(snap.food_distance)

    def test_bilateral_gradient_points_toward_off_axis_food(self):
        model = FoodOdorModel()
        snap = model.sample(
            x=0.0,
            y=0.0,
            heading=0.0,
            food=(80.0, 80.0, 1.0),
            hunger_drive=1.0,
        )
        self.assertGreater(snap.left, snap.right)
        self.assertLess(snap.gradient, 0.0)

    def test_hunger_changes_salience_not_physical_concentration(self):
        model = FoodOdorModel()
        fed = model.sample(
            x=0.0,
            y=0.0,
            heading=0.0,
            food=(100.0, 0.0, 1.0),
            hunger_drive=0.0,
        )
        hungry = model.sample(
            x=0.0,
            y=0.0,
            heading=0.0,
            food=(100.0, 0.0, 1.0),
            hunger_drive=1.0,
        )
        self.assertAlmostEqual(fed.mean, hungry.mean)
        self.assertGreater(hungry.salience, fed.salience)


if __name__ == "__main__":
    unittest.main()
