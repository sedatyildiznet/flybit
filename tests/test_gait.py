from __future__ import annotations

import unittest

from flybit.gait import SixLegGait, TRIPOD_A, TRIPOD_B


class SixLegGaitTest(unittest.TestCase):
    def test_modified_tripod_alternates_stance_groups(self):
        gait = SixLegGait()
        snapshots = []
        for _ in range(20):
            snapshots.append(
                gait.update(
                    0.020,
                    speed_norm=0.75,
                    turn_rate=0.0,
                )
            )

        saw_a = False
        saw_b = False
        for snap in snapshots:
            stance = {leg.name for leg in snap.legs if leg.stance}
            if len(stance & TRIPOD_A) >= 2:
                saw_a = True
            if len(stance & TRIPOD_B) >= 2:
                saw_b = True
        self.assertTrue(saw_a)
        self.assertTrue(saw_b)

    def test_stationary_fly_uses_six_supporting_legs(self):
        gait = SixLegGait()
        snap = gait.update(
            0.020,
            speed_norm=0.0,
            turn_rate=0.0,
        )
        self.assertEqual(snap.stance_count, 6)
        self.assertAlmostEqual(snap.support_factor, 1.0)

    def test_landing_extends_all_airborne_legs(self):
        gait = SixLegGait()
        tucked = gait.update(
            0.020,
            speed_norm=0.0,
            airborne=True,
            landing_drive=0.0,
        )
        extended = gait.update(
            0.020,
            speed_norm=0.0,
            airborne=True,
            landing_drive=1.0,
        )
        self.assertLess(tucked.leg_extension, extended.leg_extension)
        tucked_span = sum(abs(p.foot_y) for p in tucked.legs)
        extended_span = sum(abs(p.foot_y) for p in extended.legs)
        self.assertGreater(extended_span, tucked_span)

    def test_head_grooming_repositions_front_legs(self):
        gait = SixLegGait()
        snap = gait.update(
            0.020,
            speed_norm=0.0,
            groom_target="eyes",
        )
        poses = {p.name: p for p in snap.legs}
        self.assertGreater(poses["LF"].foot_x, 7.0)
        self.assertGreater(poses["RF"].foot_x, 7.0)


if __name__ == "__main__":
    unittest.main()
