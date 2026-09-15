from __future__ import annotations

import unittest

from flybit.motion import (
    FlyBodyState,
    FlyKinematics,
    MotorActivity,
)
from flybit.world import Surface


class MotionBridgeTest(unittest.TestCase):
    def test_forward_dn_moves_landed_body(self):
        body = FlyBodyState(
            x=200.0,
            y=91.0,
            heading=0.0,
            landed_surface=1,
        )
        model = FlyKinematics(body)
        surface = Surface(
            id=1,
            left=100.0,
            right=500.0,
            top=100.0,
            title="test window",
        )

        for _ in range(20):
            model.update(
                MotorActivity(
                    forward_left=1.0,
                    forward_right=1.0,
                ),
                [surface],
                (0.0, 0.0, 800.0, 600.0),
                dt=0.020,
            )

        self.assertGreater(body.x, 200.0)
        self.assertEqual(body.landed_surface, 1)
        self.assertAlmostEqual(body.y, 91.0, places=4)

    def test_escape_dn_causes_takeoff_without_environment_rule(self):
        body = FlyBodyState(
            x=200.0,
            y=91.0,
            heading=0.0,
            landed_surface=1,
        )
        model = FlyKinematics(body)
        surface = Surface(
            id=1,
            left=100.0,
            right=500.0,
            top=100.0,
            title="test window",
        )

        events = model.update(
            MotorActivity(
                escape_left=1.0,
                escape_right=1.0,
            ),
            [surface],
            (0.0, 0.0, 800.0, 600.0),
            dt=0.020,
        )

        self.assertIsNone(body.landed_surface)
        self.assertLess(body.vy, 0.0)
        self.assertTrue(
            any(event.kind == "takeoff" for event in events)
        )

    def test_falling_body_lands_on_window_top(self):
        body = FlyBodyState(
            x=250.0,
            y=82.0,
            heading=0.0,
            vx=0.0,
            vy=300.0,
        )
        model = FlyKinematics(body)
        surface = Surface(
            id=7,
            left=100.0,
            right=500.0,
            top=100.0,
            title="editor",
        )

        events = model.update(
            MotorActivity(),
            [surface],
            (0.0, 0.0, 800.0, 600.0),
            dt=0.040,
        )

        self.assertEqual(body.landed_surface, 7)
        self.assertAlmostEqual(body.y, 91.0, places=4)
        self.assertEqual(body.vy, 0.0)
        self.assertTrue(
            any(event.kind == "land" for event in events)
        )


if __name__ == "__main__":
    unittest.main()
