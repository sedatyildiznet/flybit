from __future__ import annotations

import unittest

from flybit.motion import (
    FlyBodyState,
    FlyKinematics,
    MotorActivity,
)


class MotionBridgeTest(unittest.TestCase):
    def test_forward_dn_moves_on_flat_plane(self):
        body = FlyBodyState(
            x=200.0,
            y=200.0,
            heading=0.0,
        )
        model = FlyKinematics(body)

        for _ in range(25):
            model.update(
                MotorActivity(
                    forward_left=1.0,
                    forward_right=1.0,
                ),
                [],
                (0.0, 0.0, 800.0, 600.0),
                dt=0.020,
            )

        self.assertGreater(body.x, 200.0)
        self.assertAlmostEqual(
            body.y,
            200.0,
            delta=1.0,
        )
        self.assertFalse(body.airborne)

    def test_escape_dn_starts_planar_flight_burst(self):
        body = FlyBodyState(
            x=200.0,
            y=200.0,
            heading=0.0,
        )
        model = FlyKinematics(body)

        events = []
        for _ in range(8):
            events.extend(
                model.update(
                    MotorActivity(
                        escape_left=1.0,
                        escape_right=1.0,
                    ),
                    [],
                    (0.0, 0.0, 800.0, 600.0),
                    dt=0.020,
                )
            )

        self.assertTrue(body.airborne)
        self.assertGreater(body.x, 200.0)
        self.assertTrue(
            any(
                event.kind == "takeoff"
                for event in events
            )
        )

    def test_no_gravity_drift_without_motor_output(self):
        body = FlyBodyState(
            x=320.0,
            y=240.0,
            heading=1.2,
        )
        model = FlyKinematics(body)

        for _ in range(100):
            model.update(
                MotorActivity(),
                [],
                (0.0, 0.0, 800.0, 600.0),
                dt=0.020,
            )

        self.assertAlmostEqual(
            body.x,
            320.0,
            delta=0.01,
        )
        self.assertAlmostEqual(
            body.y,
            240.0,
            delta=0.01,
        )

    def test_screen_edge_does_not_flip_heading(self):
        body = FlyBodyState(
            x=787.0,
            y=300.0,
            heading=0.0,
            vx=300.0,
        )
        model = FlyKinematics(body)

        model.update(
            MotorActivity(),
            [],
            (0.0, 0.0, 800.0, 600.0),
            dt=0.050,
        )

        self.assertAlmostEqual(
            body.heading,
            0.0,
            delta=0.01,
        )
        self.assertLessEqual(body.x, 788.0)
        self.assertGreaterEqual(body.vx, 0.0)


if __name__ == "__main__":
    unittest.main()
