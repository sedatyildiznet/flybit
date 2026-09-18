from __future__ import annotations

import unittest

from flybit.motion import (
    FlyBodyState,
    FlyKinematics,
    MotorActivity,
)
from flybit.world import Surface


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
        self.assertGreater(body.altitude, 0.0)
        self.assertGreater(body.x, 200.0)
        self.assertTrue(
            any(
                event.kind == "takeoff"
                for event in events
            )
        )

    def test_takeoff_has_preload_before_airborne_phase(self):
        body = FlyBodyState(
            x=200.0,
            y=200.0,
            heading=0.0,
        )
        model = FlyKinematics(body)
        events = model.update(
            MotorActivity(
                escape_left=1.0,
                escape_right=1.0,
            ),
            [],
            (0.0, 0.0, 800.0, 600.0),
            dt=0.020,
        )
        self.assertFalse(body.airborne)
        self.assertTrue(
            any(event.kind == "takeoff_prepare" for event in events)
        )
        self.assertGreater(model.biomechanics().takeoff_preload, 0.0)

    def test_asymmetric_escape_changes_heading(self):
        body = FlyBodyState(
            x=200.0,
            y=200.0,
            heading=0.0,
        )
        model = FlyKinematics(body)

        for _ in range(5):
            model.update(
                MotorActivity(
                    escape_right=1.0,
                    escape_left=0.0,
                ),
                [],
                (0.0, 0.0, 800.0, 600.0),
                dt=0.020,
            )

        self.assertTrue(body.airborne)
        self.assertGreater(body.heading, 0.20)

    def test_flight_lands_via_virtual_altitude_not_screen_y(self):
        body = FlyBodyState(
            x=300.0,
            y=240.0,
            heading=0.0,
        )
        model = FlyKinematics(body)

        for _ in range(8):
            model.update(
                MotorActivity(
                    escape_left=1.0,
                    escape_right=1.0,
                ),
                [],
                (0.0, 0.0, 800.0, 600.0),
                dt=0.020,
            )

        takeoff_y = body.y
        self.assertTrue(body.airborne)
        self.assertGreater(body.altitude, 0.0)

        for _ in range(260):
            model.update(
                MotorActivity(),
                [],
                (0.0, 0.0, 800.0, 600.0),
                dt=0.020,
            )

        self.assertFalse(body.airborne)
        self.assertAlmostEqual(body.altitude, 0.0, delta=0.001)
        self.assertAlmostEqual(body.y, takeoff_y, delta=40.0)

    def test_dng02_extends_existing_flight(self):
        body = FlyBodyState(
            x=220.0,
            y=220.0,
            heading=0.0,
            airborne=True,
            flight_energy=0.05,
        )
        model = FlyKinematics(body)

        for _ in range(20):
            model.update(
                MotorActivity(
                    flight_left=1.0,
                    flight_right=1.0,
                ),
                [],
                (0.0, 0.0, 800.0, 600.0),
                dt=0.020,
            )

        self.assertTrue(body.airborne)
        self.assertGreater(body.x, 220.0)
        self.assertGreater(body.flight_energy, 0.05)

    def test_landing_drive_extends_legs_and_descends(self):
        body = FlyBodyState(
            x=300.0,
            y=240.0,
            heading=0.0,
            airborne=True,
            altitude=40.0,
            vertical_velocity=0.0,
            flight_energy=1.0,
        )
        model = FlyKinematics(body)
        before = body.altitude
        for _ in range(20):
            model.update(
                MotorActivity(
                    flight_left=0.10,
                    flight_right=0.10,
                ),
                [],
                (0.0, 0.0, 800.0, 600.0),
                dt=0.020,
                landing_drive=1.0,
            )
        bio = model.biomechanics()
        self.assertGreater(bio.leg_extension, 0.8)
        self.assertLess(body.altitude, before)

    def test_corner_can_leave_when_neural_steering_turns_inward(self):
        body = FlyBodyState(
            x=12.0,
            y=9.0,
            heading=-2.4,
        )
        model = FlyKinematics(body)

        for _ in range(240):
            model.update(
                MotorActivity(
                    forward_left=0.65,
                    forward_right=0.65,
                    steer_right=0.8,
                ),
                [],
                (0.0, 0.0, 800.0, 600.0),
                dt=0.020,
            )

        self.assertGreater(body.x, 20.0)
        self.assertGreater(body.y, 15.0)

    def test_grounded_body_reports_window_substrate(self):
        body = FlyBodyState(
            x=200.0,
            y=200.0,
            heading=0.0,
        )
        model = FlyKinematics(body)
        surface = Surface(
            id=42,
            left=100.0,
            right=400.0,
            top=100.0,
            bottom=350.0,
            title="Test Window",
            z_order=0,
        )

        events = model.update(
            MotorActivity(),
            [surface],
            (0.0, 0.0, 800.0, 600.0),
            dt=0.020,
        )

        self.assertEqual(body.support_id, 42)
        self.assertEqual(body.support_title, "Test Window")
        self.assertTrue(
            any(event.kind == "surface_contact" for event in events)
        )

    def test_biomechanics_reports_gait_and_load(self):
        body = FlyBodyState(x=200.0, y=200.0, heading=0.0)
        model = FlyKinematics(body)
        for _ in range(40):
            model.update(
                MotorActivity(forward_left=1.0, forward_right=1.0),
                [],
                (0.0, 0.0, 800.0, 600.0),
                dt=0.020,
            )
        bio = model.biomechanics()
        self.assertGreater(bio.speed, 0.0)
        self.assertGreater(bio.locomotor_load, 0.0)
        self.assertGreaterEqual(bio.gait_phase, 0.0)

    def test_low_vitality_reduces_motion_capacity(self):
        strong = FlyKinematics(FlyBodyState(x=200.0, y=200.0))
        weak = FlyKinematics(FlyBodyState(x=200.0, y=200.0))
        motor = MotorActivity(forward_left=1.0, forward_right=1.0)
        for _ in range(40):
            strong.update(motor, [], (0, 0, 800, 600), dt=0.020, physiology_gain=1.0)
            weak.update(motor, [], (0, 0, 800, 600), dt=0.020, physiology_gain=0.25)
        self.assertGreater(strong.biomechanics().speed, weak.biomechanics().speed)

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

    def test_screen_edge_reflects_once_without_spin_loop(self):
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
        first_heading = body.heading

        for _ in range(40):
            model.update(
                MotorActivity(),
                [],
                (0.0, 0.0, 800.0, 600.0),
                dt=0.020,
            )

        self.assertLessEqual(body.x, 788.0)
        self.assertLess(body.vx, 5.0)
        self.assertAlmostEqual(
            abs(first_heading),
            3.141592653589793,
            delta=0.05,
        )
        wrapped_delta = (
            body.heading
            - first_heading
            + 3.141592653589793
        ) % (2.0 * 3.141592653589793) - 3.141592653589793
        self.assertAlmostEqual(
            wrapped_delta,
            0.0,
            delta=0.05,
        )


if __name__ == "__main__":
    unittest.main()
