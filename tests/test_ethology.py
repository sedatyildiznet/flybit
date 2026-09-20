from __future__ import annotations

from types import SimpleNamespace
import unittest

from flybit.ethology import EthologyModel, EthologyParameters
from flybit.motion import MotorActivity


def sensory(*, left=0.0, right=0.0, disturbance=0.0):
    return SimpleNamespace(
        retinal_loom_left=left,
        retinal_loom_right=right,
        mechanosensory_disturbance=disturbance,
    )


def odor(*, salience=0.0, mean=0.0, gradient=0.0):
    return SimpleNamespace(
        salience=salience,
        mean=mean,
        gradient=gradient,
    )


def boundary(*, strength=0.0, tangent=0.0, inward=1.57):
    return SimpleNamespace(
        strength=strength,
        tangent_heading=tangent,
        inward_heading=inward,
    )


class EthologyModelTest(unittest.TestCase):
    def test_parameters_expose_bout_ranges_and_seed_is_deterministic(self):
        params = EthologyParameters()
        self.assertLess(params.bouts["walk"].minimum, params.bouts["walk"].maximum)
        a, b = EthologyModel(seed=99), EthologyModel(seed=99)
        for model in (a, b):
            model._choose_ground_mode(hunger=.4, food_drive=.2, rest=.2,
                                      activity=.6, curiosity=.5, boundary_drive=.1)
        self.assertEqual(a.mode, b.mode)
        self.assertAlmostEqual(a.bout_remaining, b.bout_remaining)

    def test_recent_history_penalizes_repetitive_activation(self):
        model = EthologyModel(seed=3)
        model._history.extend(["turn"] * 12)
        self.assertEqual(len(model._history), 12)
    def test_left_loom_produces_directional_escape(self):
        model = EthologyModel(seed=1)
        snap = model.tick(
            0.020,
            neural=MotorActivity(),
            sensory=sensory(left=0.90),
            boldness=0.2,
        )

        self.assertEqual(snap.mode, "escape")
        self.assertGreater(snap.motor.escape_right, snap.motor.escape_left)
        self.assertGreater(snap.motor.steer_right, snap.motor.steer_left)
        self.assertGreater(snap.alertness, 0.0)

    def test_hungry_fly_uses_bilateral_odor_for_forage(self):
        model = EthologyModel(seed=2)
        model.bout_remaining = 0.0
        snap = model.tick(
            0.020,
            neural=MotorActivity(),
            odor=odor(salience=0.95, mean=0.65, gradient=0.50),
            hunger_drive=0.95,
            rest_drive=0.1,
        )

        self.assertEqual(snap.mode, "forage")
        self.assertGreater(snap.motor.forward, 0.0)
        self.assertGreater(snap.motor.steering, 0.0)
        self.assertGreater(snap.food_drive, 0.5)

    def test_strong_rest_drive_creates_quiescent_sleep_bout(self):
        model = EthologyModel(seed=3)
        model.bout_remaining = 0.0
        snap = model.tick(
            0.020,
            neural=MotorActivity(),
            rest_drive=0.90,
        )

        self.assertEqual(snap.mode, "sleep")
        self.assertTrue(snap.asleep)
        self.assertAlmostEqual(snap.motor.forward, 0.0)

    def test_strong_loom_wakes_sleep_into_escape(self):
        model = EthologyModel(seed=4)
        model.mode = "sleep"
        model.bout_remaining = 10.0
        snap = model.tick(
            0.020,
            neural=MotorActivity(),
            sensory=sensory(right=0.90),
            rest_drive=0.9,
        )

        self.assertEqual(snap.mode, "escape")
        self.assertFalse(snap.asleep)
        self.assertGreater(snap.motor.escape_left, 0.0)

    def test_grooming_is_competing_motor_program(self):
        model = EthologyModel(seed=5, grooming_need=0.95)
        model.bout_remaining = 0.0
        before = model.grooming_need
        snap = model.tick(
            0.020,
            neural=MotorActivity(),
            rest_drive=0.1,
            activity=0.4,
        )

        self.assertEqual(snap.mode, "groom")
        self.assertTrue(snap.grooming)
        self.assertLess(snap.grooming_need, before)
        self.assertAlmostEqual(snap.motor.forward, 0.0)

    def test_food_contact_enters_stationary_feeding_bout(self):
        model = EthologyModel(seed=7)
        model.begin_feeding(1.0)
        snap = model.tick(
            0.020,
            neural=MotorActivity(
                forward_left=0.08,
                forward_right=0.08,
            ),
        )

        self.assertEqual(snap.mode, "feed")
        self.assertAlmostEqual(snap.motor.forward, 0.0)

    def test_threat_interrupts_feeding_bout(self):
        model = EthologyModel(seed=8)
        model.begin_feeding(1.0)
        snap = model.tick(
            0.020,
            neural=MotorActivity(),
            sensory=sensory(left=0.95),
        )

        self.assertEqual(snap.mode, "escape")
        self.assertGreater(snap.motor.escape, 0.0)

    def test_airborne_looming_enters_landing_preparation(self):
        model = EthologyModel(seed=9)
        snap = None
        for _ in range(20):
            snap = model.tick(
                0.020,
                neural=MotorActivity(),
                sensory=SimpleNamespace(
                    retinal_loom_left=0.28,
                    retinal_loom_right=0.24,
                    mechanosensory_disturbance=0.0,
                    optic_flow=0.2,
                ),
                airborne=True,
            )
        assert snap is not None
        self.assertEqual(snap.mode, "landing")
        self.assertGreater(snap.landing_drive, 0.0)

    def test_boundary_cue_can_drive_boundary_following(self):
        model = EthologyModel(seed=10)
        model.bout_remaining = 0.0
        snap = model.tick(
            0.020,
            neural=MotorActivity(),
            boundary=boundary(strength=0.95, tangent=1.57, inward=0.0),
            heading=0.0,
            rest_drive=0.0,
            activity=0.2,
            curiosity=0.1,
        )
        self.assertEqual(snap.mode, "boundary")
        self.assertGreater(snap.motor.forward, 0.0)
        self.assertNotEqual(snap.motor.steering, 0.0)

    def test_sleep_deepens_with_time(self):
        model = EthologyModel(seed=11)
        model.mode = "sleep"
        model.bout_remaining = 30.0
        snap = None
        for _ in range(450):
            snap = model.tick(
                0.020,
                neural=MotorActivity(),
                rest_drive=0.95,
            )
        assert snap is not None
        self.assertEqual(snap.sleep_stage, "deep")

    def test_neural_motor_output_is_never_erased_outside_sleep(self):
        model = EthologyModel(seed=6)
        neural = MotorActivity(
            forward_left=0.7,
            forward_right=0.6,
            steer_left=0.2,
        )
        snap = model.tick(
            0.020,
            neural=neural,
            rest_drive=0.0,
        )

        self.assertGreaterEqual(snap.motor.forward_left, neural.forward_left)
        self.assertGreaterEqual(snap.motor.forward_right, neural.forward_right)
        self.assertGreaterEqual(snap.motor.steer_left, neural.steer_left)


if __name__ == "__main__":
    unittest.main()
