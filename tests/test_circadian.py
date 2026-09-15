from __future__ import annotations

import unittest

from flybit.circadian import CircadianModel
from flybit.state import FlybitState


class CircadianModelTest(unittest.TestCase):
    def make_state(self) -> FlybitState:
        return FlybitState(
            created_at="2026-01-01T00:00:00+00:00",
            sleep_pressure=0.50,
        )

    def test_dawn_has_more_wake_drive_than_deep_night(self):
        dawn = CircadianModel(self.make_state())
        dawn.tick(
            0.0,
            motor_load=0.0,
            ambient_luminance=0.6,
            local_hour=8.0,
        )

        night = CircadianModel(self.make_state())
        night.tick(
            0.0,
            motor_load=0.0,
            ambient_luminance=0.05,
            local_hour=3.0,
        )

        self.assertGreater(
            dawn.snapshot().wake_drive,
            night.snapshot().wake_drive,
        )
        self.assertLess(
            dawn.snapshot().rest_drive,
            night.snapshot().rest_drive,
        )

    def test_quiet_night_recovers_sleep_pressure(self):
        state = self.make_state()
        model = CircadianModel(state)
        before = state.sleep_pressure
        model.tick(
            3600.0,
            motor_load=0.0,
            ambient_luminance=0.0,
            local_hour=3.0,
        )
        self.assertLess(state.sleep_pressure, before)

    def test_active_day_builds_sleep_pressure(self):
        state = self.make_state()
        model = CircadianModel(state)
        before = state.sleep_pressure
        model.tick(
            3600.0,
            motor_load=1.0,
            ambient_luminance=0.8,
            local_hour=14.0,
        )
        self.assertGreater(state.sleep_pressure, before)


if __name__ == "__main__":
    unittest.main()
