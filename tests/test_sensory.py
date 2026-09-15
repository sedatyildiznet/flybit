from __future__ import annotations

import unittest

import numpy as np

from flybit.sensory import DesktopMotionModel
from flybit.state import normalize_display_name


class SensoryDynamicsTest(unittest.TestCase):
    def test_fast_approach_produces_looming_and_ttc(self):
        model = DesktopMotionModel(cursor_radius=14.0)
        panorama = np.linspace(0.2, 0.8, 64, dtype=np.float32)

        model.update(
            body_x=0.0,
            body_y=0.0,
            heading=0.0,
            cursor_x=300.0,
            cursor_y=0.0,
            luminance=panorama,
            timestamp=0.0,
        )
        approaching = model.update(
            body_x=0.0,
            body_y=0.0,
            heading=0.0,
            cursor_x=260.0,
            cursor_y=0.0,
            luminance=panorama,
            timestamp=0.020,
        )

        self.assertGreater(approaching.closing_speed, 0.0)
        self.assertGreater(approaching.looming_rate, 0.0)
        self.assertIsNotNone(approaching.time_to_collision)
        self.assertGreater(approaching.threat_salience, 0.0)

    def test_stationary_cursor_has_no_false_looming(self):
        model = DesktopMotionModel()
        panorama = np.ones(64, dtype=np.float32) * 0.5

        model.update(
            body_x=100.0,
            body_y=100.0,
            heading=0.0,
            cursor_x=300.0,
            cursor_y=100.0,
            luminance=panorama,
            timestamp=1.0,
        )
        stationary = model.update(
            body_x=100.0,
            body_y=100.0,
            heading=0.0,
            cursor_x=300.0,
            cursor_y=100.0,
            luminance=panorama,
            timestamp=1.020,
        )

        self.assertAlmostEqual(stationary.cursor_speed, 0.0)
        self.assertAlmostEqual(stationary.closing_speed, 0.0)
        self.assertAlmostEqual(stationary.looming_rate, 0.0)
        self.assertIsNone(stationary.time_to_collision)

    def test_dark_expansion_produces_retinal_loom_without_semantics(self):
        model = DesktopMotionModel()
        first = np.full(128, 0.9, dtype=np.float32)
        second = first.copy()
        first[28:32] = 0.1
        second[24:36] = 0.1

        model.update(
            body_x=0.0,
            body_y=0.0,
            heading=0.0,
            cursor_x=500.0,
            cursor_y=0.0,
            luminance=first,
            timestamp=1.0,
        )
        expanded = model.update(
            body_x=0.0,
            body_y=0.0,
            heading=0.0,
            cursor_x=500.0,
            cursor_y=0.0,
            luminance=second,
            timestamp=1.020,
        )

        self.assertGreater(expanded.retinal_loom_left, 0.0)
        self.assertAlmostEqual(expanded.retinal_loom_right, 0.0)

    def test_repeated_looming_habituates_but_keeps_response(self):
        model = DesktopMotionModel()
        narrow = np.full(128, 0.9, dtype=np.float32)
        wide = narrow.copy()
        narrow[28:32] = 0.1
        wide[24:36] = 0.1

        t = 3.0
        model.update(
            body_x=0.0,
            body_y=0.0,
            heading=0.0,
            cursor_x=500.0,
            cursor_y=0.0,
            luminance=narrow,
            timestamp=t,
        )
        t += 0.020
        first = model.update(
            body_x=0.0,
            body_y=0.0,
            heading=0.0,
            cursor_x=500.0,
            cursor_y=0.0,
            luminance=wide,
            timestamp=t,
        )

        latest = first
        for _ in range(160):
            t += 0.020
            model.update(
                body_x=0.0,
                body_y=0.0,
                heading=0.0,
                cursor_x=500.0,
                cursor_y=0.0,
                luminance=narrow,
                timestamp=t,
            )
            t += 0.020
            latest = model.update(
                body_x=0.0,
                body_y=0.0,
                heading=0.0,
                cursor_x=500.0,
                cursor_y=0.0,
                luminance=wide,
                timestamp=t,
            )

        self.assertGreater(latest.loom_habituation, 0.2)
        self.assertGreater(latest.retinal_loom_left, 0.0)
        self.assertLess(
            latest.retinal_loom_left,
            first.retinal_loom_left,
        )

    def test_panorama_shift_is_visible_as_optic_flow(self):
        model = DesktopMotionModel()
        x = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        first = (0.5 + 0.3 * np.sin(x) + 0.1 * np.sin(3 * x)).astype(
            np.float32
        )
        second = np.roll(first, 7)

        model.update(
            body_x=0.0,
            body_y=0.0,
            heading=0.0,
            cursor_x=500.0,
            cursor_y=0.0,
            luminance=first,
            timestamp=2.0,
        )
        shifted = model.update(
            body_x=0.0,
            body_y=0.0,
            heading=0.0,
            cursor_x=500.0,
            cursor_y=0.0,
            luminance=second,
            timestamp=2.020,
        )

        self.assertGreater(abs(shifted.optic_flow), 0.01)


class OrganismIdentityTest(unittest.TestCase):
    def test_name_is_compact_and_has_default(self):
        self.assertEqual(normalize_display_name("   Ada   Fly   "), "Ada Fly")
        self.assertEqual(normalize_display_name(""), "Flybit")
        self.assertLessEqual(len(normalize_display_name("x" * 100)), 32)


if __name__ == "__main__":
    unittest.main()
