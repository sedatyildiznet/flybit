from __future__ import annotations

import unittest

from flybit.world import Surface, support_at


class SurfaceGeometryTest(unittest.TestCase):
    def test_contains_uses_full_window_rectangle(self):
        surface = Surface(
            id=1,
            left=100.0,
            right=500.0,
            top=80.0,
            bottom=400.0,
            title="Window",
            z_order=2,
        )
        self.assertTrue(surface.contains(200.0, 200.0))
        self.assertFalse(surface.contains(20.0, 20.0))

    def test_support_at_prefers_topmost_z_order(self):
        back = Surface(
            id=1,
            left=0.0,
            right=500.0,
            top=0.0,
            bottom=500.0,
            title="Back",
            z_order=8,
        )
        front = Surface(
            id=2,
            left=100.0,
            right=400.0,
            top=100.0,
            bottom=400.0,
            title="Front",
            z_order=1,
        )
        support = support_at([back, front], 200.0, 200.0)
        self.assertIsNotNone(support)
        assert support is not None
        self.assertEqual(support.id, 2)

    def test_no_window_means_desktop_plane(self):
        self.assertIsNone(support_at([], 200.0, 200.0))


if __name__ == "__main__":
    unittest.main()
