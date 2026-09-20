from __future__ import annotations
import unittest

from flybit.world import Surface, query_local_geometry


class LocalWorldGeometryTest(unittest.TestCase):
    def test_corner_has_two_near_segments(self):
        surface = Surface(1, 100, 400, 100, 300, title="ignored", z_order=2)
        local = query_local_geometry([surface], 104, 105, 20, (0, 0, 800, 600))
        edges = {edge.edge for edge in local.edges if edge.surface_id == 1}
        self.assertTrue({"left", "top"}.issubset(edges))
        self.assertLess(local.corner_distance, 7)

    def test_z_order_and_continuity_are_geometric(self):
        lower = Surface(1, 0, 300, 0, 300, title="A", z_order=4)
        upper = Surface(2, 100, 200, 100, 200, title="B", z_order=0)
        local = query_local_geometry([lower, upper], 102, 150, 24, (0, 0, 800, 600))
        self.assertEqual(local.support_id, 2)
        self.assertEqual(local.overlap_depth, 2)
        self.assertLess(local.support_continuity, 1.0)

if __name__ == "__main__": unittest.main()
