from __future__ import annotations

import unittest

from flybit.gait import SixLegGait
from flybit.proprioception import ProprioceptionModel
from flybit.world import Surface, query_local_geometry


class ProprioceptionTest(unittest.TestCase):
    def setUp(self):
        self.bounds = (0.0, 0.0, 800.0, 600.0)
        self.surface = Surface(7, 100.0, 500.0, 100.0, 400.0, z_order=0)

    def test_six_grounded_legs_report_contact_and_load(self):
        snap = ProprioceptionModel().update(x=250, y=220, heading=0, altitude=0,
            vertical_velocity=0, legs=SixLegGait().snapshot.legs,
            surfaces=[self.surface], bounds=self.bounds, dt=0.02)
        self.assertEqual(snap.contact_count, 6)
        self.assertGreater(snap.total_load, 0.9)
        self.assertGreater(snap.stability, 0.7)

    def test_touchdown_requires_extended_physical_legs(self):
        model = ProprioceptionModel()
        air = SixLegGait().update(0.02, speed_norm=0, airborne=True, landing_drive=0)
        snap = model.update(x=250, y=220, heading=0, altitude=0.5,
            vertical_velocity=-20, legs=air.legs, surfaces=[self.surface], bounds=self.bounds, dt=0.02)
        self.assertFalse(snap.landing_contact)
        extended = SixLegGait().update(0.02, speed_norm=0, airborne=True, landing_drive=1)
        snap = model.update(x=250, y=220, heading=0, altitude=0,
            vertical_velocity=-20, legs=extended.legs, surfaces=[self.surface], bounds=self.bounds, dt=0.02)
        self.assertTrue(snap.landing_contact)

    def test_local_geometry_reports_overlap_and_edge(self):
        top = Surface(8, 200, 400, 150, 350, z_order=-1)
        geo = query_local_geometry([self.surface, top], 202, 220, 30, self.bounds)
        self.assertEqual(geo.support_id, 8)
        self.assertEqual(geo.overlap_depth, 2)
        self.assertLessEqual(geo.edge_distance, 2.1)


if __name__ == "__main__":
    unittest.main()
