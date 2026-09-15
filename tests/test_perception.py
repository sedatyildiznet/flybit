from __future__ import annotations

import unittest

from flybit.perception import DesktopSemanticScanner, PerceivedObject


class PerceptionTest(unittest.TestCase):
    def test_cursor_is_a_semantic_object(self):
        scanner = DesktopSemanticScanner()
        objects = scanner.scan((100.0, 200.0))
        cursor = next(obj for obj in objects if obj.kind == "cursor")
        self.assertEqual(cursor.label, "Mouse cursor")
        self.assertEqual(cursor.center, (100.0, 200.0))

    def test_nearest_orders_objects_by_body_distance(self):
        objects = [
            PerceivedObject("window", "far", 500, 500, 20, 20),
            PerceivedObject("button", "near", 10, 10, 10, 10),
        ]
        nearest = DesktopSemanticScanner.nearest(objects, 0, 0, limit=2)
        self.assertEqual(nearest[0].label, "near")


if __name__ == "__main__":
    unittest.main()
