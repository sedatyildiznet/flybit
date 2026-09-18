from __future__ import annotations

import unittest

from flybit.phenotype import phenotype_from_identity, seed_from_identity


class PhenotypeTest(unittest.TestCase):
    def test_same_identity_is_stable(self):
        a = phenotype_from_identity("2026-01-01T00:00:00+00:00")
        b = phenotype_from_identity("2026-01-01T00:00:00+00:00")
        self.assertEqual(a, b)
        self.assertEqual(
            seed_from_identity("abc"),
            seed_from_identity("abc"),
        )

    def test_different_identity_changes_fingerprint(self):
        a = phenotype_from_identity("organism-a")
        b = phenotype_from_identity("organism-b")
        self.assertNotEqual(a, b)

    def test_variation_stays_species_scale(self):
        p = phenotype_from_identity("bounded")
        self.assertGreaterEqual(p.stride_scale, 0.92)
        self.assertLessEqual(p.stride_scale, 1.08)
        self.assertGreaterEqual(p.body_scale, 0.94)
        self.assertLessEqual(p.body_scale, 1.04)


if __name__ == "__main__":
    unittest.main()
