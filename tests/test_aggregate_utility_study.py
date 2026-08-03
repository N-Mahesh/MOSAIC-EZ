from __future__ import annotations

from fractions import Fraction
import unittest

from aggregate_utility_study import build_study, evaluate_case


class AggregateUtilityStudyTests(unittest.TestCase):
    def test_grid_is_deterministic_and_ordered(self) -> None:
        first, second = build_study(), build_study()
        self.assertEqual(first, second)
        self.assertEqual(len(first["cases"]), 22)
        for row in first["cases"]:
            self.assertLessEqual(row["crossing_fraction_bounds"][0], row["crossing_fraction_bounds"][1])
            self.assertLessEqual(row["exposure_fraction_bounds"][0], row["exposure_fraction_bounds"][1])

    def test_cap_can_resolve_a_threshold_decision(self) -> None:
        uncapped = evaluate_case(Fraction(1, 6), Fraction(5))
        capped = evaluate_case(Fraction(1, 6), Fraction(5), 8)
        self.assertEqual(uncapped["crossing_decision"], "inconclusive")
        self.assertEqual(capped["crossing_decision"], "certify-high")
        self.assertLess(uncapped["crossing_fraction_bounds"][0], 0.5)
        self.assertGreaterEqual(capped["crossing_fraction_bounds"][0], 0.5)

    def test_low_risk_certificate_is_two_sided(self) -> None:
        row = evaluate_case(Fraction(1, 6), Fraction(3))
        self.assertEqual(row["crossing_decision"], "certify-low")
        self.assertLess(row["crossing_fraction_bounds"][1], 0.5)


if __name__ == "__main__":
    unittest.main()