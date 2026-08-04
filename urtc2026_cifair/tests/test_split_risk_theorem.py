from __future__ import annotations

import itertools
import unittest

from split_risk_theorem import (
    allocation_dp_extrema,
    balanced_allocation,
    capped_concentrated_allocation,
    concentrated_allocation,
    crossing_probability,
    exposed_expectation,
    theorem_certificate,
)


def allocations(groups: int, total: int, minimum: int = 2):
    if groups == 1:
        if total >= minimum:
            yield (total,)
        return
    for first in range(minimum, total + 1):
        for tail in allocations(groups - 1, total - first, first):
            yield (first,) + tail


class SplitRiskTheoremTests(unittest.TestCase):
    def test_public_cifar10_closed_form_witnesses(self):
        result = theorem_certificate(60_000, 10_000, 288, 608)
        crossing = result["results"]["expected_crossing_groups"]
        exposed = result["results"]["expected_exposed_test_files"]
        self.assertAlmostEqual(crossing["minimum"], 80.722, places=3)
        self.assertAlmostEqual(crossing["maximum"], 84.446, places=3)
        self.assertAlmostEqual(exposed["minimum"], 85.390, places=3)
        self.assertAlmostEqual(exposed["maximum"], 86.668, places=3)

    def test_closed_form_extrema_match_brute_force(self):
        for population, test, groups, total in ((20, 4, 3, 9), (24, 6, 4, 12), (30, 8, 4, 15)):
            feasible = list(allocations(groups, total))
            for function in (crossing_probability, exposed_expectation):
                scored = [(sum(function(population, test, size) for size in row), row) for row in feasible]
                self.assertEqual(min(scored)[1], concentrated_allocation(groups, total))
                self.assertEqual(max(scored)[1], balanced_allocation(groups, total))

    def test_dynamic_program_matches_exhaustive_small_fibers(self):
        checked = 0
        for population in range(8, 15):
            for test in range(1, (population - 1) // 2 + 1):
                for groups in range(1, min(4, population // 2) + 1):
                    for total in range(2 * groups, min(population, 2 * groups + 5) + 1):
                        feasible = list(allocations(groups, total))
                        for function in (crossing_probability, exposed_expectation):
                            values = [sum(function(population, test, size) for size in row) for row in feasible]
                            self.assertEqual(
                                allocation_dp_extrema(function, population, test, groups, total),
                                (min(values), max(values)),
                            )
                            checked += 1
        self.assertGreater(checked, 500)
    def test_capped_corollary_matches_dynamic_program_and_enumeration(self):
        checked = 0
        for population in range(8, 13):
            for test in range(1, (population - 1) // 2 + 1):
                for groups in range(1, min(3, population // 2) + 1):
                    for total in range(2 * groups, population + 1):
                        minimum_cap = (total + groups - 1) // groups
                        for cap in range(max(2, minimum_cap), min(6, total - 2 * (groups - 1)) + 1):
                            feasible = [row for row in allocations(groups, total) if max(row) <= cap]
                            if not feasible:
                                continue
                            witness = capped_concentrated_allocation(groups, total, cap)
                            self.assertLessEqual(max(witness), cap)
                            for function in (crossing_probability, exposed_expectation):
                                values = [sum(function(population, test, size) for size in row) for row in feasible]
                                witness_value = sum(function(population, test, size) for size in witness)
                                self.assertEqual(witness_value, min(values))
                                self.assertEqual(
                                    allocation_dp_extrema(function, population, test, groups, total, cap),
                                    (min(values), max(values)),
                                )
                                checked += 1
        self.assertGreater(checked, 100)
    def test_invalid_majority_test_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "test <="):
            theorem_certificate(20, 11, 3, 9)

    def test_group_size_may_exceed_test_size(self):
        result = theorem_certificate(20, 4, 2, 9)
        for name in ("expected_crossing_groups", "expected_exposed_test_files"):
            row = result["results"][name]
            self.assertEqual(row["minimum_allocation"], [2, 7])
            self.assertEqual(row["maximum_allocation"], [4, 5])
            self.assertTrue(row["discrete_concave_on_sizes_2_through"] >= 7)

    def test_short_domains_are_vacuously_certified(self):
        result_k2 = theorem_certificate(20, 4, 3, 6)
        result_k3 = theorem_certificate(20, 4, 3, 7)
        for result in (result_k2, result_k3):
            rows = result["results"]["expected_crossing_groups"]
            self.assertEqual(rows["nondecreasing_on_sizes_2_through"], max(result["results"]["expected_crossing_groups"]["minimum_allocation"]))
        self.assertIsNone(result_k2["results"]["expected_crossing_groups"]["minimum_first_difference"])
        self.assertIsNone(result_k2["results"]["expected_crossing_groups"]["largest_second_difference"])
        self.assertIsNotNone(result_k3["results"]["expected_crossing_groups"]["minimum_first_difference"])
        self.assertIsNone(result_k3["results"]["expected_crossing_groups"]["largest_second_difference"])

    def test_crossing_is_concave_for_majority_test_splits(self):
        for population in range(6, 31):
            for test in range((population + 1) // 2, population):
                values = [crossing_probability(population, test, size) for size in range(2, population + 1)]
                first = [right - left for left, right in zip(values, values[1:])]
                second = [values[i + 2] - 2 * values[i + 1] + values[i] for i in range(len(values) - 2)]
                self.assertTrue(all(value >= 0 for value in first))
                self.assertTrue(all(value <= 0 for value in second))
    def test_pointwise_exposure_is_monotone_under_added_relations(self):
        # For every assignment of a three-member group, adding a fourth member
        # cannot reduce the number of test members that have a train counterpart.
        for bits in itertools.product((0, 1), repeat=4):  # 1=test, 0=train
            before = sum(bits[:3]) if 0 < sum(bits[:3]) < 3 else 0
            after = sum(bits) if 0 < sum(bits) < 4 else 0
            self.assertGreaterEqual(after, before)


if __name__ == "__main__":
    unittest.main()
