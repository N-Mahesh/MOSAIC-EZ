from __future__ import annotations

from collections import Counter
from fractions import Fraction
import itertools
import unittest

from public_cifair_validation import (
    UnionFind,
    crossing_variance,
    joint_crossing_probability,
    weighted_sum,
)
from split_risk_theorem import crossing_probability


class PublicCifairValidationTests(unittest.TestCase):
    def test_pair_probability_and_variance_match_brute_force(self):
        population, test_files = 6, 2
        left, right = {0, 1}, {2, 3}
        crossing_counts = []
        both_cross = 0
        for selected in itertools.combinations(range(population), test_files):
            chosen = set(selected)
            left_cross = bool(chosen & left) and bool(left - chosen)
            right_cross = bool(chosen & right) and bool(right - chosen)
            both_cross += int(left_cross and right_cross)
            crossing_counts.append(int(left_cross) + int(right_cross))
        denominator = len(crossing_counts)
        mean = Fraction(sum(crossing_counts), denominator)
        brute_variance = sum((Fraction(value) - mean) ** 2 for value in crossing_counts) / denominator
        self.assertEqual(
            joint_crossing_probability(population, test_files, 2, 2),
            Fraction(both_cross, denominator),
        )
        self.assertEqual(
            crossing_variance(Counter({2: 2}), population, test_files),
            brute_variance,
        )
    def test_union_find_closes_transitive_components(self):
        groups = UnionFind()
        groups.union(("test", 1), ("train", 2))
        groups.union(("test", 3), ("train", 2))
        self.assertEqual(groups.find(("test", 1)), groups.find(("test", 3)))
        self.assertNotEqual(groups.find(("test", 1)), groups.find(("test", 4)))

    def test_all_size_two_histogram_is_not_collapsed(self):
        histogram = Counter({2: 41})
        value = weighted_sum(histogram, crossing_probability)
        self.assertEqual(value, 41 * crossing_probability(60_000, 10_000, 2))
    def test_weighted_sum_matches_expansion(self):
        histogram = Counter({2: 3, 4: 1})
        weighted = weighted_sum(histogram, crossing_probability, population=30, test_files=6)
        expanded = sum(
            (crossing_probability(30, 6, size) for size in (2, 2, 2, 4)),
            Fraction(0),
        )
        self.assertEqual(weighted, expanded)


if __name__ == "__main__":
    unittest.main()