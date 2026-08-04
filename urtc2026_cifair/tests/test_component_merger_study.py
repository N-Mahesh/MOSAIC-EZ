from __future__ import annotations

from collections import Counter
import unittest

from component_merger_study import greedy_path, merger_changes, relation_changes


class ComponentMergerStudyTests(unittest.TestCase):
    def test_merger_signs_on_exhaustive_small_domain(self) -> None:
        for population in range(4, 31):
            for test in range(1, population):
                for left in range(2, population):
                    for right in range(2, population - left + 1):
                        crossing, exposure = merger_changes(population, test, left, right)
                        self.assertLessEqual(crossing, 0)
                        self.assertGreaterEqual(exposure, 0)

    def test_singleton_boundary_has_opposite_crossing_direction(self) -> None:
        for population in range(2, 31):
            for test in range(1, population):
                pair_crossing, pair_exposure = relation_changes(population, test, 1, 1)
                self.assertGreater(pair_crossing, 0)
                self.assertGreater(pair_exposure, 0)
                for component in range(2, population):
                    if component + 1 <= population:
                        crossing, exposure = relation_changes(population, test, component, 1)
                        self.assertGreaterEqual(crossing, 0)
                        self.assertGreaterEqual(exposure, 0)

    def test_greedy_paths_are_deterministic_and_directional(self) -> None:
        histogram = Counter({2: 30, 3: 5, 5: 2})
        for objective in ("minimize-crossing", "maximize-exposure"):
            first = greedy_path(histogram, 1000, 100, objective)
            self.assertEqual(first, greedy_path(histogram, 1000, 100, objective))
            self.assertEqual(len(first), 25)
            self.assertTrue(all("selected_component_sizes" in row for row in first))
            crossing = [row["crossing_groups"] for row in first]
            exposure = [row["exposed_test_files"] for row in first]
            self.assertEqual(crossing, sorted(crossing, reverse=True))
            self.assertEqual(exposure, sorted(exposure))


if __name__ == "__main__":
    unittest.main()