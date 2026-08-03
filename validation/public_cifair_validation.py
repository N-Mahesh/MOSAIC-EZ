"""Public validation of aggregate-only leakage bounds on ciFAIR metadata."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from split_risk_theorem import (
    allocation_dp_extrema,
    all_selected_probability,
    crossing_probability,
    exposed_expectation,
)


POPULATION = 60_000
TEST_FILES = 10_000
SOURCE_COMMIT = "4c2764277f5fda8fec6784a78c1818eab13236c5"
EXPECTED_SHA256 = {
    "duplicates_cifar10.csv": "4cb7e99a7dfff346082c9d8fa4c2989a196e4a37d1e58f75936046696f1ba6a4",
    "duplicates_cifar10_test.csv": "ef7b33e2f32d056cdb342046b437ee819541a07cb3bb0520a3ada9b9c035f532",
    "duplicates_cifar100.csv": "3891498a862f2d73df00cd5dc2ee9ae2b6dccf5238691d3c29d9abb12e1c63cb",
    "duplicates_cifar100_test.csv": "78c9d8d17eda11b468a137119ed51f09332393d65a6329035e2250888edb2760",
}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(self, item: tuple[str, int]) -> tuple[str, int]:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: tuple[str, int], right: tuple[str, int]) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_rows(path: Path) -> list[dict[str, str]]:
    expected = EXPECTED_SHA256[path.name]
    if sha256(path) != expected:
        raise ValueError(f"checksum mismatch: {path.name}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_fields = {"TestID", "TrainID", "Distance", "Judgment"}
    if not rows or set(rows[0]) != expected_fields:
        raise ValueError(f"unexpected CSV schema: {path.name}")
    if any(row["Judgment"] not in {"0", "1", "2"} for row in rows):
        raise ValueError(f"unexpected judgment value: {path.name}")
    return rows


def weighted_sum(
    histogram: Counter[int],
    function,
    population: int = POPULATION,
    test_files: int = TEST_FILES,
) -> Fraction:
    return sum(
        (count * function(population, test_files, size) for size, count in histogram.items()),
        Fraction(0),
    )


@lru_cache(maxsize=None)
def joint_crossing_probability(
    population: int,
    test_files: int,
    left_size: int,
    right_size: int,
) -> Fraction:
    """Probability that two disjoint groups both cross a fixed-size split."""
    if left_size < 1 or right_size < 1 or left_size + right_size > population:
        raise ValueError("group sizes must be positive, disjoint, and within the population")

    def choose_or_zero(n: int, r: int) -> int:
        return 0 if r < 0 or r > n else math.comb(n, r)

    denominator = math.comb(population, test_files)
    left_non_crossing = all_selected_probability(population, test_files, left_size) + all_selected_probability(
        population, population - test_files, left_size
    )
    right_non_crossing = all_selected_probability(population, test_files, right_size) + all_selected_probability(
        population, population - test_files, right_size
    )
    residual = population - left_size - right_size
    both_non_crossing = Fraction(
        choose_or_zero(residual, test_files)
        + choose_or_zero(residual, test_files - right_size)
        + choose_or_zero(residual, test_files - left_size)
        + choose_or_zero(residual, test_files - left_size - right_size),
        denominator,
    )
    return 1 - left_non_crossing - right_non_crossing + both_non_crossing


def crossing_variance(
    histogram: Counter[int],
    population: int = POPULATION,
    test_files: int = TEST_FILES,
) -> Fraction:
    """Exact variance of the crossing-group count for known group sizes."""
    probabilities = {
        size: crossing_probability(population, test_files, size) for size in histogram
    }
    variance = sum(
        (
            count * probabilities[size] * (1 - probabilities[size])
            for size, count in histogram.items()
        ),
        Fraction(0),
    )
    sizes = sorted(histogram)
    for index, left_size in enumerate(sizes):
        for right_size in sizes[index:]:
            pair_count = (
                histogram[left_size] * (histogram[left_size] - 1) // 2
                if left_size == right_size
                else histogram[left_size] * histogram[right_size]
            )
            covariance = joint_crossing_probability(
                population, test_files, left_size, right_size
            ) - probabilities[left_size] * probabilities[right_size]
            variance += 2 * pair_count * covariance
    return variance

def dataset_result(
    meta: Path,
    dataset: int,
    judgments: frozenset[str] = frozenset({"0", "1", "2"}),
    audit_edge_deletions: bool = False,
) -> dict[str, object]:
    cross_name = f"duplicates_cifar{dataset}.csv"
    test_name = f"duplicates_cifar{dataset}_test.csv"
    cross_rows = [row for row in checked_rows(meta / cross_name) if row["Judgment"] in judgments]
    test_rows = [row for row in checked_rows(meta / test_name) if row["Judgment"] in judgments]
    if not judgments or not judgments <= {"0", "1", "2"}:
        raise ValueError("judgments must be a nonempty subset of {0,1,2}")

    edges = [
        (("test", int(row["TestID"])), ("train", int(row["TrainID"])))
        for row in cross_rows
    ] + [
        (("test", int(row["TestID"])), ("test", int(row["TrainID"])))
        for row in test_rows
    ]

    def build_components(selected_edges):
        groups = UnionFind()
        for left, right in selected_edges:
            groups.union(left, right)
        built: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
        for node in list(groups.parent):
            built[groups.find(node)].add(node)
        return built

    components = build_components(edges)
    histogram = Counter(len(component) for component in components.values())
    if histogram and min(histogram) < 2:
        raise AssertionError("all annotated components must be non-singletons")

    group_count = len(components)
    grouped_files = sum(size * count for size, count in histogram.items())
    known_crossing = weighted_sum(histogram, crossing_probability)
    known_crossing_variance = crossing_variance(histogram)
    known_exposed = weighted_sum(histogram, exposed_expectation)
    if group_count:
        maximum_feasible = grouped_files - 2 * (group_count - 1)
        concentrated = Counter({2: group_count - 1})
        concentrated[maximum_feasible] += 1
        quotient, remainder = divmod(grouped_files, group_count)
        balanced = Counter({quotient: group_count - remainder, quotient + 1: remainder})
        if remainder == 0:
            balanced.pop(quotient + 1, None)
        crossing_min = weighted_sum(concentrated, crossing_probability)
        crossing_max = weighted_sum(balanced, crossing_probability)
        exposed_min = weighted_sum(concentrated, exposed_expectation)
        exposed_max = weighted_sum(balanced, exposed_expectation)
        dp_crossing = allocation_dp_extrema(
            crossing_probability, POPULATION, TEST_FILES, group_count, grouped_files
        )
        dp_exposed = allocation_dp_extrema(
            exposed_expectation, POPULATION, TEST_FILES, group_count, grouped_files
        )
        if dp_crossing != (crossing_min, crossing_max):
            raise AssertionError("crossing closed forms disagree with the allocation dynamic program")
        if dp_exposed != (exposed_min, exposed_max):
            raise AssertionError("exposure closed forms disagree with the allocation dynamic program")
    else:
        crossing_min = crossing_max = Fraction(0)
        exposed_min = exposed_max = Fraction(0)
    if not crossing_min <= known_crossing <= crossing_max:
        raise AssertionError("known crossing expectation escaped aggregate bounds")
    if not exposed_min <= known_exposed <= exposed_max:
        raise AssertionError("known exposure expectation escaped aggregate bounds")

    official_crossing = 0
    official_exposed = 0
    for component in components.values():
        has_train = any(split == "train" for split, _ in component)
        test_count = sum(split == "test" for split, _ in component)
        official_crossing += int(has_train and test_count > 0)
        if has_train:
            official_exposed += test_count


    def rounded(value: Fraction) -> float:
        return round(float(value), 12)
    edge_deletion_sensitivity = None
    if audit_edge_deletions and edges:
        crossing_values = []
        exposure_values = []
        group_values = []
        file_values = []
        for omitted in range(len(edges)):
            reduced_components = build_components(edges[:omitted] + edges[omitted + 1 :])
            reduced_histogram = Counter(len(component) for component in reduced_components.values())
            group_values.append(len(reduced_components))
            file_values.append(sum(size * count for size, count in reduced_histogram.items()))
            crossing_values.append(weighted_sum(reduced_histogram, crossing_probability))
            exposure_values.append(weighted_sum(reduced_histogram, exposed_expectation))
        edge_deletion_sensitivity = {
            "omissions_checked": len(edges),
            "group_count_range": [min(group_values), max(group_values)],
            "grouped_files_range": [min(file_values), max(file_values)],
            "known_crossing_range": [rounded(min(crossing_values)), rounded(max(crossing_values))],
            "known_exposed_range": [rounded(min(exposure_values)), rounded(max(exposure_values))],
            "maximum_absolute_crossing_change": rounded(
                max(abs(value - known_crossing) for value in crossing_values)
            ),
            "maximum_absolute_exposed_change": rounded(
                max(abs(value - known_exposed) for value in exposure_values)
            ),
        }
    return {

        "dataset": f"CIFAR-{dataset}",
        "population": POPULATION,
        "official_test_files": TEST_FILES,
        "annotated_edge_rows": len(cross_rows) + len(test_rows),
        "annotated_group_count": group_count,
        "annotated_grouped_files": grouped_files,
        "known_group_size_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "known_size_expectation": {
            "crossing_groups": rounded(known_crossing),
            "crossing_group_variance": rounded(known_crossing_variance),
            "crossing_group_standard_deviation": round(math.sqrt(float(known_crossing_variance)), 12),
            "exposed_test_files": rounded(known_exposed),
        },
        "aggregate_only_bounds": {
            "crossing_groups": [rounded(crossing_min), rounded(crossing_max)],
            "exposed_test_files": [rounded(exposed_min), rounded(exposed_max)],
        },
        "official_split_observation": {
            "crossing_groups": official_crossing,
            "exposed_test_files": official_exposed,
        },
        "single_edge_deletion_sensitivity": edge_deletion_sensitivity,
    }


def run(meta: Path) -> dict[str, object]:
    policies = (
        ("{0}", frozenset({"0"})),
        ("{0,1}", frozenset({"0", "1"})),
        ("{0,1,2}", frozenset({"0", "1", "2"})),
    )
    sensitivity = []
    for label, judgments in policies:
        rows = [dataset_result(meta, dataset, judgments) for dataset in (10, 100)]
        sensitivity.append({"judgments": label, "datasets": rows})
    full_datasets = [
        dataset_result(meta, dataset, frozenset({"0", "1", "2"}), audit_edge_deletions=True)
        for dataset in (10, 100)
    ]
    return {
        "schema_version": 2,
        "source_repository": "https://github.com/cvjena/cifair",
        "source_commit": SOURCE_COMMIT,
        "edge_policy": "union all published judgment categories 0, 1, and 2; train/test node namespaces remain distinct",
        "file_sha256": EXPECTED_SHA256,
        "datasets": full_datasets,
        "policy_sensitivity": sensitivity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    result = run(args.meta)
    if args.verify:
        if result != json.loads(args.verify.read_text(encoding="utf-8")):
            print("ciFAIR validation mismatch.")
            return 1
        print("ciFAIR validation matches the frozen public result.")
        return 0
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"Wrote ciFAIR validation: {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())