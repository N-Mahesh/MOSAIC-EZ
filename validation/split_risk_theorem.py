"""Exact certificate for the aggregate-only split-risk extrema theorem.

The theorem applies to a uniformly sampled fixed-size test set when only the
population M, number G of disjoint duplicate groups, and their total size S are
known. It verifies discrete concavity of the per-group estimands and the
closed-form concentrated/balanced extremizers without record access, including
group sizes larger than the test set.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from functools import lru_cache
import json
import math
from pathlib import Path


TOTAL_FILES = 2258
GROUP_COUNT = 103
GROUPED_FILES = 280
TEST_FILES = 452


@lru_cache(maxsize=None)
def all_selected_probability(population: int, selected: int, size: int) -> Fraction:
    if size > selected:
        return Fraction(0)
    return Fraction(math.comb(population - size, selected - size), math.comb(population, selected))


@lru_cache(maxsize=None)
def crossing_probability(population: int, test: int, size: int) -> Fraction:
    return (
        Fraction(1)
        - all_selected_probability(population, test, size)
        - all_selected_probability(population, population - test, size)
    )


@lru_cache(maxsize=None)
def exposed_expectation(population: int, test: int, size: int) -> Fraction:
    return Fraction(size * test, population) - size * all_selected_probability(
        population, test, size
    )


def concentrated_allocation(groups: int, total: int) -> tuple[int, ...]:
    if groups < 1 or total < 2 * groups:
        raise ValueError("infeasible group aggregates")
    return tuple([2] * (groups - 1) + [total - 2 * (groups - 1)])


def balanced_allocation(groups: int, total: int) -> tuple[int, ...]:
    if groups < 1 or total < 2 * groups:
        raise ValueError("infeasible group aggregates")
    quotient, remainder = divmod(total, groups)
    return tuple([quotient] * (groups - remainder) + [quotient + 1] * remainder)


def theorem_certificate(population: int, test: int, groups: int, total: int) -> dict[str, object]:
    if population < 1 or groups < 1:
        raise ValueError("population and group count must be positive")
    if total < 2 * groups or total > population:
        raise ValueError("grouped total must satisfy 2G <= S <= M")
    if not 1 <= test <= (population - 1) // 2:
        raise ValueError("the exposed-file concavity theorem requires test <= (M-1)/2")
    concentrated = concentrated_allocation(groups, total)
    balanced = balanced_allocation(groups, total)
    maximum_size = max(concentrated)
    rows = {}
    for name, function in (
        ("expected_crossing_groups", crossing_probability),
        ("expected_exposed_test_files", exposed_expectation),
    ):
        values = [function(population, test, size) for size in range(2, maximum_size + 1)]
        first = [right - left for left, right in zip(values, values[1:])]
        second = [values[index + 2] - 2 * values[index + 1] + values[index] for index in range(len(values) - 2)]
        if not all(value >= 0 for value in first):
            raise AssertionError(f"{name} is not nondecreasing")
        if name == "expected_exposed_test_files" and not all(value > 0 for value in first):
            raise AssertionError(f"{name} is not strictly increasing")
        if not all(value <= 0 for value in second):
            raise AssertionError(f"{name} is not discrete-concave")
        rows[name] = {
            "nondecreasing_on_sizes_2_through": maximum_size,
            "strictly_increasing": all(value > 0 for value in first),
            "discrete_concave_on_sizes_2_through": maximum_size,
            "strictly_discrete_concave": all(value < 0 for value in second),
            "minimum_first_difference": float(min(first)) if first else None,
            "largest_second_difference": float(max(second)) if second else None,
            "minimum_allocation": list(concentrated),
            "maximum_allocation": list(balanced),
            "minimum": float(sum(function(population, test, size) for size in concentrated)),
            "maximum": float(sum(function(population, test, size) for size in balanced)),
        }
    return {
        "schema_version": 1,
        "population": population,
        "test_files": test,
        "group_count": groups,
        "grouped_files": total,
        "theorem_scope": "uniform fixed-size test sample; disjoint groups; k_i>=2; T<=(M-1)/2; no K<=T restriction",
        "crossing_only_scope": "the crossing functional is discrete-concave for every 1<=T<M",
        "proof_strategy": (
            "discrete concavity plus majorization: the concentrated "
            "allocation minimizes and the balanced allocation maximizes each sum"
        ),
        "results": rows,
    }


def run_analysis() -> dict[str, object]:
    return theorem_certificate(TOTAL_FILES, TEST_FILES, GROUP_COUNT, GROUPED_FILES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", type=Path)
    mode.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_analysis()
    if args.verify:
        expected = json.loads(args.verify.read_text(encoding="utf-8"))
        if result != expected:
            print("Split-risk theorem certificate mismatch.")
            return 1
        print("Split-risk theorem certificate matches the frozen result.")
        return 0
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"Wrote split-risk theorem certificate: {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
