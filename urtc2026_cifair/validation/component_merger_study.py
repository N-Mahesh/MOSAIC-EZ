"""Size-only component-merger stress paths for split-conditioned graphs."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from split_risk_theorem import crossing_probability, exposed_expectation

BUDGETS = (1, 10, 25)


def relation_changes(population: int, test: int, left: int, right: int) -> tuple[Fraction, Fraction]:
    if left < 1 or right < 1 or left + right > population:
        raise ValueError("relation endpoint sizes must be positive and fit in the population")
    crossing = (
        crossing_probability(population, test, left + right)
        - crossing_probability(population, test, left)
        - crossing_probability(population, test, right)
    )
    exposure = (
        exposed_expectation(population, test, left + right)
        - exposed_expectation(population, test, left)
        - exposed_expectation(population, test, right)
    )
    return crossing, exposure

def merger_changes(population: int, test: int, left: int, right: int) -> tuple[Fraction, Fraction]:
    if left < 2 or right < 2:
        raise ValueError("merger sizes must both be at least two")
    return relation_changes(population, test, left, right)


def _weighted(histogram: Counter[int], function, population: int, test: int) -> Fraction:
    return sum((count * function(population, test, size) for size, count in histogram.items()), start=Fraction(0))


def _pairs(histogram: Counter[int]):
    sizes = sorted(histogram)
    for index, left in enumerate(sizes):
        for right in sizes[index:]:
            if left != right or histogram[left] >= 2:
                yield left, right


def greedy_path(histogram: Counter[int], population: int, test: int, objective: str) -> list[dict[str, object]]:
    if objective not in {"minimize-crossing", "maximize-exposure"}:
        raise ValueError("unknown merger objective")
    current = histogram.copy()
    crossing = _weighted(current, crossing_probability, population, test)
    exposure = _weighted(current, exposed_expectation, population, test)
    rows = []
    maximum_budget = min(max(BUDGETS), sum(current.values()) - 1)
    for step in range(1, maximum_budget + 1):
        candidates = []
        for left, right in _pairs(current):
            delta_crossing, delta_exposure = merger_changes(population, test, left, right)
            if delta_crossing > 0 or delta_exposure < 0:
                raise AssertionError("component merger violated expected sign certificate")
            if objective == "minimize-crossing":
                key = (delta_crossing, -delta_exposure, left, right)
            else:
                key = (-delta_exposure, delta_crossing, left, right)
            candidates.append((key, left, right, delta_crossing, delta_exposure))
        _, left, right, delta_crossing, delta_exposure = min(candidates)
        current[left] -= 1
        current[right] -= 1
        if not current[left]:
            del current[left]
        if right in current and not current[right]:
            del current[right]
        current[left + right] += 1
        crossing += delta_crossing
        exposure += delta_exposure
        rows.append({
            "merger_budget": step,
            "selected_component_sizes": [left, right],
            "delta_crossing_groups": float(delta_crossing),
            "delta_exposed_test_files": float(delta_exposure),
            "remaining_components": sum(current.values()),
            "crossing_groups": float(crossing),
            "exposed_test_files": float(exposure),
        })
    return rows


def build_study(cifair: dict[str, object]) -> dict[str, object]:
    datasets = []
    for source in cifair["datasets"]:
        histogram = Counter({int(size): count for size, count in source["known_group_size_histogram"].items()})
        population = source["population"]
        test = source["official_test_files"]
        baseline_crossing = _weighted(histogram, crossing_probability, population, test)
        baseline_exposure = _weighted(histogram, exposed_expectation, population, test)
        datasets.append({
            "dataset": source["dataset"],
            "baseline": {
                "components": sum(histogram.values()),
                "crossing_groups": float(baseline_crossing),
                "exposed_test_files": float(baseline_exposure),
            },
            "minimize_crossing_path": greedy_path(histogram, population, test, "minimize-crossing"),
            "maximize_exposure_path": greedy_path(histogram, population, test, "maximize-exposure"),
        })
    return {
        "schema_version": 1,
        "method": "deterministic size-only greedy merger stress with every selected size pair recorded; trajectories are not sharp missing-edge bounds",
        "sign_certificate": "merging two non-singleton components weakly decreases expected crossing count and weakly increases exposure",
        "boundary_certificate": "attaching a singleton or linking two singletons weakly increases both expected crossing count and exposure",
        "datasets": datasets,
    }


def render_tex(data: dict[str, object]) -> str:
    values = {}
    for dataset in data["datasets"]:
        name = "Ten" if dataset["dataset"] == "CIFAR-10" else "Hundred"
        baseline = dataset["baseline"]
        crossing_rows = {row["merger_budget"]: row for row in dataset["minimize_crossing_path"]}
        exposure_rows = {row["merger_budget"]: row for row in dataset["maximize_exposure_path"]}
        budget_names = {1: "One", 10: "Ten", 25: "TwentyFive"}
        for budget in BUDGETS:
            values[f"Cifar{name}Merge{budget_names[budget]}CrossDrop"] = f"{baseline['crossing_groups'] - crossing_rows[budget]['crossing_groups']:.3f}"
            values[f"Cifar{name}Merge{budget_names[budget]}ExposureRise"] = f"{exposure_rows[budget]['exposed_test_files'] - baseline['exposed_test_files']:.3f}"
    return "".join(f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in values.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifair-result", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = build_study(json.loads(args.cifair_result.read_text(encoding="utf-8")))
    rendered_json = json.dumps(result, indent=2, sort_keys=True) + "\n"
    rendered_tex = render_tex(result)
    if args.verify:
        if args.json.read_text(encoding="utf-8") != rendered_json or args.tex.read_text(encoding="utf-8") != rendered_tex:
            print("component merger study mismatch")
            return 1
        print("Component merger study matches frozen JSON and TeX macros.")
        return 0
    args.json.write_text(rendered_json, encoding="utf-8", newline="\n")
    args.tex.write_text(rendered_tex, encoding="utf-8", newline="\n")
    print(f"Wrote component merger study: {args.json} and {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())