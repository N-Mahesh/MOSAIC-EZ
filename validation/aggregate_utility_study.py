"""Deterministic utility study for aggregate-only split-risk certificates."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

from split_risk_theorem import (
    balanced_allocation,
    capped_concentrated_allocation,
    concentrated_allocation,
    crossing_probability,
    exposed_expectation,
)

POPULATION = 60_000
GROUPS = 100
TEST_FRACTIONS = (Fraction(1, 20), Fraction(1, 6), Fraction(1, 3))
AVERAGE_SIZES = (Fraction(2), Fraction(11, 5), Fraction(3), Fraction(5), Fraction(10), Fraction(20))
CAP_CASES = ((Fraction(11, 5), 4), (Fraction(3), 4), (Fraction(5), 8), (Fraction(10), 16))
DECISION_THRESHOLD = Fraction(1, 2)


def _sum(function, allocation: tuple[int, ...], population: int, test: int) -> Fraction:
    return sum((function(population, test, size) for size in allocation), start=Fraction(0))


def _decision(lower: Fraction, upper: Fraction, scale: int) -> str:
    if lower >= DECISION_THRESHOLD * scale:
        return "certify-high"
    if upper < DECISION_THRESHOLD * scale:
        return "certify-low"
    return "inconclusive"


def evaluate_case(test_fraction: Fraction, average_size: Fraction, cap: int | None = None) -> dict[str, object]:
    test = round(POPULATION * test_fraction)
    total = round(GROUPS * average_size)
    lower_allocation = (
        concentrated_allocation(GROUPS, total)
        if cap is None
        else capped_concentrated_allocation(GROUPS, total, cap)
    )
    upper_allocation = balanced_allocation(GROUPS, total)
    crossing_lower = _sum(crossing_probability, lower_allocation, POPULATION, test)
    crossing_upper = _sum(crossing_probability, upper_allocation, POPULATION, test)
    exposure_lower = _sum(exposed_expectation, lower_allocation, POPULATION, test)
    exposure_upper = _sum(exposed_expectation, upper_allocation, POPULATION, test)
    crossing_width = Fraction(0) if not crossing_upper else (crossing_upper - crossing_lower) / crossing_upper
    exposure_width = Fraction(0) if not exposure_upper else (exposure_upper - exposure_lower) / exposure_upper
    return {
        "test_fraction": f"{test_fraction.numerator}/{test_fraction.denominator}",
        "test_files": test,
        "average_group_size": float(average_size),
        "grouped_files": total,
        "maximum_group_size": cap,
        "crossing_fraction_bounds": [float(crossing_lower / GROUPS), float(crossing_upper / GROUPS)],
        "exposure_fraction_bounds": [float(exposure_lower / test), float(exposure_upper / test)],
        "crossing_relative_width": float(crossing_width),
        "exposure_relative_width": float(exposure_width),
        "crossing_threshold": float(DECISION_THRESHOLD),
        "crossing_decision": _decision(crossing_lower, crossing_upper, GROUPS),
    }


def build_study() -> dict[str, object]:
    cases = [evaluate_case(test_fraction, average) for test_fraction in TEST_FRACTIONS for average in AVERAGE_SIZES]
    cases.extend(evaluate_case(Fraction(1, 6), average, cap) for average, cap in CAP_CASES)
    return {
        "schema_version": 1,
        "population": POPULATION,
        "group_count": GROUPS,
        "decision_rule": "certify-high iff lower>=0.5G; certify-low iff upper<0.5G; otherwise inconclusive",
        "cases": cases,
    }


def _case(data: dict[str, object], average: float, cap: int | None) -> dict[str, object]:
    return next(
        row for row in data["cases"]
        if row["test_fraction"] == "1/6"
        and row["average_group_size"] == average
        and row["maximum_group_size"] == cap
    )


def render_tex(data: dict[str, object]) -> str:
    rows = {
        "Three": _case(data, 3.0, None),
        "Five": _case(data, 5.0, None),
        "FiveCapEight": _case(data, 5.0, 8),
        "TenCapSixteen": _case(data, 10.0, 16),
    }
    values: dict[str, str] = {}
    labels = {"certify-high": "high", "certify-low": "low", "inconclusive": "inconclusive"}
    for name, row in rows.items():
        low, high = row["crossing_fraction_bounds"]
        values[f"Utility{name}CrossLowPct"] = f"{100 * low:.1f}"
        values[f"Utility{name}CrossHighPct"] = f"{100 * high:.1f}"
        values[f"Utility{name}CrossWidthPct"] = f"{100 * row['crossing_relative_width']:.1f}"
        values[f"Utility{name}Decision"] = labels[row["crossing_decision"]]
    uncapped = [row for row in data["cases"] if row["maximum_group_size"] is None]
    values["UtilityGridMaxCrossWidthPct"] = f"{100 * max(row['crossing_relative_width'] for row in uncapped):.1f}"
    values["UtilityGridMaxExposureWidthPct"] = f"{100 * max(row['exposure_relative_width'] for row in uncapped):.1f}"
    return "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in values.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    data = build_study()
    rendered_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
    rendered_tex = render_tex(data)
    if args.verify:
        if args.json.read_text(encoding="utf-8") != rendered_json:
            print("aggregate utility JSON mismatch")
            return 1
        if args.tex.read_text(encoding="utf-8") != rendered_tex:
            print("aggregate utility TeX mismatch")
            return 1
        print("Aggregate utility study matches frozen JSON and TeX macros.")
        return 0
    args.json.write_text(rendered_json, encoding="utf-8", newline="\n")
    args.tex.write_text(rendered_tex, encoding="utf-8", newline="\n")
    print(f"Wrote utility study: {args.json} and {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())