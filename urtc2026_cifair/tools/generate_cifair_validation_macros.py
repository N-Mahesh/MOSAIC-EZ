"""Generate or verify LaTeX macros from the frozen public ciFAIR validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: float) -> str:
    return f"{value:.3f}"


def render(data: dict[str, object]) -> str:
    rows = {row["dataset"]: row for row in data["datasets"]}
    ten, hundred = rows["CIFAR-10"], rows["CIFAR-100"]
    values = {
        "CifarTenGroups": str(ten["annotated_group_count"]),
        "CifarTenFiles": str(ten["annotated_grouped_files"]),
        "CifarTenEdges": str(ten["annotated_edge_rows"]),
        "CifarTenKnownCross": fmt(ten["known_size_expectation"]["crossing_groups"]),
        "CifarTenCrossSD": fmt(ten["known_size_expectation"]["crossing_group_standard_deviation"]),
        "CifarTenEdgeDeletionMaxCross": fmt(ten["single_edge_deletion_sensitivity"]["maximum_absolute_crossing_change"]),
        "CifarTenEdgeDeletionMaxExposed": fmt(ten["single_edge_deletion_sensitivity"]["maximum_absolute_exposed_change"]),
        "CifarTenCrossMin": fmt(ten["aggregate_only_bounds"]["crossing_groups"][0]),
        "CifarTenCrossMax": fmt(ten["aggregate_only_bounds"]["crossing_groups"][1]),
        "CifarTenCap": str(ten["cap_aware_bounds"]["maximum_group_size"]),
        "CifarTenCapCrossMin": fmt(ten["cap_aware_bounds"]["crossing_groups"][0]),
        "CifarTenCapExposedMin": fmt(ten["cap_aware_bounds"]["exposed_test_files"][0]),
        "CifarTenKnownExposed": fmt(ten["known_size_expectation"]["exposed_test_files"]),
        "CifarTenExposedMin": fmt(ten["aggregate_only_bounds"]["exposed_test_files"][0]),
        "CifarTenExposedMax": fmt(ten["aggregate_only_bounds"]["exposed_test_files"][1]),
        "CifarTenObservedCross": str(ten["official_split_observation"]["crossing_groups"]),
        "CifarTenObservedExposed": str(ten["official_split_observation"]["exposed_test_files"]),
        "CifarHundredGroups": str(hundred["annotated_group_count"]),
        "CifarHundredFiles": f"{hundred['annotated_grouped_files']:,}",
        "CifarHundredEdges": str(hundred["annotated_edge_rows"]),
        "CifarHundredKnownCross": fmt(hundred["known_size_expectation"]["crossing_groups"]),
        "CifarHundredCrossSD": fmt(hundred["known_size_expectation"]["crossing_group_standard_deviation"]),
        "CifarHundredEdgeDeletionMaxCross": fmt(hundred["single_edge_deletion_sensitivity"]["maximum_absolute_crossing_change"]),
        "CifarHundredEdgeDeletionMaxExposed": fmt(hundred["single_edge_deletion_sensitivity"]["maximum_absolute_exposed_change"]),
        "CifarHundredCrossMin": fmt(hundred["aggregate_only_bounds"]["crossing_groups"][0]),
        "CifarHundredCrossMax": fmt(hundred["aggregate_only_bounds"]["crossing_groups"][1]),
        "CifarHundredCap": str(hundred["cap_aware_bounds"]["maximum_group_size"]),
        "CifarHundredCapCrossMin": fmt(hundred["cap_aware_bounds"]["crossing_groups"][0]),
        "CifarHundredCapExposedMin": fmt(hundred["cap_aware_bounds"]["exposed_test_files"][0]),
        "CifarHundredKnownExposed": fmt(hundred["known_size_expectation"]["exposed_test_files"]),
        "CifarHundredExposedMin": fmt(hundred["aggregate_only_bounds"]["exposed_test_files"][0]),
        "CifarHundredExposedMax": fmt(hundred["aggregate_only_bounds"]["exposed_test_files"][1]),
        "CifarHundredObservedCross": str(hundred["official_split_observation"]["crossing_groups"]),
        "CifarHundredObservedExposed": str(hundred["official_split_observation"]["exposed_test_files"]),
    }
    policy_names = {"{0}": "PZero", "{0,1}": "PZeroOne", "{0,1,2}": "PAll"}
    for policy in data["policy_sensitivity"]:
        policy_name = policy_names[policy["judgments"]]
        for row in policy["datasets"]:
            dataset_name = "Ten" if row["dataset"] == "CIFAR-10" else "Hundred"
            prefix = f"Cifar{dataset_name}{policy_name}"
            values[f"{prefix}Groups"] = str(row["annotated_group_count"])
            values[f"{prefix}Files"] = f"{row['annotated_grouped_files']:,}"
            values[f"{prefix}KnownCross"] = fmt(row["known_size_expectation"]["crossing_groups"])
            values[f"{prefix}CrossMin"] = fmt(row["aggregate_only_bounds"]["crossing_groups"][0])
            values[f"{prefix}CrossMax"] = fmt(row["aggregate_only_bounds"]["crossing_groups"][1])
            values[f"{prefix}KnownExposed"] = fmt(row["known_size_expectation"]["exposed_test_files"])
            values[f"{prefix}ExposedMin"] = fmt(row["aggregate_only_bounds"]["exposed_test_files"][0])
            values[f"{prefix}ExposedMax"] = fmt(row["aggregate_only_bounds"]["exposed_test_files"][1])
    return "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in values.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    rendered = render(json.loads(args.result.read_text(encoding="utf-8")))
    if args.verify:
        if args.output.read_text(encoding="utf-8") != rendered:
            print("ciFAIR macro mismatch.")
            return 1
        print("ciFAIR macros match the frozen public validation.")
        return 0
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote ciFAIR macros: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())