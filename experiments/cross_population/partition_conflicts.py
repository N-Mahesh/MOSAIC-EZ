"""Which 3-way coarsening of the label set actually absorbs the contradictions?

The protocol-stability argument for valence zones turns out to be generic: any
collapse from five classes to three narrows the spread. So stability cannot
distinguish the zone partition from an arbitrary one.

The label-conflict structure can. For each partition of the five zoneable
categories into blocks of size 2, 2, 1, count how many byte-identical duplicate
groups still straddle a block boundary. A partition that absorbs many
contradictions is one whose boundaries the filing process respected; a partition
that absorbs few is one it ignored. This is a property of the specific boundary,
not of the class count, because every partition here has the same shape.

Requires no model training -- only hashes and label lookups.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from audit_archive import scan
from control_collapse import FIVE, PARTITIONS, enumerate_2_2_1_partitions


def conflicts_under(records, mapping) -> dict:
    by_hash = collections.defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)
    groups = conflicted = files = 0
    for members in by_hash.values():
        if len(members) < 2:
            continue
        groups += 1
        if len({mapping[m["label"]] for m in members}) > 1:
            conflicted += 1
            files += len(members)
    return {
        "duplicate_groups": groups,
        "conflicted_groups": conflicted,
        "conflicted_files": files,
        "conflicted_group_share": conflicted / groups if groups else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [r for r in scan(args.archive_root.resolve()) if r["zone"] is not None]
    five_way = {label: index for index, label in enumerate(FIVE)}
    baseline = conflicts_under(records, five_way)

    named = dict(PARTITIONS)
    for label, mapping in enumerate_2_2_1_partitions():
        named.setdefault(label, mapping)

    rows = []
    for name, mapping in named.items():
        stats = conflicts_under(records, mapping)
        stats["name"] = name
        stats["absorbed_vs_five_way"] = baseline["conflicted_groups"] - stats["conflicted_groups"]
        rows.append(stats)
    rows.sort(key=lambda r: r["conflicted_groups"])

    zones = next(r for r in rows if r["name"] == "zones")
    rank = 1 + sum(1 for r in rows if r["conflicted_groups"] < zones["conflicted_groups"])

    output = {
        "five_way_baseline": baseline,
        "zone_partition": zones,
        "zone_rank_ascending": rank,
        "partitions_evaluated": len(rows),
        "ranked": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"five-way baseline: {baseline['conflicted_groups']} conflicted of "
          f"{baseline['duplicate_groups']} duplicate groups\n")
    print(f"{'partition':44s} {'confl.':>7s} {'files':>6s} {'absorbed':>9s}")
    for r in rows:
        star = "  <-- valence zones" if r["name"] == "zones" else ""
        print(f"{r['name']:44s} {r['conflicted_groups']:7d} {r['conflicted_files']:6d} "
              f"{r['absorbed_vs_five_way']:9d}{star}")
    print(f"\nzones rank {rank} of {len(rows)} (1 = fewest surviving conflicts)")


if __name__ == "__main__":
    main()
