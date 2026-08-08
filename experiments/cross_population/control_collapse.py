"""Is the zone collapse special, or would any 3-way coarsening do?

Collapsing five classes to three mechanically reduces estimator variance at
fixed n: fewer, larger per-class buckets are less sensitive to which items land
where. Chance-correction fixes the floor but not that. So a drop in
protocol-induced spread under the valence zones is not by itself evidence that
*these* zones are the right partition -- any 3-way collapse should buy some of
it.

This script runs control partitions through the identical five-protocol sweep:

    zones            {natural, joy} {anger, fear} {sadness}   -- the claim
    valence_crossing {joy, anger} {natural, fear} {sadness}   -- deliberately
                     cuts across valence, so it should NOT inherit whatever
                     makes the zone boundary reproducible
    arousal_like     {natural, sadness} {joy, anger} {fear}   -- a second
                     non-valence partition, as a check that the first control
                     is not a fluke
    random_*         uniformly drawn 2-2-1 partitions

If the zone partition's spread reduction is inside the distribution of the
controls, the "zones are more protocol-stable" claim is a statement about class
count, not about zones, and the paper must say so.

Also adds a test-set bootstrap to every protocol, because seed dispersion is not
the uncertainty that matters when a protocol's test set has 69 images.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from audit_archive import ZONE_MAP
from leakage_ablation import SIX, build_records, evaluate, leakage_groups, train_head, metric_values

FIVE = tuple(c for c in SIX if c != "surprise")

PARTITIONS = {
    "zones": {"natural": 0, "joy": 0, "anger": 1, "fear": 1, "sadness": 2},
    "valence_crossing": {"joy": 0, "anger": 0, "natural": 1, "fear": 1, "sadness": 2},
    "arousal_like": {"natural": 0, "sadness": 0, "joy": 1, "anger": 1, "fear": 2},
}


def enumerate_2_2_1_partitions():
    """All distinct partitions of the five labels into blocks of size 2, 2, 1.

    Matching the zone partition's block structure exactly, so the comparison is
    against alternatives of the same shape rather than against arbitrary
    coarsenings that would differ in class balance too.
    """
    out = []
    labels = list(FIVE)
    for singleton in labels:
        rest = [l for l in labels if l != singleton]
        for pair in itertools.combinations(rest, 2):
            other = tuple(l for l in rest if l not in pair)
            key = tuple(sorted([tuple(sorted(pair)), tuple(sorted(other))]))
            mapping = {}
            for l in pair:
                mapping[l] = 0
            for l in other:
                mapping[l] = 1
            mapping[singleton] = 2
            entry = (key, singleton, mapping)
            if not any(e[0] == key and e[1] == singleton for e in out):
                out.append(entry)
    return [(f"{'|'.join('+'.join(b) for b in k)}|{s}", m) for k, s, m in out]


def chance_corrected(value: float, k: int) -> float:
    return (value - 1.0 / k) / (1.0 - 1.0 / k)


def bootstrap_ci(y_true, y_pred, groups, replicates, seed, n_classes):
    """Percentile CI resampling leakage groups, so duplicate families move together."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in unique}
    samples = []
    for _ in range(replicates):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_group[g] for g in picked])
        if len(np.unique(y_true[idx])) < n_classes:
            continue
        samples.append(metric_values(y_true[idx], y_pred[idx], n_classes)["balanced_accuracy"])
    if not samples:
        return {"lower_95": float("nan"), "upper_95": float("nan"), "replicates_used": 0}
    return {
        "lower_95": float(np.percentile(samples, 2.5)),
        "upper_95": float(np.percentile(samples, 97.5)),
        "replicates_used": len(samples),
    }


def run_partition(mapping, protocols, embeddings, index, groups, seeds, n_classes, want_ci, boot):
    """Evaluate one label partition across every protocol; return per-protocol BA."""
    out = {}
    for name, (train_rows, test_rows) in protocols.items():
        train_rows = [r for r in train_rows if r["zone"] is not None]
        test_rows = [r for r in test_rows if r["zone"] is not None]
        if not train_rows or not test_rows:
            continue
        for row in itertools.chain(train_rows, test_rows):
            row["_p"] = mapping[row["label"]]
        label_index = {i: i for i in range(n_classes)}
        result = evaluate(train_rows, test_rows, embeddings, index, label_index, "_p", seeds, groups)
        if want_ci:
            x_train = embeddings[[index[r["rel"]] for r in train_rows]]
            y_train = np.array([r["_p"] for r in train_rows])
            x_test = embeddings[[index[r["rel"]] for r in test_rows]]
            y_test = np.array([r["_p"] for r in test_rows])
            rng = np.random.default_rng(seeds[0])
            order = rng.permutation(len(y_train))
            cut = max(n_classes, int(round(len(y_train) * 0.15)))
            predict = train_head(
                x_train[order[cut:]], y_train[order[cut:]],
                x_train[order[:cut]], y_train[order[:cut]], seeds[0], n_classes,
            )
            test_groups = np.asarray([groups[r["rel"]] for r in test_rows])
            result["test_bootstrap_ci"] = bootstrap_ci(
                y_test, predict(x_test), test_groups, boot, 8675309, n_classes
            )
        out[name] = result
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 5, 7, 11, 13])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--phash-threshold", type=int, default=4)
    args = parser.parse_args()

    records = build_records(args.archive_root.resolve())
    archive = np.load(args.embeddings, allow_pickle=False)
    index = {r: i for i, r in enumerate(archive["rels"].tolist())}
    embeddings = archive["embeddings"]
    groups = leakage_groups(records, args.phash_threshold)

    trees = sorted({r["tree"] for r in records})
    balanced_tree, talaat_tree = trees[0], trees[1]

    def subset(p):
        return [r for r in records if p(r)]

    protocols = {
        "vendor_balanced": (
            subset(lambda r: r["tree"] == balanced_tree and r["split"] == "train"),
            subset(lambda r: r["tree"] == balanced_tree and r["split"] == "test"),
        ),
        "vendor_talaat": (
            subset(lambda r: r["tree"] == talaat_tree and r["split"] == "train"),
            subset(lambda r: r["tree"] == talaat_tree and r["split"] == "test"),
        ),
        "cross_tree": (
            subset(lambda r: r["tree"] == balanced_tree),
            subset(lambda r: r["tree"] == talaat_tree and r["split"] == "test"),
        ),
    }
    zoneable = [r for r in records if r["zone"] is not None]
    for name, group_aware in (("naive_random", False), ("group_aware", True)):
        rng = np.random.default_rng(1729)
        if group_aware:
            ug = sorted({groups[r["rel"]] for r in zoneable})
            order = rng.permutation(len(ug))
            test_groups = {ug[i] for i in order[: int(round(len(ug) * 0.15))]}
            tr = [r for r in zoneable if groups[r["rel"]] not in test_groups]
            te = [r for r in zoneable if groups[r["rel"]] in test_groups]
        else:
            order = rng.permutation(len(zoneable))
            cut = int(round(len(zoneable) * 0.15))
            te = [zoneable[i] for i in order[:cut]]
            tr = [zoneable[i] for i in order[cut:]]
        protocols[name] = (tr, te)

    named = dict(PARTITIONS)
    for label, mapping in enumerate_2_2_1_partitions():
        named.setdefault(f"all::{label}", mapping)

    results, spreads = {}, {}
    for name, mapping in named.items():
        want_ci = name in PARTITIONS
        per_protocol = run_partition(
            mapping, protocols, embeddings, index, groups, args.seeds, 3, want_ci, args.bootstrap
        )
        values = [e["balanced_accuracy_mean"] for e in per_protocol.values()]
        cc = [chance_corrected(v, 3) for v in values]
        spreads[name] = float(max(cc) - min(cc))
        results[name] = {
            "mapping": {k: int(v) for k, v in mapping.items()},
            "per_protocol": per_protocol,
            "chance_corrected_range": spreads[name],
        }
        print(f"{name:44s} cc-range={spreads[name]:.4f}", flush=True)

    # five-category reference range, same protocols
    five_index = {c: i for i, c in enumerate(FIVE)}
    five = {}
    for name, (train_rows, test_rows) in protocols.items():
        train_rows = [r for r in train_rows if r["zone"] is not None]
        test_rows = [r for r in test_rows if r["zone"] is not None]
        five[name] = evaluate(train_rows, test_rows, embeddings, index, five_index, "label", args.seeds, groups)
    five_cc = [chance_corrected(e["balanced_accuracy_mean"], 5) for e in five.values()]
    five_range = float(max(five_cc) - min(five_cc))

    all_control = {k: v for k, v in spreads.items() if k.startswith("all::")}
    zone_range = spreads["zones"]
    reductions = {k: (five_range - v) / five_range for k, v in spreads.items()}
    control_values = sorted(all_control.values())
    output = {
        "seeds": args.seeds,
        "five_category_chance_corrected_range": five_range,
        "zone_chance_corrected_range": zone_range,
        "zone_reduction_vs_five": reductions["zones"],
        "named_controls": {k: {"cc_range": spreads[k], "reduction": reductions[k]} for k in PARTITIONS},
        "all_2_2_1_partitions": {
            "count": len(all_control),
            "cc_range_min": control_values[0],
            "cc_range_median": float(np.median(control_values)),
            "cc_range_max": control_values[-1],
            "zone_rank_ascending": 1 + sum(1 for v in control_values if v < zone_range),
            "share_of_partitions_more_stable_than_zones": (
                sum(1 for v in control_values if v < zone_range) / len(control_values)
            ),
        },
        "per_partition": results,
        "five_category_per_protocol": five,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nfive-category cc-range      {five_range:.4f}")
    print(f"zone cc-range               {zone_range:.4f}  (reduction {reductions['zones']*100:.1f}%)")
    for k in PARTITIONS:
        if k != "zones":
            print(f"{k:28s}{spreads[k]:.4f}  (reduction {reductions[k]*100:.1f}%)")
    print(f"\nall 2-2-1 partitions: n={len(all_control)} "
          f"min={control_values[0]:.4f} median={np.median(control_values):.4f} max={control_values[-1]:.4f}")
    print(f"zones rank {output['all_2_2_1_partitions']['zone_rank_ascending']} of {len(all_control)} "
          f"({output['all_2_2_1_partitions']['share_of_partitions_more_stable_than_zones']*100:.0f}% more stable)")


if __name__ == "__main__":
    main()
