"""Leakage ablation: how much of a reported number is the split protocol?

Same archive, same frozen ViT features, same head recipe, same label space.
The only thing that changes is how train and test are separated. Each protocol
below corresponds to a way a real paper has plausibly used this archive.

    vendor_balanced   the balanced tree's own train/ and test/ directories
    vendor_talaat     the Talaat tree's own Train/ and Test/ directories
    cross_tree        train on the balanced tree, test on the Talaat tree
                      -- the two shipped copies share 45% of the smaller one's
                      images, so this is the protocol that leaks hardest
    naive_random      pool everything, split at random, ignore duplicates
                      -- what an ImageFolder + train_test_split pipeline does
    group_aware       split over connected components of the exact-duplicate,
                      near-duplicate, and augmentation-family graph

Reporting all five side by side turns "this dataset is contaminated" from an
assertion into a measured effect size, and shows which published numbers are
reachable only by which protocol.

Run at two label granularities -- the vendor's six categories and the three-zone
collapse -- because the zone question and the leakage question interact.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import re
import warnings
from pathlib import Path

import numpy as np

from audit_archive import IMAGE_EXTENSIONS, LABEL_ALIASES, SPLIT_DIR_NAMES, ZONE_MAP, Union, phash

SIX = ("natural", "joy", "anger", "fear", "sadness", "surprise")
ZONES = ("zone1", "zone2", "zone3")
AUG = re.compile(r"(?i)_aug_[0-9]+$")

TRAIN_CFG = {"alpha_grid": [1e-5, 1e-4, 1e-3], "hidden": 256, "max_epochs": 100, "patience": 12}


def metric_values(y_true, y_pred, n_classes):
    from sklearn.metrics import balanced_accuracy_score, f1_score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "accuracy": float((y_true == y_pred).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


def kappa(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    total = matrix.sum()
    observed = np.trace(matrix) / total
    expected = (matrix.sum(axis=0) * matrix.sum(axis=1)).sum() / (total * total)
    return float((observed - expected) / (1.0 - expected)) if expected < 1 else 0.0


def train_head(x_train, y_train, x_val, y_val, seed, n_classes):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    train, validation = scaler.transform(x_train), scaler.transform(x_val)
    best = None
    for alpha in TRAIN_CFG["alpha_grid"]:
        model = MLPClassifier(
            hidden_layer_sizes=(TRAIN_CFG["hidden"],), activation="relu", solver="adam",
            alpha=float(alpha), batch_size=min(64, len(train)), max_iter=1,
            warm_start=True, shuffle=True, random_state=seed,
        )
        patience, local = TRAIN_CFG["patience"], None
        for epoch in range(1, TRAIN_CFG["max_epochs"] + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(train, y_train)
            score = metric_values(y_val, model.predict(validation), n_classes)["macro_f1"]
            cand = (score, -epoch, -float(alpha))
            if local is None or cand > local[0]:
                local, patience = (cand, copy.deepcopy(model)), TRAIN_CFG["patience"]
            else:
                patience -= 1
                if patience == 0:
                    break
        if best is None or local[0] > best[0]:
            best = local
    selected = best[1]
    return lambda x: selected.predict(scaler.transform(x))


def build_records(root: Path):
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = list(path.relative_to(root).parts)
        if len(parts) < 2:
            continue
        label = LABEL_ALIASES.get(parts[-2].lower())
        if label is None:
            continue
        import hashlib

        records.append({
            "path": path,
            "rel": path.relative_to(root).as_posix(),
            "tree": parts[0],
            "split": next((p.lower() for p in parts if p.lower() in SPLIT_DIR_NAMES), "unsplit"),
            "label": label,
            "zone": ZONE_MAP.get(label),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "stem": path.stem,
        })
    return records


def leakage_groups(records, threshold: int):
    """Connected components over exact hash, perceptual near-duplicate, and
    augmentation-family edges. This is the unit that must not straddle a split."""
    union = Union()
    for record in records:
        union.find(record["rel"])
    by_hash = collections.defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record["rel"])
    for members in by_hash.values():
        for other in members[1:]:
            union.union(members[0], other)
    by_family = collections.defaultdict(list)
    for record in records:
        by_family[f"{record['tree']}/{AUG.sub('', record['stem']).lower()}"].append(record["rel"])
    for members in by_family.values():
        for other in members[1:]:
            union.union(members[0], other)
    unique = {}
    for record in records:
        unique.setdefault(record["sha256"], record)
    keys = list(unique)
    hashes = {k: phash(unique[k]["path"]) for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if bin(hashes[a] ^ hashes[b]).count("1") <= threshold:
                union.union(unique[a]["rel"], unique[b]["rel"])
    return {record["rel"]: union.find(record["rel"]) for record in records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
    parser.add_argument("--phash-threshold", type=int, default=4)
    args = parser.parse_args()

    records = build_records(args.archive_root.resolve())
    archive = np.load(args.embeddings, allow_pickle=False)
    index = {r: i for i, r in enumerate(archive["rels"].tolist())}
    embeddings = archive["embeddings"]
    groups = leakage_groups(records, args.phash_threshold)

    trees = sorted({r["tree"] for r in records})
    balanced_tree, talaat_tree = trees[0], trees[1]

    def subset(predicate):
        return [r for r in records if predicate(r)]

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

    # Both label spaces must be evaluated over the SAME images, otherwise a
    # "categories vs zones" comparison silently compares different test sets.
    # `surprise` has no zone, so it is dropped from both arms and the categorical
    # arm becomes a five-way task over exactly the images the zone arm sees.
    zoneable = [r for r in records if r["zone"] is not None]
    FIVE = tuple(c for c in SIX if c != "surprise")

    results = {}
    for space, classes, key in (("five_category", FIVE, "label"), ("three_zone", ZONES, "zone")):
        results[space] = {}
        pool = zoneable
        label_index = {c: i for i, c in enumerate(classes)}

        for name, (train_rows, test_rows) in protocols.items():
            train_rows = [r for r in train_rows if r["zone"] is not None]
            test_rows = [r for r in test_rows if r["zone"] is not None]
            if not train_rows or not test_rows:
                continue
            results[space][name] = evaluate(
                train_rows, test_rows, embeddings, index, label_index, key, args.seeds, groups
            )

        # naive random pooled split, duplicates ignored
        for name, group_aware in (("naive_random", False), ("group_aware", True)):
            per_seed = []
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                if group_aware:
                    unique_groups = sorted({groups[r["rel"]] for r in pool})
                    order = rng.permutation(len(unique_groups))
                    cut = int(round(len(unique_groups) * 0.15))
                    test_groups = {unique_groups[i] for i in order[:cut]}
                    train_rows = [r for r in pool if groups[r["rel"]] not in test_groups]
                    test_rows = [r for r in pool if groups[r["rel"]] in test_groups]
                else:
                    order = rng.permutation(len(pool))
                    cut = int(round(len(pool) * 0.15))
                    test_rows = [pool[i] for i in order[:cut]]
                    train_rows = [pool[i] for i in order[cut:]]
                per_seed.append(
                    evaluate(train_rows, test_rows, embeddings, index, label_index, key, [seed], groups)
                )
            results[space][name] = aggregate(per_seed)

    output = {
        "n_records": len(records),
        "seeds": args.seeds,
        "leakage_group_count": len({v for v in groups.values()}),
        "protocols": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for space, entries in results.items():
        print(f"\n=== {space} ===")
        print(f"{'protocol':16s} {'bal.acc':>8s} {'sd':>6s} {'kappa':>7s} {'1NN':>7s} {'hard%':>6s} {'grp%':>6s} {'n':>5s}")
        for name, entry in entries.items():
            print(f"{name:16s} {entry['balanced_accuracy_mean']:8.4f} {entry['balanced_accuracy_std']:6.4f} "
                  f"{entry['kappa_mean']:+7.4f} {entry['nn1_balanced_accuracy']:7.4f} "
                  f"{entry['test_leaked_share']*100:5.1f}% {entry['test_group_leaked_share']*100:5.1f}% {entry['n_test']:5d}")


def nearest_neighbour_ceiling(x_train, y_train, x_test, y_test, label_index):
    """1-NN in feature space: an explicit upper bound on what memorisation buys.

    A frozen low-capacity probe cannot exploit a leaked duplicate, so it cannot
    tell us what an end-to-end fine-tuned network could extract from a
    contaminated split. 1-NN is the opposite extreme: it memorises the training
    set exactly and returns the label of the closest stored example, so a test
    image whose byte-identical twin is in training is recovered with certainty.
    Reporting it bounds the contamination effect from above without training
    anything.
    """
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier(n_neighbors=1).fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "accuracy": float((y_test == predictions).mean()),
    }


def evaluate(train_rows, test_rows, embeddings, index, label_index, key, seeds, groups):
    x_train = embeddings[[index[r["rel"]] for r in train_rows]]
    y_train = np.array([label_index[r[key]] for r in train_rows])
    x_test = embeddings[[index[r["rel"]] for r in test_rows]]
    y_test = np.array([label_index[r[key]] for r in test_rows])

    # Hold out a stratified slice of train for early stopping.
    scores, kappas = [], []
    train_hashes = {r["sha256"] for r in train_rows}
    train_groups = {groups[r["rel"]] for r in train_rows}
    leaked = sum(1 for r in test_rows if r["sha256"] in train_hashes)
    group_leaked = sum(1 for r in test_rows if groups[r["rel"]] in train_groups)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(y_train))
        cut = max(len(label_index), int(round(len(y_train) * 0.15)))
        val_idx, tr_idx = order[:cut], order[cut:]
        predict = train_head(
            x_train[tr_idx], y_train[tr_idx], x_train[val_idx], y_train[val_idx], seed, len(label_index)
        )
        predictions = predict(x_test)
        scores.append(metric_values(y_test, predictions, len(label_index)))
        kappas.append(kappa(y_test, predictions, len(label_index)))
    nn_ceiling = nearest_neighbour_ceiling(x_train, y_train, x_test, y_test, label_index)
    return {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "test_leaked_share": leaked / len(y_test),
        "test_group_leaked_share": group_leaked / len(y_test),
        "nn1_balanced_accuracy": nn_ceiling["balanced_accuracy"],
        "nn1_accuracy": nn_ceiling["accuracy"],
        "balanced_accuracy_mean": float(np.mean([s["balanced_accuracy"] for s in scores])),
        "balanced_accuracy_std": float(np.std([s["balanced_accuracy"] for s in scores], ddof=1) if len(scores) > 1 else 0.0),
        "accuracy_mean": float(np.mean([s["accuracy"] for s in scores])),
        "macro_f1_mean": float(np.mean([s["macro_f1"] for s in scores])),
        "kappa_mean": float(np.mean(kappas)),
    }


def aggregate(entries):
    out = dict(entries[0])
    for field in ("balanced_accuracy_mean", "accuracy_mean", "macro_f1_mean", "kappa_mean",
                  "test_leaked_share", "test_group_leaked_share", "n_test", "n_train",
                  "nn1_balanced_accuracy", "nn1_accuracy"):
        out[field] = float(np.mean([e[field] for e in entries]))
    out["balanced_accuracy_std"] = float(np.std([e["balanced_accuracy_mean"] for e in entries], ddof=1))
    out["n_test"] = int(round(out["n_test"]))
    return out


if __name__ == "__main__":
    main()
