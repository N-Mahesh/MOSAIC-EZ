"""Cross-population transfer: does an emotion head trained on neurotypical faces
transfer to autistic children, and does it transfer back?

The published gap between neurotypical (NT) and autistic emotion recognition is
usually inferred by running a commercial classifier -- trained on undisclosed NT
data with an undisclosed recipe -- against an autistic test set. That confounds
the training population with the architecture, the label space, and the
preprocessing. This script removes those confounds: one frozen ViT backbone, one
head recipe, one three-zone label space, four train/test population pairings.

    NT  -> NT   within-population reference
    NT  -> ASD  the transfer claim
    ASD -> ASD  the locked v2 result, recomputed here from the same embeddings
    ASD -> NT   reverse direction, which bounds how much of the NT->ASD drop is
                explained by the shared resolution/colorspace domain gap rather
                than by the population

Two NT training regimes are run: the full FER-2013 train split, and a
size-matched subsample drawn to the ASD train split's per-zone counts, so the
transfer number is not confounded by NT simply having twenty times more data.

Confidence intervals on the ASD test set use a cluster bootstrap over
``leakage_group``, matching the v2 protocol: duplicate and augmentation families
are resampled as units because their members are not independent observations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import warnings
from pathlib import Path

import numpy as np

ZONE_NAMES = ("zone1_natural_joy", "zone2_anger_fear", "zone3_sadness")

TRAIN_CFG = {
    "mlp_alpha_grid": [1e-5, 1e-4, 1e-3],
    "mlp_hidden_units": 256,
    "max_epochs": 100,
    "patience": 12,
}


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "accuracy": float((y_true == y_pred).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((len(ZONE_NAMES), len(ZONE_NAMES)), dtype=np.int64)
    for true_label, predicted in zip(y_true, y_pred):
        matrix[true_label, predicted] += 1
    return matrix.tolist()


def cohens_kappa(matrix: np.ndarray) -> float:
    total = matrix.sum()
    observed = np.trace(matrix) / total
    expected = (matrix.sum(axis=0) * matrix.sum(axis=1)).sum() / (total * total)
    return float((observed - expected) / (1.0 - expected))


def train_mlp(x_train, y_train, x_val, y_val, seed: int):
    """Train the v2 MLP head recipe: one hidden layer, adam, epoch-wise early
    stopping on validation macro-F1 with alpha selected on the same signal."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    train = scaler.transform(x_train)
    validation = scaler.transform(x_val)
    best = None
    for alpha in TRAIN_CFG["mlp_alpha_grid"]:
        model = MLPClassifier(
            hidden_layer_sizes=(int(TRAIN_CFG["mlp_hidden_units"]),),
            activation="relu",
            solver="adam",
            alpha=float(alpha),
            batch_size=min(64, len(train)),
            max_iter=1,
            warm_start=True,
            shuffle=True,
            random_state=seed,
        )
        patience_left = int(TRAIN_CFG["patience"])
        local_best = None
        for epoch in range(1, int(TRAIN_CFG["max_epochs"]) + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(train, y_train)
            scores = metric_values(y_val, model.predict(validation))
            candidate = ((scores["macro_f1"], scores["balanced_accuracy"]), -epoch, -float(alpha))
            if local_best is None or candidate > local_best[0]:
                local_best = (candidate, copy.deepcopy(model), epoch)
                patience_left = int(TRAIN_CFG["patience"])
            else:
                patience_left -= 1
                if patience_left == 0:
                    break
        if best is None or local_best[0] > best[0]:
            best = (local_best[0], local_best[1], float(alpha), local_best[2])
    selected = best[1]
    return (
        lambda x: selected.predict(scaler.transform(x)),
        {"selected_alpha": best[2], "selected_epoch": best[3], "validation_macro_f1": best[0][0][0]},
    )


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Percentile CI resampling whole leakage groups with replacement.

    Resampling images independently would understate uncertainty because
    duplicate and augmentation families contribute near-identical rows.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    index_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    samples: dict[str, list[float]] = {"accuracy": [], "balanced_accuracy": [], "macro_f1": []}
    for _ in range(replicates):
        picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        index = np.concatenate([index_by_group[group] for group in picked])
        if len(np.unique(y_true[index])) < len(ZONE_NAMES):
            continue
        for key, value in metric_values(y_true[index], y_pred[index]).items():
            samples[key].append(value)
    return {
        key: {
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
            "replicates_used": len(values),
        }
        for key, values in samples.items()
    }


def load_asd_split(embedding_dir: Path, manifest_dir: Path, split: str):
    archive = np.load(embedding_dir / f"{split}.npz", allow_pickle=False)
    rows = list(csv.DictReader((manifest_dir / f"{split}.csv").open(encoding="utf-8")))
    by_id = {row["sample_id"]: row for row in rows}
    ids = archive["sample_ids"].tolist()
    if sorted(ids) != sorted(by_id):
        raise RuntimeError(f"ASD {split} embeddings and manifest disagree on sample ids")
    labels = np.asarray([ZONE_NAMES.index(by_id[sample_id]["zone"]) for sample_id in ids], dtype=np.int64)
    groups = np.asarray([by_id[sample_id]["leakage_group"] for sample_id in ids])
    return archive["embeddings"], labels, groups


def load_nt_split(embedding_dir: Path, split: str):
    archive = np.load(embedding_dir / f"{split}.npz", allow_pickle=False)
    return archive["embeddings"], archive["labels"]


def size_match(x: np.ndarray, y: np.ndarray, target_counts: dict[int, int], seed: int):
    """Draw a per-zone subsample matching the ASD train split's class counts.

    Matching per zone rather than in total keeps the class prior identical too,
    so the size-matched arm differs from the ASD arm only in which population
    the faces came from.
    """
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for label, count in sorted(target_counts.items()):
        pool = np.flatnonzero(y == label)
        if len(pool) < count:
            raise ValueError(f"NT pool for zone {label} has {len(pool)} < requested {count}")
        keep.append(rng.choice(pool, size=count, replace=False))
    index = np.sort(np.concatenate(keep))
    return x[index], y[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asd-embeddings", type=Path, required=True)
    parser.add_argument("--asd-manifests", type=Path, required=True)
    parser.add_argument("--nt-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=8675309)
    args = parser.parse_args()

    asd = {
        split: load_asd_split(args.asd_embeddings, args.asd_manifests, split)
        for split in ("train", "validation", "test")
    }
    nt = {split: load_nt_split(args.nt_embeddings, split) for split in ("nt_train", "nt_validation", "nt_test")}

    asd_train_counts = {
        label: int((asd["train"][1] == label).sum()) for label in range(len(ZONE_NAMES))
    }

    arms = {
        "asd_trained": {
            "train": (asd["train"][0], asd["train"][1]),
            "validation": (asd["validation"][0], asd["validation"][1]),
        },
        "nt_trained_full": {
            "train": (nt["nt_train"][0], nt["nt_train"][1]),
            "validation": (nt["nt_validation"][0], nt["nt_validation"][1]),
        },
        # Same NT pool, but subsampled per seed to the ASD train split's exact
        # per-zone counts. Without this arm a reviewer cannot tell whether an
        # NT->ASD drop reflects the population or simply a different train size.
        "nt_trained_matched": {
            "train": (nt["nt_train"][0], nt["nt_train"][1]),
            "validation": (nt["nt_validation"][0], nt["nt_validation"][1]),
        },
    }

    results: dict[str, dict] = {}
    for arm_name, arm in arms.items():
        per_seed = {"asd_test": [], "nt_test": []}
        details = []
        for seed in args.seeds:
            x_train, y_train = arm["train"]
            if arm_name == "nt_trained_matched":
                x_train, y_train = size_match(x_train, y_train, asd_train_counts, seed)
            predict, info = train_mlp(x_train, y_train, arm["validation"][0], arm["validation"][1], seed)
            info["seed"] = seed
            info["train_size"] = int(len(y_train))
            for target, (x_eval, y_eval) in (
                ("asd_test", (asd["test"][0], asd["test"][1])),
                ("nt_test", (nt["nt_test"][0], nt["nt_test"][1])),
            ):
                predictions = predict(x_eval)
                scores = metric_values(y_eval, predictions)
                scores["confusion_matrix"] = confusion(y_eval, predictions)
                scores["cohens_kappa"] = cohens_kappa(np.asarray(scores["confusion_matrix"]))
                per_seed[target].append(scores)
                info[f"{target}_balanced_accuracy"] = scores["balanced_accuracy"]
            details.append(info)
            print(f"  {arm_name} seed={seed} asd={info['asd_test_balanced_accuracy']:.4f} "
                  f"nt={info['nt_test_balanced_accuracy']:.4f}", flush=True)

        arm_result = {"per_seed_details": details, "targets": {}}
        for target, scores in per_seed.items():
            pooled = np.sum([np.asarray(s["confusion_matrix"]) for s in scores], axis=0)
            arm_result["targets"][target] = {
                "seed_mean": {
                    key: float(np.mean([s[key] for s in scores]))
                    for key in ("accuracy", "balanced_accuracy", "macro_f1", "cohens_kappa")
                },
                "seed_std": {
                    key: float(np.std([s[key] for s in scores], ddof=1))
                    for key in ("accuracy", "balanced_accuracy", "macro_f1")
                },
                "pooled_confusion": pooled.tolist(),
                "pooled_per_zone_recall": {
                    zone: float(pooled[i, i] / pooled[i].sum()) for i, zone in enumerate(ZONE_NAMES)
                },
            }
        results[arm_name] = arm_result

    # Cluster-bootstrap the ASD test CI for the median-performing seed of each
    # arm, so the interval describes a single realized model rather than an
    # average over models that never existed.
    for arm_name, arm_result in results.items():
        details = arm_result["per_seed_details"]
        order = sorted(details, key=lambda d: d["asd_test_balanced_accuracy"])
        median_seed = order[len(order) // 2]["seed"]
        seed_index = [d["seed"] for d in details].index(median_seed)
        scores = results[arm_name]["targets"]["asd_test"]
        x_train, y_train = arms[arm_name]["train"]
        if arm_name == "nt_trained_matched":
            x_train, y_train = size_match(x_train, y_train, asd_train_counts, median_seed)
        predict, _ = train_mlp(
            x_train, y_train, arms[arm_name]["validation"][0], arms[arm_name]["validation"][1], median_seed
        )
        predictions = predict(asd["test"][0])
        scores["cluster_bootstrap_ci_median_seed"] = {
            "seed": median_seed,
            **cluster_bootstrap_ci(
                asd["test"][1], predictions, asd["test"][2], args.bootstrap_replicates, args.bootstrap_seed
            ),
        }

    output = {
        "zone_order": list(ZONE_NAMES),
        "seeds": args.seeds,
        "asd_train_counts_by_zone": {ZONE_NAMES[k]: v for k, v in asd_train_counts.items()},
        "nt_split_sizes": {k: int(len(v[1])) for k, v in nt.items()},
        "asd_split_sizes": {k: int(len(v[1])) for k, v in asd.items()},
        "asd_test_leakage_groups": int(len(np.unique(asd["test"][2]))),
        "bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.bootstrap_seed,
                      "resampling_unit": "asd test leakage_group"},
        "arms": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n=== TRANSFER MATRIX (balanced accuracy, seed mean) ===")
    for arm_name, arm_result in results.items():
        for target, scores in arm_result["targets"].items():
            mean = scores["seed_mean"]["balanced_accuracy"]
            std = scores["seed_std"]["balanced_accuracy"]
            kappa = scores["seed_mean"]["cohens_kappa"]
            print(f"{arm_name:20s} -> {target:9s} {mean:.4f} +/- {std:.4f}  kappa={kappa:+.4f}")


if __name__ == "__main__":
    main()
