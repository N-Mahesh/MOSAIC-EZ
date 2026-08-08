"""Derive chance-adjusted and per-zone statistics from the locked v2 results.

The v2 evaluation reports balanced accuracy, which has a different chance floor
for a 3-zone task (1/3) than for the 6-emotion task the original draft compared
against (1/6). Comparing those two numbers directly overstates what the zone
reframing buys. This script recomputes every model under Cohen's kappa and a
chance-corrected balanced accuracy so the zone claim is stated on a scale where
the two label spaces are commensurable.

It reads only committed aggregate artifacts -- confusion matrices and the
manifest metadata -- and writes a JSON summary. No image data is touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ZONE_NAMES = ("zone1_natural_joy", "zone2_anger_fear", "zone3_sadness")

# Reported in paper/draftv1/main.tex, Tables 2-4. Balanced accuracy over the
# label space named in the key. These are the numbers the rewrite must either
# reproduce or explicitly supersede.
DRAFTV1_REPORTED = {
    "deepface_six_emotion": {"balanced_accuracy": 0.2817, "n_classes": 6, "macro_f1": 0.2604},
    "deepface_three_zone": {"balanced_accuracy": 0.4512, "n_classes": 3, "macro_f1": 0.4566},
    "vit_mlp_three_zone": {"balanced_accuracy": 0.5813, "n_classes": 3, "macro_f1": 0.5764},
}


def cohens_kappa(confusion: np.ndarray) -> float:
    total = confusion.sum()
    observed = np.trace(confusion) / total
    expected = (confusion.sum(axis=0) * confusion.sum(axis=1)).sum() / (total * total)
    return float((observed - expected) / (1.0 - expected))


def chance_corrected_balanced_accuracy(balanced_accuracy: float, n_classes: int) -> float:
    """Rescale balanced accuracy so 0 is chance and 1 is perfect.

    For balanced accuracy the chance floor is 1/n_classes regardless of class
    prevalence, which makes this the right correction for comparing a 6-way and
    a 3-way result on one axis.
    """
    chance = 1.0 / n_classes
    return float((balanced_accuracy - chance) / (1.0 - chance))


def per_zone_recall(confusion: np.ndarray) -> dict[str, float]:
    return {
        zone: float(confusion[index, index] / confusion[index].sum()) if confusion[index].sum() else float("nan")
        for index, zone in enumerate(ZONE_NAMES)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--manifest-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest_metadata.read_text(encoding="utf-8"))

    models: dict[str, dict] = {}
    for name, payload in metrics["models"].items():
        runs = payload.get("runs", [])
        if not runs:
            continue
        # Every seed shares the locked test manifest, so summing confusions over
        # runs gives the seed-pooled contingency table for kappa. Deterministic
        # models (deepface, majority) contribute a single identical run.
        confusion = np.sum([np.asarray(run["confusion_matrix"], dtype=np.int64) for run in runs], axis=0)
        summary = payload.get("summary", {})
        balanced = summary.get("balanced_accuracy", {}).get("mean")
        models[name] = {
            "n_runs": len(runs),
            "pooled_confusion": confusion.tolist(),
            "balanced_accuracy": balanced,
            "accuracy": summary.get("accuracy", {}).get("mean"),
            "macro_f1": summary.get("macro_f1", {}).get("mean"),
            "balanced_accuracy_ci": [
                summary.get("balanced_accuracy", {}).get("lower_95"),
                summary.get("balanced_accuracy", {}).get("upper_95"),
            ],
            "cohens_kappa_pooled": cohens_kappa(confusion),
            "chance_corrected_balanced_accuracy": (
                chance_corrected_balanced_accuracy(balanced, len(ZONE_NAMES)) if balanced is not None else None
            ),
            "per_zone_recall_pooled": per_zone_recall(confusion),
        }

    # The zone-reframing claim in draftv1 compares a 6-way score against a 3-way
    # score. Restate both on the chance-corrected axis to show how much of the
    # apparent gain is just the floor moving from 1/6 to 1/3.
    zone_reframing = {}
    for key, entry in DRAFTV1_REPORTED.items():
        zone_reframing[key] = {
            **entry,
            "chance_floor": 1.0 / entry["n_classes"],
            "chance_corrected_balanced_accuracy": chance_corrected_balanced_accuracy(
                entry["balanced_accuracy"], entry["n_classes"]
            ),
        }
    six = zone_reframing["deepface_six_emotion"]["chance_corrected_balanced_accuracy"]
    three = zone_reframing["deepface_three_zone"]["chance_corrected_balanced_accuracy"]
    zone_reframing["interpretation"] = {
        "raw_balanced_accuracy_gain": (
            DRAFTV1_REPORTED["deepface_three_zone"]["balanced_accuracy"]
            - DRAFTV1_REPORTED["deepface_six_emotion"]["balanced_accuracy"]
        ),
        "chance_corrected_gain": three - six,
        "note": (
            "The headline 28.2%->45.1% gain is measured against different chance floors. "
            "On a chance-corrected axis the same relabeling moves DeepFace much less, so the "
            "zone framing must be defended on actionability rather than on discrimination gain."
        ),
    }

    audit = manifest.get("exact_label_conflict_audit", {})
    excluded_by_zone = audit.get("excluded_counts_by_zone", {})
    total_excluded = sum(excluded_by_zone.values()) or 1
    label_integrity = {
        **audit,
        "duplicate_audit": manifest.get("duplicate_audit", {}),
        "leakage_group_count": manifest.get("leakage_group_count"),
        "sample_count": manifest.get("sample_count"),
        "negative_zone_share_of_conflicts": (
            excluded_by_zone.get("zone2_anger_fear", 0) + excluded_by_zone.get("zone3_sadness", 0)
        )
        / total_excluded,
    }

    # Link the dataset defect to the model defect: if the surviving confusion in
    # the strongest model concentrates where the labels contradicted each other,
    # the residual error is partly irreducible label noise rather than a modeling
    # failure. This is a descriptive alignment, not a causal decomposition.
    if "mlp" in models:
        confusion = np.asarray(models["mlp"]["pooled_confusion"], dtype=np.float64)
        off_diagonal = confusion.sum() - np.trace(confusion)
        zone23_confusion = confusion[1, 2] + confusion[2, 1]
        label_integrity["mlp_error_concentration"] = {
            "zone2_zone3_share_of_all_errors": float(zone23_confusion / off_diagonal),
            "conflict_share_in_zone2_zone3": label_integrity["negative_zone_share_of_conflicts"],
            "note": (
                "Both quantities describe the same two zones; this is an alignment between the "
                "dataset's contradictory-label concentration and the model's residual confusion, "
                "not a claim that one causes the other."
            ),
        }

    output = {
        "source_metrics": str(args.metrics),
        "experiment_name": metrics.get("experiment_name"),
        "fixed_test_count": metrics.get("fixed_test_count"),
        "fixed_test_counts_by_zone": metrics.get("fixed_test_counts_by_zone"),
        "training_seed_count": metrics.get("training_seeds", {}).get("count"),
        "models": models,
        "zone_reframing_reanalysis": zone_reframing,
        "label_integrity": label_integrity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
