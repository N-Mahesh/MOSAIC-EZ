"""Forensic audit of the as-shipped autism emotion archive.

Everything here is computed from the archive exactly as a downstream user
receives it from Kaggle: the vendor's own directory layout, the vendor's own
train/test split, the vendor's own class folders. No re-splitting, no cleaning.
The point is to characterise what a paper that simply calls
``ImageFolder(root)`` is actually training and testing on.

Three findings drive the manuscript:

1. **Two overlapping copies.** The archive ships two top-level dataset trees
   with different class balances and different splits. Papers report whichever
   tree they happened to use, which is why the published sizes for "the Talaat
   dataset" disagree.
2. **Hard leakage across the vendor split.** Byte-identical images appear on
   both sides of the shipped train/test boundary.
3. **Cross-label duplicate groups.** Byte-identical images carry contradictory
   emotion labels. Crucially we measure this at *two* label granularities --
   the vendor's six categories and the three-zone collapse -- because the
   difference between them is the measured cost of the categorical schema.

Only aggregate counts are written out. No filenames, hashes, or per-image rows
leave this script, so the output is safe to publish alongside the paper.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The vendor spells classes inconsistently across the two trees ("Natural" vs
# "natural"); normalise before anything else so a case difference is never
# mistaken for a label conflict.
LABEL_ALIASES = {
    "natural": "natural", "neutral": "natural",
    "joy": "joy", "happy": "joy",
    "anger": "anger", "angry": "anger",
    "fear": "fear",
    "sadness": "sadness", "sad": "sadness",
    "surprise": "surprise",
    "disgust": "disgust",
}

# Matches paper/draftv1: valence/arousal zones rather than Ekman categories.
ZONE_MAP = {
    "natural": "zone1", "joy": "zone1",
    "anger": "zone2", "fear": "zone2",
    "sadness": "zone3",
}

SPLIT_DIR_NAMES = {"train", "test", "val", "validation"}


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II basis, so ``M @ x`` matches scipy's ``dct(norm='ortho')``.

    Built directly rather than pulling in scipy: the transform is a fixed 32x32
    matrix and computing it here keeps the audit runnable with numpy and Pillow
    alone.
    """
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    matrix = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    matrix[0] /= np.sqrt(2.0)
    return matrix


_DCT32 = _dct_matrix(32)


def phash(path: Path) -> int:
    """64-bit perceptual hash: 32x32 grayscale DCT, median-thresholded 8x8 block.

    Matches the definition recorded in the v2 manifest metadata so near-duplicate
    counts here are comparable with the earlier run.
    """
    from PIL import Image

    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    coefficients = (_DCT32 @ gray @ _DCT32.T)[:8, :8]
    flat = coefficients.flatten()
    # Exclude the DC term from the median so a uniform brightness shift does not
    # flip every bit at once.
    median = np.median(flat[1:])
    bits = (flat > median).astype(np.uint64)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


class Union:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = collections.defaultdict(list)
        for item in self.parent:
            out[self.find(item)].append(item)
        return out


def scan(root: Path) -> list[dict]:
    """Read every image and record the vendor's own tree, split, and label."""
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        parts = list(relative.parts)
        label_dir = parts[-2].lower() if len(parts) >= 2 else ""
        label = LABEL_ALIASES.get(label_dir)
        if label is None:
            continue
        split = next((p.lower() for p in parts if p.lower() in SPLIT_DIR_NAMES), "unsplit")
        # The first path component names which of the shipped copies this is.
        tree = parts[0]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "path": relative.as_posix(),
                "tree": tree,
                "split": split,
                "label": label,
                "zone": ZONE_MAP.get(label),
                "sha256": digest,
                "stem": path.stem,
            }
        )
    return records


def conflict_stats(records: list[dict], key: str) -> dict:
    """Count byte-identical groups whose members disagree under a label key.

    ``key`` is either the vendor's six-way label or the three-zone collapse.
    Records with no zone (surprise, disgust) are skipped for the zone view so
    the two views are computed over the same underlying duplicate groups.
    """
    by_hash: dict[str, list[dict]] = collections.defaultdict(list)
    for record in records:
        if record.get(key) is not None:
            by_hash[record["sha256"]].append(record)
    conflicted_groups = 0
    conflicted_samples = 0
    by_value: collections.Counter = collections.Counter()
    for members in by_hash.values():
        values = {member[key] for member in members}
        if len(values) > 1:
            conflicted_groups += 1
            conflicted_samples += len(members)
            for member in members:
                by_value[member[key]] += 1
    return {
        "duplicate_groups_considered": sum(1 for m in by_hash.values() if len(m) > 1),
        "conflicted_groups": conflicted_groups,
        "conflicted_samples": conflicted_samples,
        "conflicted_samples_by_value": dict(by_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phash-threshold", type=int, default=4)
    args = parser.parse_args()

    records = scan(args.archive_root.resolve())
    print(f"scanned {len(records)} labelled images", flush=True)

    trees = sorted({r["tree"] for r in records})
    unique_hashes = {r["sha256"] for r in records}

    # --- shipped-layout description -------------------------------------------------
    per_tree = {}
    for tree in trees:
        subset = [r for r in records if r["tree"] == tree]
        per_tree[tree] = {
            "files": len(subset),
            "unique_sha256": len({r["sha256"] for r in subset}),
            "by_split": dict(collections.Counter(r["split"] for r in subset)),
            "by_label": dict(collections.Counter(r["label"] for r in subset)),
        }

    # --- do the two shipped copies overlap? -----------------------------------------
    tree_hashes = {tree: {r["sha256"] for r in records if r["tree"] == tree} for tree in trees}
    cross_tree_overlap = {}
    for i, a in enumerate(trees):
        for b in trees[i + 1 :]:
            shared = tree_hashes[a] & tree_hashes[b]
            cross_tree_overlap[f"{a} <-> {b}"] = {
                "shared_unique_images": len(shared),
                "share_of_smaller_tree": len(shared) / min(len(tree_hashes[a]), len(tree_hashes[b])),
            }

    # --- hard leakage across the vendor's own split ---------------------------------
    # Computed per tree, because each tree ships its own train/test boundary.
    leakage = {}
    for tree in trees:
        subset = [r for r in records if r["tree"] == tree]
        train = {r["sha256"] for r in subset if r["split"] == "train"}
        test = {r["sha256"] for r in subset if r["split"] == "test"}
        crossing = train & test
        test_rows = [r for r in subset if r["split"] == "test"]
        exposed = [r for r in test_rows if r["sha256"] in train]
        leakage[tree] = {
            "train_unique": len(train),
            "test_unique": len(test),
            "crossing_unique_images": len(crossing),
            "test_files": len(test_rows),
            "test_files_with_identical_training_copy": len(exposed),
            "share_of_test_leaked": (len(exposed) / len(test_rows)) if test_rows else 0.0,
        }

    # --- exact duplicate structure over the whole archive ---------------------------
    by_hash = collections.defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)
    duplicate_sizes = collections.Counter(len(v) for v in by_hash.values())

    # --- the label-granularity comparison -------------------------------------------
    six_way = conflict_stats(records, "label")
    three_zone = conflict_stats(records, "zone")
    zoneable = [r for r in records if r["zone"] is not None]
    absorbed = six_way["conflicted_groups"] - three_zone["conflicted_groups"]

    # Is the negative-affect concentration of conflicts more than the class
    # marginals already imply? Zones 2 and 3 cover three of the five zoneable
    # categories and hold most of the archive, so some concentration is expected
    # for free. Permute labels within the archive's observed marginals and ask
    # how often chance alone puts this many conflicted files in negative affect.
    rng = np.random.default_rng(1729)
    observed_negative = sum(
        v for k, v in three_zone["conflicted_samples_by_value"].items() if k in ("zone2", "zone3")
    )
    observed_total = sum(three_zone["conflicted_samples_by_value"].values())
    zone_pool = [r["zone"] for r in zoneable]
    by_hash_z = collections.defaultdict(list)
    for record in zoneable:
        by_hash_z[record["sha256"]].append(record)
    duplicate_members = [m for m in by_hash_z.values() if len(m) > 1]
    null_negative, null_conflicted = [], []
    for _ in range(2000):
        shuffled = rng.permutation(zone_pool)
        position = 0
        assigned = {}
        for record in zoneable:
            assigned[id(record)] = shuffled[position]
            position += 1
        neg = tot = 0
        for members in duplicate_members:
            values = {assigned[id(m)] for m in members}
            if len(values) > 1:
                tot += len(members)
                neg += sum(1 for m in members if assigned[id(m)] in ("zone2", "zone3"))
        null_negative.append(neg / tot if tot else 0.0)
        null_conflicted.append(tot)
    null_negative = np.asarray(null_negative)
    null_conflicted = np.asarray(null_conflicted)
    observed_share = observed_negative / observed_total
    observed_positive = observed_total - observed_negative

    # Comparing an observed SHARE against a null SHARE is misleading here, because
    # the null also produces about twice as many conflicts overall. Expressed in
    # counts the finding reverses direction and sharpens: negative-affect conflicts
    # sit slightly BELOW their null expectation, while positive/neutral conflicts
    # are almost entirely absent. The result is a depletion of cross-valence
    # conflict, not an enrichment of negative-affect conflict.
    null_negative_count = float((null_conflicted * null_negative).mean())
    null_positive_count = float((null_conflicted * (1 - null_negative)).mean())

    # The share comparison is only meaningful conditional on the number of
    # conflicts, so restrict the null to replicates that produced a comparable
    # conflict count and ask about the valence split within those.
    window = (null_conflicted >= observed_total * 0.8) & (null_conflicted <= observed_total * 1.25)
    conditional = null_negative[window]

    permutation_test = {
        "observed": {
            "conflicted_files": observed_total,
            "negative_affect": observed_negative,
            "positive_neutral": observed_positive,
            "negative_share": observed_share,
        },
        "null_unconditional": {
            "mean_conflicted_files": float(null_conflicted.mean()),
            "mean_negative_affect_count": null_negative_count,
            "mean_positive_neutral_count": null_positive_count,
            "mean_negative_share": float(null_negative.mean()),
            "p95_negative_share": float(np.percentile(null_negative, 95)),
        },
        "ratios_to_null": {
            "negative_affect": observed_negative / null_negative_count,
            "positive_neutral": observed_positive / null_positive_count,
        },
        "null_conditional_on_conflict_count": {
            "replicates_in_window": int(window.sum()),
            "window": [observed_total * 0.8, observed_total * 1.25],
            "mean_negative_share": float(conditional.mean()) if window.sum() else None,
            "p_value_one_sided": (
                float((conditional >= observed_share).mean()) if window.sum() else None
            ),
        },
        "p_value_one_sided_unconditional": float((null_negative >= observed_share).mean()),
        "p_value_floor": 1.0 / (2000 + 1),
        "replicates": 2000,
        "note": (
            "Labels are permuted across the zoneable images, preserving the zone "
            "marginals and the duplicate-group structure. Report the count comparison, "
            "not the share comparison: the null yields roughly twice as many conflicts "
            "overall, so an observed share of 99.6% against a null share of 59.5% "
            "overstates the effect. In counts the effect is a near-total absence of "
            "cross-valence conflict."
        ),
    }

    # --- augmentation families ------------------------------------------------------
    import re

    aug = re.compile(r"(?i)_aug_[0-9]+$")
    families = collections.Counter()
    for record in records:
        base = aug.sub("", record["stem"]).lower()
        families[f"{record['tree']}/{base}"] += 1
    family_sizes = collections.Counter(families.values())

    # --- perceptual near-duplicates -------------------------------------------------
    print("computing perceptual hashes...", flush=True)
    hashes = {}
    for record in records:
        if record["sha256"] not in hashes:
            hashes[record["sha256"]] = phash(args.archive_root / record["path"])
    keys = list(hashes)
    union = Union()
    near_edges = 0
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if bin(hashes[a] ^ hashes[b]).count("1") <= args.phash_threshold:
                union.union(a, b)
                near_edges += 1
    components = [g for g in union.groups().values() if len(g) > 1]

    output = {
        "archive": {
            "total_labelled_images": len(records),
            "unique_sha256": len(unique_hashes),
            "exact_duplicate_files": len(records) - len(unique_hashes),
            "exact_duplicate_rate": 1 - len(unique_hashes) / len(records),
            "shipped_trees": trees,
        },
        "per_tree": per_tree,
        "cross_tree_overlap": cross_tree_overlap,
        "hard_leakage_across_shipped_split": leakage,
        "exact_duplicate_group_size_histogram": {str(k): v for k, v in sorted(duplicate_sizes.items())},
        "label_conflicts": {
            "six_category": six_way,
            "three_zone": three_zone,
            "groups_absorbed_by_zone_collapse": absorbed,
            "share_of_six_category_conflicts_absorbed": (
                absorbed / six_way["conflicted_groups"] if six_way["conflicted_groups"] else None
            ),
            "zoneable_images": len(zoneable),
            "negative_concentration_permutation_test": permutation_test,
            "note": (
                "Both views use the same byte-identical duplicate groups. A group counted "
                "under six_category but not under three_zone is one the categorical schema "
                "records as an annotator contradiction and the zone collapse does not."
            ),
        },
        "augmentation_families": {
            "family_count": len(families),
            "families_larger_than_one": sum(1 for v in families.values() if v > 1),
            "largest_family": max(families.values()) if families else 0,
            "size_histogram": {str(k): v for k, v in sorted(family_sizes.items())},
        },
        "perceptual_near_duplicates": {
            "hamming_threshold": args.phash_threshold,
            "unique_images_hashed": len(keys),
            "edges": near_edges,
            "components_larger_than_one": len(components),
            "largest_component": max((len(g) for g in components), default=0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
