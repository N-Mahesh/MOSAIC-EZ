"""Extract frozen ViT embeddings for FER-2013 under the ASD experiment's recipe.

The cross-population study trains one classifier head on neurotypical (NT) data
and evaluates it on the autistic-children test set, and vice versa. For that
comparison to isolate the training population, both sides must pass through the
identical backbone and preprocessing transform used by
``experiments/visual_emotion_fair``: ``vit_base_patch16_224`` with timm's
inference transform, no fine-tuning.

FER-2013 ships as 48x48 grayscale PNG/JPG under ``{train,test}/{class}/``. The
timm transform upsamples to 224x224 and replicates the single channel to RGB.
That resolution and colorspace gap is a real confound for the transfer claim and
is reported alongside the result rather than hidden; see ``README.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

# Mirrors experiments/visual_emotion_fair/config.json primary_zone_map. FER-2013's
# `disgust` and `surprise` are dropped because the ASD protocol excludes the same
# two source labels, so both populations span an identical three-zone label space.
FER_ZONE_MAP = {
    "neutral": "zone1_natural_joy",
    "happy": "zone1_natural_joy",
    "angry": "zone2_anger_fear",
    "fear": "zone2_anger_fear",
    "sad": "zone3_sadness",
}
EXCLUDED_FER_LABELS = ("disgust", "surprise")
ZONE_NAMES = ("zone1_natural_joy", "zone2_anger_fear", "zone3_sadness")

VIT_MODEL_NAME = "vit_base_patch16_224"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_rows(fer_root: Path) -> list[dict[str, str]]:
    """Enumerate FER-2013 in a filesystem-order-independent way.

    ``sample_id`` is derived from the relative path so the manifest is stable
    across machines and reruns, and so a later split is reproducible from the
    identifier alone rather than from directory iteration order.
    """
    rows: list[dict[str, str]] = []
    for source_split in ("train", "test"):
        split_dir = fer_root / source_split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing FER-2013 split directory: {split_dir}")
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = class_dir.name.lower()
            if label in EXCLUDED_FER_LABELS:
                continue
            if label not in FER_ZONE_MAP:
                raise ValueError(f"Unmapped FER-2013 class directory: {class_dir}")
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                relative = image_path.relative_to(fer_root).as_posix()
                rows.append(
                    {
                        "sample_id": hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20],
                        "relative_path": relative,
                        "source_label": label,
                        "zone": FER_ZONE_MAP[label],
                        "source_split": source_split,
                    }
                )
    if not rows:
        raise RuntimeError(f"No FER-2013 images found under {fer_root}")
    return rows


def assign_splits(rows: list[dict[str, str]], validation_fraction: float, seed: int) -> None:
    """Carve a validation split out of FER-2013's own train directory.

    FER-2013's published test directory is held out untouched so the NT->NT
    number is comparable to what the dataset's own users report. The validation
    split only ever draws from ``train``; it selects a per-zone quota so the
    early-stopping signal is not dominated by the majority zone.
    """
    rng = np.random.default_rng(seed)
    by_zone: dict[str, list[dict[str, str]]] = {zone: [] for zone in ZONE_NAMES}
    for row in rows:
        if row["source_split"] == "train":
            by_zone[row["zone"]].append(row)
        else:
            row["split"] = "nt_test"
    for zone, zone_rows in by_zone.items():
        # Sort by the content-independent sample_id so the permutation depends
        # only on (seed, identifiers) and never on directory listing order.
        zone_rows.sort(key=lambda r: r["sample_id"])
        order = rng.permutation(len(zone_rows))
        cut = int(round(len(zone_rows) * validation_fraction))
        for rank, index in enumerate(order):
            zone_rows[index]["split"] = "nt_validation" if rank < cut else "nt_train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=1729)
    args = parser.parse_args()

    import timm
    import torch
    from PIL import Image, ImageOps
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    fer_root = args.fer_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(fer_root)
    assign_splits(rows, args.validation_fraction, args.split_seed)

    model = timm.create_model(VIT_MODEL_NAME, pretrained=True, num_classes=0)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    class ImageRows(Dataset):
        def __init__(self, subset):
            self.rows = subset

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            with Image.open(fer_root / row["relative_path"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                return transform(image), row["sample_id"]

    manifest_path = output_dir / "fer_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "relative_path", "source_label", "zone", "source_split", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    split_shapes = {}
    with torch.inference_mode():
        for split in ("nt_train", "nt_validation", "nt_test"):
            subset = [row for row in rows if row["split"] == split]
            loader = DataLoader(
                ImageRows(subset),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device == "cuda",
            )
            arrays, ids = [], []
            for images, batch_ids in loader:
                arrays.append(model(images.to(device)).detach().cpu().float().numpy())
                ids.extend(batch_ids)
                if len(ids) % (args.batch_size * 20) == 0:
                    print(f"{split}: {len(ids)}/{len(subset)}", flush=True)
            embeddings = np.concatenate(arrays, axis=0)
            labels = np.asarray([ZONE_NAMES.index(row["zone"]) for row in subset], dtype=np.int64)
            np.savez_compressed(
                output_dir / f"{split}.npz",
                embeddings=embeddings,
                sample_ids=np.asarray(ids),
                labels=labels,
            )
            split_shapes[split] = list(embeddings.shape)
            print(f"done {split}: {embeddings.shape}", flush=True)

    metadata = {
        "model_name": VIT_MODEL_NAME,
        "pretrained": True,
        "model_data_config": json.loads(json.dumps(data_config)),
        "device": device,
        "split_shapes": split_shapes,
        "zone_order": list(ZONE_NAMES),
        "fer_zone_map": FER_ZONE_MAP,
        "excluded_fer_labels": list(EXCLUDED_FER_LABELS),
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "manifest_sha256": sha256_file(manifest_path),
        "source_note": (
            "FER-2013 is 48x48 grayscale; the timm inference transform upsamples to 224x224 RGB. "
            "Resolution and colorspace differ from the ASD archive and are a stated confound."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
