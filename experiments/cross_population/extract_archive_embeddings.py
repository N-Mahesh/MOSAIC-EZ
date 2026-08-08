"""Frozen ViT embeddings for every labelled image in an archive tree.

Keyed by path relative to the archive root, so the leakage ablation can slice
the same feature matrix under any split protocol without ever re-running the
backbone. Identical recipe to the ASD arm: ``vit_base_patch16_224``, timm
inference transform, no fine-tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIT_MODEL_NAME = "vit_base_patch16_224"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    import timm
    import torch
    from PIL import Image, ImageOps
    from torch.utils.data import DataLoader, Dataset

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)

    root = args.root.resolve()
    rels = [
        p.relative_to(root).as_posix()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    print(f"{len(rels)} images", flush=True)

    model = timm.create_model(VIT_MODEL_NAME, pretrained=True, num_classes=0).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    class Rows(Dataset):
        def __len__(self):
            return len(rels)

        def __getitem__(self, i):
            with Image.open(root / rels[i]) as image:
                return transform(ImageOps.exif_transpose(image).convert("RGB")), i

    arrays, order = [], []
    with torch.inference_mode():
        for images, idx in DataLoader(Rows(), batch_size=args.batch_size, shuffle=False):
            arrays.append(model(images).detach().cpu().float().numpy())
            order.extend(idx.tolist())
            if len(order) % (args.batch_size * 10) == 0:
                print(f"{len(order)}/{len(rels)}", flush=True)

    embeddings = np.concatenate(arrays, axis=0)
    assert order == list(range(len(rels))), "loader returned rows out of order"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, embeddings=embeddings, rels=np.asarray(rels))
    print(json.dumps({"shape": list(embeddings.shape), "model": VIT_MODEL_NAME}, indent=2))


if __name__ == "__main__":
    main()
