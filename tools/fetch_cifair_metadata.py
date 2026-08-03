"""Fetch the pinned ciFAIR metadata commit and verify the four source CSV hashes."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

COMMIT = "4c2764277f5fda8fec6784a78c1818eab13236c5"
FILES = {
    "duplicates_cifar10.csv": "4cb7e99a7dfff346082c9d8fa4c2989a196e4a37d1e58f75936046696f1ba6a4",
    "duplicates_cifar10_test.csv": "ef7b33e2f32d056cdb342046b437ee819541a07cb3bb0520a3ada9b9c035f532",
    "duplicates_cifar100.csv": "3891498a862f2d73df00cd5dc2ee9ae2b6dccf5238691d3c29d9abb12e1c63cb",
    "duplicates_cifar100_test.csv": "78c9d8d17eda11b468a137119ed51f09332393d65a6329035e2250888edb2760",
}


def run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise ValueError(f"command failed ({completed.returncode}): {' '.join(command)}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fetch(destination: Path, repository: str) -> Path:
    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing destination: {destination}")
    run(["git", "clone", "--no-checkout", "--filter=blob:none", repository, str(destination)])
    run(["git", "-C", str(destination), "checkout", "--detach", COMMIT])
    meta = destination / "meta"
    for name, expected in FILES.items():
        observed = digest(meta / name)
        if observed != expected:
            raise ValueError(f"hash mismatch for {name}: {observed}")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repository", default="https://github.com/cvjena/cifair.git")
    args = parser.parse_args()
    try:
        meta = fetch(args.destination, args.repository)
    except (OSError, ValueError) as exc:
        print(f"NO-GO: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: pinned ciFAIR metadata retrieved and verified: {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())