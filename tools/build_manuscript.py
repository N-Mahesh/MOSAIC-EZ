"""Rebuild the manuscript and compare its normalized content with the release PDF."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from pypdf import PdfReader

SOURCES = (
    "urtc2026.tex",
    "urtc2026.bib",
    "generated_cifair_validation.tex",
    "generated_utility_study.tex",
)
BAD_LOG = re.compile(r"undefined|Overfull|LaTeX Error|Emergency stop|Fatal error", re.IGNORECASE)


def run(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if completed.returncode:
        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        raise ValueError(f"command failed ({completed.returncode}): {' '.join(command)}\n{output}")


def normalized_text(pdf: Path) -> str:
    reader = PdfReader(pdf)
    return re.sub(r"\s+", " ", " ".join(page.extract_text() or "" for page in reader.pages)).strip()


def geometry(pdf: Path) -> list[tuple[float, float]]:
    return [(float(page.mediabox.width), float(page.mediabox.height)) for page in PdfReader(pdf).pages]


def build(bundle: Path, output: Path | None, work_directory: Path | None = None) -> Path:
    bundle = bundle.resolve()
    manuscript = bundle / "manuscript"
    committed = manuscript / "urtc2026.pdf"
    for executable in ("pdflatex", "bibtex"):
        if shutil.which(executable) is None:
            raise ValueError(f"required executable not found: {executable}")
    if work_directory is not None:
        work_directory = work_directory.resolve()
        if not work_directory.is_dir():
            raise ValueError(f"work directory must already exist: {work_directory}")
        if any(work_directory.iterdir()):
            raise ValueError(f"work directory must be empty: {work_directory}")
        manager = nullcontext(str(work_directory))
    else:
        manager = tempfile.TemporaryDirectory(prefix="urtc-build-", dir=bundle)
    with manager as temporary:
        work = Path(temporary)
        for name in SOURCES:
            source = manuscript / name
            if not source.is_file():
                raise ValueError(f"missing manuscript source: {source}")
            shutil.copy2(source, work / name)
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "urtc2026.tex"], work)
        run(["bibtex", "urtc2026"], work)
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "urtc2026.tex"], work)
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "urtc2026.tex"], work)
        log = (work / "urtc2026.log").read_text(encoding="utf-8", errors="replace")
        if match := BAD_LOG.search(log):
            raise ValueError(f"final LaTeX log contains prohibited warning: {match.group(0)}")
        rebuilt = work / "urtc2026.pdf"
        if geometry(rebuilt) != geometry(committed):
            raise ValueError("rebuilt and committed PDF geometry differ")
        if normalized_text(rebuilt) != normalized_text(committed):
            raise ValueError("rebuilt and committed PDF normalized text differ")
        metadata = PdfReader(rebuilt).metadata or {}
        if metadata.get("/Title") != "Sharp Aggregate Bounds for Cluster-Induced Split Leakage: A Public ciFAIR Case Study":
            raise ValueError("rebuilt PDF title metadata mismatch")
        if metadata.get("/Author") != "Nikhil Mahesh":
            raise ValueError("rebuilt PDF author metadata mismatch")
        if output is not None:
            output = output.resolve()
            if output.exists():
                raise ValueError(f"refusing to overwrite output: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rebuilt, output)
            return output
        return committed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-directory", type=Path)
    args = parser.parse_args()
    try:
        result = build(args.bundle, args.output, args.work_directory)
    except (OSError, ValueError) as exc:
        print(f"NO-GO: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: manuscript source rebuild matches normalized release PDF: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())