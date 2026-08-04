"""Verify the exact public ciFAIR manuscript reproduction bundle."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

EXPECTED_FILES = frozenset({
    ".gitattributes", "LICENSE", "README.md", "LICENSES.md", "LITERATURE_SEARCH.md", "TOOLCHAIN.md",
    "requirements-release.txt", "RELEASE_MANIFEST.sha256",
    "manuscript/urtc2026.tex", "manuscript/urtc2026.bib",
    "manuscript/generated_cifair_validation.tex", "manuscript/generated_utility_study.tex",
    "manuscript/generated_component_merger_study.tex", "manuscript/urtc2026.pdf",
    "claims/public_cifair_validation_results.v1.json", "claims/public_cifair_claim_ledger.v1.json",
    "claims/aggregate_utility_study.v1.json", "claims/component_merger_study.v1.json",
    "tools/generate_cifair_validation_macros.py", "tools/generate_cifair_claim_ledger.py",
    "tools/build_manuscript.py", "tools/fetch_cifair_metadata.py",
    "tools/verify_release_pdf.py", "tools/verify_public_release_bundle.py",
    "validation/public_cifair_validation.py", "validation/split_risk_theorem.py",
    "validation/aggregate_utility_study.py", "validation/component_merger_study.py",
    "tests/test_public_cifair_validation.py", "tests/test_split_risk_theorem.py",
    "tests/test_aggregate_utility_study.py", "tests/test_component_merger_study.py",
})
MANIFEST = "RELEASE_MANIFEST.sha256"
HASH_ALLOWLIST = frozenset({
    "claims/public_cifair_validation_results.v1.json",
    "claims/public_cifair_claim_ledger.v1.json",
    "validation/public_cifair_validation.py",
    "manuscript/urtc2026.tex",
    "tools/verify_release_pdf.py",
    "tools/fetch_cifair_metadata.py",
})
TEXT_SUFFIXES = frozenset({".md", ".txt", ".tex", ".bib", ".json", ".py"})
FORBIDDEN_TEXT = {
    "Windows absolute path": re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),
    "home-directory path": re.compile(r"(?:/" + "Users/|/" + r"home/)[^\s]+"),
    "private-key material": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "API-key-shaped token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "GitHub-token-shaped token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
}
HEX64 = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_under(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        # A reviewer normally runs this command from a fresh Git clone. Git's
        # administrative directory is transport metadata, not bundle content.
        if rel == ".git" or rel.startswith(".git/"):
            continue
        if path.is_symlink():
            raise ValueError(f"symlink prohibited: {rel}")
        if path.is_file():
            found[rel] = path
    return found


def run_checked(command: list[str], cwd: Path, label: str, env_extra: dict[str, str] | None = None) -> str:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env_extra:
        env.update(env_extra)
    completed = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", check=False)
    output = "\n".join(x.strip() for x in (completed.stdout, completed.stderr) if x.strip())
    if completed.returncode:
        raise ValueError(f"{label} failed (exit {completed.returncode}):\n{output}")
    return output


def verify(root: Path, cifair_meta: Path | None = None, rebuild_pdf: bool = False, build_work_directory: Path | None = None) -> list[str]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"bundle directory missing: {root}")
    files = files_under(root)
    actual = set(files)
    if actual != EXPECTED_FILES:
        raise ValueError(f"allowlist mismatch; missing={sorted(EXPECTED_FILES-actual)}; unexpected={sorted(actual-EXPECTED_FILES)}")

    entries: dict[str, str] = {}
    for number, line in enumerate(files[MANIFEST].read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\\]+)", line)
        if not match:
            raise ValueError(f"malformed manifest line {number}")
        digest, rel = match.groups()
        if rel.startswith("/") or PurePosixPath(rel).as_posix() != rel or ".." in PurePosixPath(rel).parts:
            raise ValueError(f"unsafe manifest path: {rel}")
        entries[rel] = digest
    if set(entries) != EXPECTED_FILES - {MANIFEST}:
        raise ValueError("manifest entries differ from allowlist")
    for rel, digest in entries.items():
        if sha256(root / rel) != digest:
            raise ValueError(f"manifest mismatch: {rel}")

    for rel, path in files.items():
        if rel == MANIFEST or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        value = path.read_text(encoding="utf-8")
        if rel not in HASH_ALLOWLIST and (match := HEX64.search(value)):
            raise ValueError(f"{rel} contains unallowlisted 64-hex value: {match.group(0)!r}")
        for label, pattern in FORBIDDEN_TEXT.items():
            if match := pattern.search(value):
                raise ValueError(f"{rel} contains {label}: {match.group(0)!r}")

    reports = [run_checked([sys.executable, "tools/verify_release_pdf.py", "manuscript/urtc2026.pdf"], root, "PDF check")]
    reports.append(run_checked([
        sys.executable, "tools/generate_cifair_validation_macros.py",
        "claims/public_cifair_validation_results.v1.json",
        "manuscript/generated_cifair_validation.tex", "--verify"
    ], root, "macro check"))
    reports.append(run_checked([
        sys.executable, "validation/aggregate_utility_study.py",
        "--json", "claims/aggregate_utility_study.v1.json",
        "--tex", "manuscript/generated_utility_study.tex", "--verify"
    ], root, "aggregate utility study check", {"PYTHONPATH": str(root / "validation")}))
    reports.append(run_checked([
        sys.executable, "validation/component_merger_study.py",
        "--cifair-result", "claims/public_cifair_validation_results.v1.json",
        "--json", "claims/component_merger_study.v1.json",
        "--tex", "manuscript/generated_component_merger_study.tex", "--verify"
    ], root, "component merger study check", {"PYTHONPATH": str(root / "validation")}))
    reports.append(run_checked([
        sys.executable, "tools/generate_cifair_claim_ledger.py",
        "claims/public_cifair_validation_results.v1.json",
        "claims/public_cifair_claim_ledger.v1.json", "--verify"
    ], root, "claim-ledger check"))
    env = {"PYTHONPATH": str(root / "validation")}
    reports.append(run_checked([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], root, "unit tests", env))
    if cifair_meta is not None:
        reports.append(run_checked([
            sys.executable, "validation/public_cifair_validation.py", "--meta", str(cifair_meta.resolve()),
            "--verify", "claims/public_cifair_validation_results.v1.json"
        ], root, "pinned ciFAIR metadata validation", env))
    else:
        reports.append("Pinned metadata re-computation skipped; pass --cifair-meta after cloning the cited commit.")
    if rebuild_pdf:
        command = [sys.executable, "tools/build_manuscript.py", "."]
        if build_work_directory is not None:
            command.extend(["--work-directory", str(build_work_directory.resolve())])
        reports.append(run_checked(command, root, "source-to-PDF rebuild"))
    else:
        reports.append("Source-to-PDF rebuild skipped; pass --rebuild-pdf in a documented TeX environment.")
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--cifair-meta", type=Path)
    parser.add_argument("--rebuild-pdf", action="store_true")
    parser.add_argument("--build-work-directory", type=Path)
    args = parser.parse_args()
    try:
        reports = verify(args.bundle, args.cifair_meta, args.rebuild_pdf, args.build_work_directory)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"NO-GO: {exc}", file=sys.stderr)
        return 1
    for report in reports:
        if report:
            print(report)
    print("PASS: exact public reproduction bundle verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())