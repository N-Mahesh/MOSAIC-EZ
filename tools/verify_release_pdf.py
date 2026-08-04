"""Fail closed when a candidate manuscript PDF contains release hazards.

Passing this technical surface check is not institutional approval and does not
authorize submission or release.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install pypdf before running the PDF release check.") from exc


EXPECTED_TITLE_PARTS = (
    "Sharp Aggregate Bounds for Cluster-Induced Split Leakage:",
    "A Public ciFAIR Case Study",
)
ALLOWED_PUBLIC_SHA256 = (
    "4cb7e99a7dfff346082c9d8fa4c2989a196e4a37d1e58f75936046696f1ba6a4",
    "ef7b33e2f32d056cdb342046b437ee819541a07cb3bb0520a3ada9b9c035f532",
    "3891498a862f2d73df00cd5dc2ee9ae2b6dccf5238691d3c29d9abb12e1c63cb",
    "78c9d8d17eda11b468a137119ed51f09332393d65a6329035e2250888edb2760",
)
REQUIRED_LITERALS = (
    "OpenAI Codex materially assisted",
    "4c2764277f5fda8fec6784a78c1818eab13236c5",
    "83.573", "80.722", "84.446", "86.097",
    "246.137", "231.559", "248.615", "255.795",
    "82.689", "241.359", "85.665", "253.389",
    "28.5", "41.7", "52.3", "59.8", "83.8", "inconclusive", "split-conditioned",
    "288", "831", "7.617", "12.915", "4.933", "13.961", "2.700",
    "Proposition 2", "residual singleton", "urtc2026-cifair-v12", *ALLOWED_PUBLIC_SHA256,
)
FORBIDDEN_LITERALS = {
    "legacy model-centered title": "Leakage-Controlled Facial-Affect Recognition",
    "legacy architecture label": "Frozen ViT",
    "legacy head label": "MLP",
    "legacy ordinary-accuracy value": "63.2%",
    "legacy balanced-accuracy value": "58.1%",
    "legacy macro-F1 value": "57.6%",
}
FORBIDDEN_PATTERNS = {
    "64-hex identifier": re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])"),
    "Windows absolute path": re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),
    "home-directory path": re.compile(r"(?:/" + "Users/" + r"|/" + "home/" + r")[^\s]+"),
    "record-level identifier": re.compile(
        r"\b(?:sample|image|component|family)[_-][0-9A-Fa-f]{8,}\b",
        re.IGNORECASE,
    ),
    "headline numeric model metric": re.compile(
        r"\b(?:balanced accuracy|macro-F1|ordinary accuracy|test accuracy)"
        r"\s*(?:of|=|:)\s*\d",
        re.IGNORECASE,
    ),
}
ALLOWED_PUBLIC_SHA256 = (
    "4cb7e99a7dfff346082c9d8fa4c2989a196e4a37d1e58f75936046696f1ba6a4",
    "ef7b33e2f32d056cdb342046b437ee819541a07cb3bb0520a3ada9b9c035f532",
    "3891498a862f2d73df00cd5dc2ee9ae2b6dccf5238691d3c29d9abb12e1c63cb",
    "78c9d8d17eda11b468a137119ed51f09332393d65a6329035e2250888edb2760",
)
FORBIDDEN_CATALOG_KEYS = ("/AA", "/AcroForm", "/AF")
FORBIDDEN_NAME_TREES = ("/EmbeddedFiles", "/JavaScript")
FORBIDDEN_ANNOTATIONS = ("/FileAttachment", "/RichMedia", "/Movie", "/Sound", "/3D", "/Screen")
FORBIDDEN_ACTIONS = ("/JavaScript", "/Launch", "/GoToR", "/ImportData", "/SubmitForm")


def compact(text: str) -> str:
    return " ".join(text.replace("-\n", "").split())


def resolved(value: Any) -> Any:
    return value.get_object() if hasattr(value, "get_object") else value


def scan_text(label: str, text: str, failures: list[str]) -> None:
    for allowed_hash in ALLOWED_PUBLIC_SHA256:
        text = text.replace(allowed_hash, "")
    for hazard, literal in FORBIDDEN_LITERALS.items():
        if literal.casefold() in text.casefold():
            failures.append(f"{label} contains {hazard}: {literal!r}")
    for hazard, pattern in FORBIDDEN_PATTERNS.items():
        if match := pattern.search(text):
            failures.append(f"{label} contains {hazard}: {match.group(0)!r}")


def verify_pdf(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise ValueError(f"candidate PDF does not exist: {path}")
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise ValueError("candidate PDF must not be encrypted")
    page_count = len(reader.pages)
    if page_count == 0 or page_count > 5:
        raise ValueError(f"page count must be 1--5; found {page_count}")

    failures: list[str] = []
    root = resolved(reader.trailer["/Root"])
    for key in FORBIDDEN_CATALOG_KEYS:
        if key in root:
            failures.append(f"catalog contains prohibited key {key}")
    if "/OpenAction" in root:
        open_action = resolved(root["/OpenAction"])
        if not isinstance(open_action, dict) or str(open_action.get("/S", "")) != "/GoTo":
            failures.append("catalog contains a nonlocal or malformed OpenAction")
    names = resolved(root.get("/Names", {}))
    for key in FORBIDDEN_NAME_TREES:
        if key in names:
            failures.append(f"catalog name tree contains prohibited key {key}")

    metadata_text = " ".join(str(value) for value in (reader.metadata or {}).values())
    scan_text("PDF metadata", metadata_text, failures)

    page_text: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if abs(width - 612.0) > 1.0 or abs(height - 792.0) > 1.0:
            failures.append(f"page {page_number} is not US letter: {width:g}x{height:g} pt")
        text = page.extract_text() or ""
        if len(compact(text)) < 100:
            failures.append(f"page {page_number} is blank or extraction is incomplete")
        page_text.append(text)

        resources = resolved(page.get("/Resources", {}))
        xobjects = resolved(resources.get("/XObject", {}))
        for name, reference in xobjects.items():
            obj = resolved(reference)
            if obj.get("/Subtype") == "/Image":
                failures.append(f"page {page_number} contains unallowlisted raster image XObject {name}")

        for reference in page.get("/Annots", []):
            annotation = resolved(reference)
            subtype = str(annotation.get("/Subtype", ""))
            if subtype in FORBIDDEN_ANNOTATIONS:
                failures.append(f"page {page_number} contains prohibited annotation {subtype}")
            for action_key in ("/A", "/AA"):
                if action_key not in annotation:
                    continue
                action = resolved(annotation[action_key])
                action_type = str(action.get("/S", ""))
                if action_type in FORBIDDEN_ACTIONS:
                    failures.append(f"page {page_number} contains prohibited action {action_type}")

    joined = "\n".join(page_text)
    normalized = compact(joined)
    for part in EXPECTED_TITLE_PARTS:
        if compact(part) not in normalized:
            failures.append(f"expected title text is missing: {part!r}")
    whitespace_free = re.sub(r"\s+", "", joined)
    for literal in REQUIRED_LITERALS:
        haystack = whitespace_free if re.fullmatch(r"[0-9A-Fa-f]{64}", literal) else normalized
        if compact(literal) not in haystack:
            failures.append(f"required governance/disclosure text is missing: {literal!r}")
    scan_text("PDF page text", joined, failures)
    if failures:
        raise ValueError("release check failed:\n- " + "\n- ".join(failures))
    return page_count, len(normalized)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="freshly built candidate manuscript PDF")
    args = parser.parse_args()
    try:
        pages, text_chars = verify_pdf(args.pdf.resolve())
    except (OSError, ValueError) as exc:
        print(f"NO-GO: {exc}", file=sys.stderr)
        return 1
    print(f"PDF technical check passed: {pages} pages, {text_chars} normalized text characters.")
    print("This result does not authorize submission or release.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())