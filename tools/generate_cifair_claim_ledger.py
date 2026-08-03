"""Generate or verify a machine-readable manuscript macro claim ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from generate_cifair_validation_macros import render

MACRO = re.compile(r"\\newcommand\{\\([^}]+)\}\{([^}]*)\}")


def ledger(result_path: Path) -> dict[str, object]:
    raw = result_path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    rendered = render(data)
    macros = dict(MACRO.findall(rendered))
    if not macros:
        raise ValueError("no generated macros found")
    return {
        "schema_version": 1,
        "source_claim_file": "public_cifair_validation_results.v1.json",
        "source_claim_sha256": hashlib.sha256(raw).hexdigest(),
        "rounding_policy": "displayed real-valued macros use three digits after the decimal point",
        "manuscript_macro_values": macros,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    value = ledger(args.result)
    if args.verify:
        if value != json.loads(args.output.read_text(encoding="utf-8")):
            print("ciFAIR claim ledger mismatch.")
            return 1
        print("ciFAIR claim ledger matches the frozen result and generated macros.")
        return 0
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote ciFAIR claim ledger: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())