# Public ciFAIR reproduction bundle

Immutable release path:
https://github.com/N-Mahesh/MOSAIC-EZ/tree/urtc2026-cifair-v5

Verified environment: Python 3.12.5 and pypdf 6.14.2. The analysis itself uses only the Python standard library. Expected full verification time is under two minutes on a typical laptop.

Quick integrity, PDF, claim-ledger, and unit-test check:

    python tools/verify_public_release_bundle.py .

Full metadata re-computation:

    git clone https://github.com/cvjena/cifair.git cifair
    git -C cifair checkout 4c2764277f5fda8fec6784a78c1818eab13236c5
    python tools/verify_public_release_bundle.py . --cifair-meta cifair/meta

Success ends with `PASS: exact public reproduction bundle verified.` Any manifest, PDF, checksum, schema, frozen-claim, macro, ledger, metadata, or test mismatch exits nonzero. The bundle contains no CIFAR images or upstream annotation CSVs.

Technical verification does not determine conference eligibility. Written URTC guidance is still required for the disclosed AI assistance.
