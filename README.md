# Public ciFAIR reproduction bundle

Annotated release tag:
https://github.com/N-Mahesh/MOSAIC-EZ/tree/urtc2026-cifair-v11

A tag name is not itself content identity. Reviewers should record `git rev-parse HEAD`, verify the annotated tag target, and run the manifest check below. `RELEASE_MANIFEST.sha256` binds every release file except itself.

Verified Python environment: Python 3.12.5 and pypdf 6.14.2. The analysis uses only the Python standard library. Git and network access are required only to retrieve the pinned upstream metadata. The last successful retrieval and hash verification was Aug. 3, 2026. A pinned-data verification run took 134.9 seconds on the source workstation; this is an observation, not a runtime guarantee.

Environment bootstrap and quick integrity, PDF, claim-ledger, utility/merger-study, and unit-test check:

    python -m pip install -r requirements-release.txt
    python tools/verify_public_release_bundle.py .

Checksum-verifying upstream retrieval and full recomputation:

    python tools/fetch_cifair_metadata.py cifair
    python tools/verify_public_release_bundle.py . --cifair-meta cifair/meta --rebuild-pdf

Standalone source-to-PDF check in the documented TeX environment:

    python tools/build_manuscript.py .

Success ends with `PASS: exact public reproduction bundle verified.` Any manifest, PDF, checksum, schema, frozen-claim, macro, ledger, utility-study, merger-study, metadata, source-rebuild, or test mismatch exits nonzero. The bundle contains no CIFAR images or upstream annotation CSVs, so full empirical reproduction is explicitly network-dependent.

Current venue-format reference: https://urtc.mit.edu/paper_submission_2026.pdf . Technical verification does not determine conference eligibility. Written URTC guidance is still required for the disclosed AI assistance.
