# Literature search protocol

Search date: 2026-08-03. This was a focused novelty audit, not a systematic review.

## Scope

Included peer-reviewed or primary technical sources on cluster-induced train/test leakage, duplicate-aware dataset splitting, finite-population cluster allocation, occupancy/majorization, and visual duplicate audits. Sources whose only contribution was a blog explanation, generic data-cleaning advice, or model architecture were excluded.

## Search channels

- IEEE Xplore, ACM Digital Library, SpringerLink, Nature, CVF Open Access, Crossref/DOI metadata, and Google Scholar/web discovery.
- Backward and forward citation tracing from Guignard et al. (2024), DataSAIL (2025), ciFAIR (2020), and Kapoor and Narayanan (2023).

## Queries

- `cluster induced data leakage hypergeometric expected exposed test elements`
- `hidden cluster sizes aggregate bounds train test leakage`
- `duplicate clusters random split expected leakage majorization`
- `group size allocation discrete concavity balanced concentrated extrema`
- `occupancy number distinct outcomes majorization`
- `duplicate aware dataset splitting known similarities`
- `visual dataset leakage duplicate benchmark audit`

## Closest results and disposition

- Guignard et al. (2024): closest mathematical source. It derives the known-size multivariate-hypergeometric allocation and exposed-element moments. It does not optimize over hidden size vectors given only group count and total membership.
- Wong and Yue (1973) and Marshall et al. (2011): conceptual majorization antecedents. They motivate balancing/concentration arguments but do not state the fixed-size split-leakage functionals here.
- DataSAIL (2025): solves split construction with known similarities, not aggregate-only partial identification.
- ciFAIR (2020), Ramos et al. (2025), and Adimoolam et al. (2026): detect or document visual duplicates and benchmark effects; they do not derive the aggregate extrema.

No exact aggregate-only theorem was located. The manuscript therefore uses the bounded phrase `to the author's knowledge` and identifies the contribution specifically as sharp optimization of known hypergeometric expectations over the compatible integer fiber.