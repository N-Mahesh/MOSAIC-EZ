# Literature search protocol

Search date: 2026-08-03. This was a focused novelty audit, not a systematic review.

## Scope

Included peer-reviewed or primary technical sources on cluster-induced train/test leakage, duplicate-aware dataset splitting, finite-population cluster allocation, observed-class and expected-distinct-value sampling, database selectivity/cardinality estimation, occupancy/majorization, and visual duplicate audits. Sources whose only contribution was a blog explanation, generic data-cleaning advice, or model architecture were excluded.

## Search channels

- IEEE Xplore, ACM Digital Library, SpringerLink, Nature, CVF Open Access, JASA, Ecology, Crossref/DOI metadata, institutional technical-report archives, and Google Scholar/web discovery.
- Backward and forward citation tracing from Guignard et al. (2024), Hurlbert (1971), Walton (1986), Christodoulakis (1989), DataSAIL (2025), ciFAIR (2020), and Kapoor and Narayanan (2023).

## Queries

- `cluster induced data leakage hypergeometric expected exposed test elements`
- `hidden cluster sizes aggregate bounds train test leakage`
- `duplicate clusters random split expected leakage majorization`
- `group size allocation discrete concavity balanced concentrated extrema`
- `finite population expected number distinct classes without replacement arbitrary frequencies`
- `multiple hypergeometric observed classes distribution moments`
- `database selectivity expected distinct values sampling without replacement duplicates`
- `duplicate record elimination equal multiplicities sampling`
- `occupancy number distinct outcomes majorization`
- `duplicate aware dataset splitting known similarities`
- `visual dataset leakage duplicate benchmark audit`

## Closest results and disposition

- Hurlbert (1971), Walton (1986), and Christodoulakis (1989): derive the arbitrary-frequency expected number of observed classes under fixed-size sampling without replacement; Walton also studies its distribution. Because a crossing count is pointwise `D_test + D_train - G`, these are direct antecedents of the crossing expectation, but they do not optimize over frequency vectors known only through group count and total membership.
- Bitton and DeWitt (1983): peer-reviewed database treatment of duplicate elimination under equal multiplicities; less general than the arbitrary-frequency sources, but a relevant application antecedent.
- Guignard et al. (2024): closest cluster-leakage source. It derives known-size multivariate-hypergeometric allocations and exposed-element expectation and variance. It does not optimize over hidden size vectors given only group count and total membership.
- Wong and Yue (1973) and Marshall et al. (2011): conceptual majorization antecedents. They supply balancing/concentration tools but do not state the aggregate-only fixed-size split bounds here.
- DataSAIL (2025): solves split construction with known similarities, not aggregate-only partial identification.
- ciFAIR (2020), Ramos et al. (2025), and Adimoolam et al. (2026): detect or document visual duplicates and benchmark effects; they do not derive aggregate extrema.

No source was located that optimizes either split-leakage functional over every integer group-size vector compatible only with `(M,G,S,T)` and proves sharp attainable extrema. The manuscript therefore states explicitly that the sampling formulas and majorization method are not new, uses the bounded phrase `to the author's knowledge`, and identifies the contribution as the partial-identification layer: discrete-concavity arguments and sharp aggregate-only optimization.
