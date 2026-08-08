# Positioning for MIT URTC 2026

Status: 2026-08-07. Synthesis of two independent literature sweeps.

## The single most important context: this dataset family is being retracted

Between November 2025 and mid-2026, roughly 90 published papers built on
web-scraped autism face datasets came under scrutiny. Springer Nature retracted
or removed ~38 publications; IEEE issued Expressions of Concern on ~25 articles;
PLOS and Elsevier followed.

Two retractions land directly on our dataset:

- **Talaat, F. M., et al.**, "Real-time facial emotion recognition model based on
  kernel autoencoder and convolutional neural network for autism children,"
  *Soft Computing* (2023), retracted **16 Nov 2025**. Grounds: images of minors
  from a non-curated dataset, apparently collected from the internet without
  documented clinical history, confirmed ASD diagnosis, ethical oversight, or
  guardian consent.
- **Güngör, İ., Sarman, A., & Tuncay, S.**, "Can artificial intelligence and face
  recognition using deep learning detect emotions in children with autism?"
  *PLOS ONE* (2025), retracted **23 Dec 2025**. Same grounds.

The archive our v2 experiment used is
`fatmamtalaat/autistic-children-emotions-dr-fatma-m-talaat` -- the dataset
associated with the first retraction.

**Consequence for framing.** The provenance failure is now editorially
established fact. What nobody has supplied is the *measurement* failure that
rides on top of it. Our paper is the technical complement to an ethical
consensus that already exists. It must read unmistakably as an **audit**, never
as a performance claim. We are not building an emotion recogniser for autistic
children; we are showing that the published numbers for one cannot be believed.

This distinction is not cosmetic. URTC papers go to IEEE Xplore, and IEEE is
currently flagging papers in exactly this space. A paper whose headline is "our
model reaches 77.8%" invites the same treatment. A paper whose headline is "the
benchmark is contaminated and the reported 97.95-99.99% figures are artifacts"
is the correction the record needs.

## What we may not claim as novel

**NT->ASD transfer degradation is taken.** Grossard et al., "Children with autism
spectrum disorder produce more ambiguous and less socially meaningful facial
expressions," *Molecular Autism* 11:5 (2020), ran the controlled experiment:
157 TD + 36 ASD children, subject-independent CV, random forest on landmark and
HOG features.

| Train -> Test | Global accuracy |
|---|---|
| TD -> TD | 82.05% |
| TD -> ASD | 66.43% |
| ASD -> ASD | 69.3% |

Gaya-Morey et al. (*Multimedia Systems*, 2026) repeated it with 12 CNNs on an
adjacent population (intellectual disabilities): ~90% within-population,
<55% cross-population.

**We must cite both in the first paragraph of Related Work.** Our transfer arm is
a replication in the deep-model, in-the-wild-benchmark regime, not a discovery.

**Duplication in this dataset has been asserted before.** The FERAC authors
(IEEE 2024, DOI 10.1109/10724499) identified "class overlap and image
duplication" in the Talaat archive, removed duplicates, and dropped Sadness and
Surprise as unrecoverable. They did not quantify the duplicate rate, did not
state a detection methodology or threshold, did not measure cross-split leakage,
and did not measure what cleaning does to the measured gap. Cite them as
independent corroboration -- two groups have now flagged this -- then state
precisely what remains unmeasured.

## What is genuinely unoccupied

1. A **quantified** duplicate and label-conflict audit of an autism emotion
   dataset, with a stated detection protocol.
2. The **class asymmetry** of the contradictory labels: 279 of 280 conflicted
   samples fall in the negative-emotion zones.
3. The effect of **cleaning on the measured gap** -- no work in any modality
   measures how deduplication changes an NT/ASD transfer gap.
4. **Cluster bootstrap** over duplicate-induced dependence in a vision benchmark.

## Numbers that frame the contribution

Published claims on this dataset family:

- 99.99% and 99.8% six-emotion accuracy (IIETA; various)
- 97.95% (Hanumantharayappa & Bharamagoudra, *JTCSST* 8(3), July 2026) -- which
  explicitly claims to control leakage via "stratified 5-fold partition before
  augmentation," but performs no deduplication, so near-duplicates still cross
  folds
- 80.0% / F1 0.789 (Radočaj & Martinović, *Applied Sciences* 15(17):9555, 2025),
  Swin Transformer, on a 155-image validation set

Ethically collected corpora, by contrast, report 40-78%:

- FaceReader on the EMBOA autism corpus: **40%** (Kiejdo et al., *Sensors* 25(24):7485, 2025)
- Commercial classifiers on parent-reported autism: 57.95-64.92%
  (Kalantarian et al., *JMIR Mental Health* 7(4):e13174, 2020)
- CALMED, DE-ENIGMA, Hugging Rain Man: all in the same band

**The correlation between provenance quality and reported accuracy is the
paper's central figure.** Every benchmark reporting >90% is a web-scraped
derivative of one unverified source; every ethically collected corpus reports
40-78%. Our leakage-free number on the scraped archive, 77.8% [72.5, 83.4],
lands in the *ethical* band, not the scraped one.

Supporting analogy from an adjacent clinical field: in Alzheimer's imaging,
studies with confirmed subject-wise splitting report 66-90% (mean 78.5%) while
high-leakage-risk studies report 95-99% (mean 97.1%); within a single study,
proper validation moved accuracy 94% -> 66% (*Diagnostics* 15(18):2348, 2025).

## Framing citations worth using

- **Keating et al.**, "Mismatching Expressions," *Autism Research* 19(2):e70157
  (2026). 4,896 expressions, >265M motion-capture points. Autistic and
  non-autistic people may be "essentially speaking a different language" when
  conveying emotion -- bidirectional mismatch, not deficit. This converts the
  thesis from "models are bad at autistic faces" to "models learned one dialect
  and are scored on another."
- **Nagy, J.**, "Autism and the making of emotion AI: Disability as resource for
  surveillance capitalism," *New Media & Society* 26(8) (2024). Cite early; it
  pre-empts the reviewer who thinks we are building surveillance tech.
- **Barrett et al.**, "Emotional Expressions Reconsidered," *PSPI* 20(1) (2019).
  The validity ceiling: even a perfectly clean benchmark measures
  configuration-label agreement, not emotion.
- **Manelis-Baram et al.**, *Molecular Autism* 16:50 (2025). Three commercial
  tools on the same footage disagree by 4x on happiness detection rate, and find
  no significant ASD/control differences. Handle head-on: it shows the
  instruments are unstable, which is *why* benchmark integrity must come first.
- **EU AI Act Art. 5(1)(f)**, in force 2 Feb 2025, bans emotion inference in
  workplace and education except for medical or safety reasons -- and the
  Commission's guidance names autism diagnosis as within the medical exemption.
  We are auditing an instrument the regulator carved out an exemption *for
  autism* to permit. That raises the evidentiary bar rather than lowering it.

## Vocabulary to use

| Finding | Term | Source |
|---|---|---|
| Byte-identical images, contradictory labels | cross-label duplicate groups | Flaws of ImageNet, ICLR 2025 blogpost |
| Identical across the split | **hard leakage** (vs **soft** for near-identical) | Ramos et al., ICCV-W 2025 |
| Within one dataset's own splits | **intra-dataset leakage** | Ramos et al. 2025 |
| `_aug_N` copies crossing the split | **augmented duplicates** / **augmentation families** | Adimoolam et al., CVPR 2026 (Oral) |
| The formal mechanism | **synthesis leakage** | Apicella et al., *AI Review* 2025 |
| Our fix | group-aware / leakage-reduced splitting | DataSAIL, *Nat. Commun.* 2025; Saeb et al. 2017 |
| Original vs group-aware split, same model | **leakage ablation** | arXiv:2606.24944 (2026) |
| Our CI method | cluster bootstrap | Field & Welsh, *JRSS-B* 2007 |
| Multi-seed protocol | accounting for variance | Bouthillier et al., MLSys 2021 |
| The downstream harm story | **data cascade** | Sambasivan et al., CHI 2021 |

Say **byte-identical (identical SHA-256 digest)**, not "near-duplicate," for the
103 conflict groups. It is a stronger and cleaner claim than anything in the
near-duplicate literature, and it needs no external ground truth to establish --
which is epistemically stronger than Northcutt-style label-error detection.

## Risks to manage

1. **Percentile cluster bootstrap may under-cover** (Anglin, arXiv:2606.26422,
   2026). State the interval type explicitly and note the limitation, or move to
   BCa.
2. **Schlett et al.** (ICPRAM 2024) found duplicates had only minor impact in
   large face datasets. Pre-empt: their corpora are 10^3-10^6 images so
   duplicates are a small fraction; ours is ~10^3 where they are not.
3. **FairDeDup** (CVPR 2024) already showed deduplication can widen subgroup
   disparities. Cite as precedent for the shape of the result; differentiate on
   domain and on the class-conditional conflict mechanism.
4. **Do not overstate the transfer arm.** Grossard owns the finding.
