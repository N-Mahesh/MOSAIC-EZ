# Cross-population transfer study

This directory holds the neurotypical-to-autistic transfer experiment and the
chance-corrected reanalysis of the locked `visual_emotion_fair` v2 results.

Nothing here reads or redistributes the autistic-children image archive. The ASD
arm consumes only the frozen 768-dimensional ViT embeddings and the zone labels
already produced by v2; the raw images are not required and are not present.

## Why this experiment exists

The usual evidence that emotion classifiers fail on autistic children is a
commercial classifier -- trained on undisclosed neurotypical (NT) data, with an
undisclosed architecture, label space, and preprocessing -- evaluated on an
autistic test set. Any gap measured that way confounds four things at once. This
study holds three of them fixed and varies only the training population:

| | trained on NT | trained on ASD |
|---|---|---|
| **tested on NT** | within-population reference | reverse transfer |
| **tested on ASD** | the transfer claim | locked v2 result |

Same frozen `vit_base_patch16_224` backbone, same timm inference transform, same
one-hidden-layer MLP head recipe, same three-zone label space, same early-stopping
rule. FER-2013's `disgust` and `surprise` are dropped because the ASD protocol
excludes the same two source labels, so both populations span an identical label
space.

## The confound we do not hide

FER-2013 is 48x48 grayscale. The autistic-children archive is higher-resolution
colour. The timm transform upsamples both to 224x224 RGB, but it cannot
manufacture detail that was never captured. **A raw NT->ASD drop therefore mixes
population shift with resolution and colourspace shift**, and must not be
reported as if it were population shift alone.

Three things constrain that confound:

1. **The reverse direction.** ASD->NT crosses the same domain gap in the opposite
   direction. If the drop were purely domain shift it should be roughly
   symmetric; a large asymmetry is evidence about population, not sensor.
2. **The size-matched arm.** The NT training pool is roughly eighteen times the
   ASD one. `nt_trained_matched` subsamples NT per seed to the ASD split's exact
   per-zone counts, so training-set size cannot explain a gap.
3. **The within-population references.** NT->NT and ASD->ASD establish that the
   shared recipe can learn each population when trained on it.

A fully controlled version would degrade the ASD images to 48x48 grayscale and
re-extract, which requires re-downloading the source archive. That run is not
included here, and the transfer number is reported as bounded rather than exact
until it is.

## Files

- `extract_fer_embeddings.py` — frozen ViT embeddings for FER-2013 under the ASD
  recipe. Carves a per-zone validation split from FER's own `train` directory and
  leaves FER's published `test` directory untouched.
- `transfer_study.py` — trains the head on each population and evaluates on both
  test sets, over ten seeds, with a cluster bootstrap over ASD `leakage_group`.
- `reanalyze_v2.py` — Cohen's kappa, chance-corrected balanced accuracy, and
  per-zone recall from the locked v2 confusion matrices.

## Statistical notes

Confidence intervals on the ASD test set resample **leakage groups**, not images.
The archive contains duplicate and augmentation families whose members are not
independent observations; resampling images directly would understate
uncertainty. The 251-image ASD test set contains 181 such groups.

Balanced accuracy is reported alongside Cohen's kappa because the two label
spaces compared in the original draft (six emotions, three zones) have different
chance floors -- 1/6 and 1/3. Comparing raw balanced accuracy across them
overstates what the zone reframing buys; `reanalyze_v2.py` quantifies by how
much.

## Reproduce

```bash
python experiments/cross_population/extract_fer_embeddings.py \
  --fer-root data/FER-2013 \
  --output-dir experiments/cross_population/fer_embeddings --device cpu

python experiments/cross_population/transfer_study.py \
  --asd-embeddings experiments/visual_emotion_fair/artifacts/v2/embeddings \
  --asd-manifests  experiments/visual_emotion_fair/artifacts/v2/manifests \
  --nt-embeddings  experiments/cross_population/fer_embeddings \
  --output experiments/cross_population/results/transfer_matrix.json

python experiments/cross_population/reanalyze_v2.py \
  --metrics experiments/visual_emotion_fair/artifacts/v2/results/metrics.json \
  --manifest-metadata experiments/visual_emotion_fair/artifacts/v2/manifests/metadata.json \
  --output experiments/cross_population/results/v2_reanalysis.json
```

FER-2013 embeddings and manifests are build products and are not committed. The
aggregate JSON under `results/` is.
