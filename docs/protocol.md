# Experimental protocol

This document defines the protocol shared by all models. Its machine-readable
counterpart is [`configs/protocol.yaml`](../configs/protocol.yaml). Dataset facts
(file layout, machine IDs, sampling rate, channels, clip pairing across SNRs) are
documented separately in `docs/data.md`, from inspection of the downloaded files.

## Research question

When the background-noise level at test time differs from the one seen during
training (e.g. a model trained in a quiet plant and deployed in a noisy one), how do
the main families of unsupervised anomalous sound detection (ASD) models behave?
Does multi-condition training (several noise levels mixed) close the gap?

## Data

MIMII (Purohit et al., 2019; Zenodo record 3384388): four machine types (fan, pump,
slider, valve), four machine IDs per type, three signal-to-noise ratios
(−6, 0, +6 dB). The three SNR versions are obtained by mixing the same machine
recordings with real factory noise, so every clip exists at the three SNRs.

**Excluded machine ID: valve `id_00`.** None of its anomalous clips can be
identified across SNRs (see "Clip identity" and [`data.md`](data.md)), so the group
has no test anomaly. Valve `id_00` is removed entirely from the main analyses:
from training and test, for every model, including as a class of the auxiliary
ID-classification tasks, so that the five models see exactly the same data. The
study therefore covers 15 machine IDs; valve results average 3 IDs. This is the
only exception to the rule that all groups are kept ("Small groups" below): the
group is empty, not small (`dataset.excluded_machine_ids`).

## Clip identity

A **clip** is one machine recording. MIMII provides each recording at three SNRs,
but file names do not identify recordings across SNRs: at −6 dB, part of the files
hold the recording of a differently named 0 dB file (see
[`data.md`](data.md)). Clip identity is therefore established **by content**, with
the rule below. The rule was fixed on 2026-09-29, before any exclusion count was
computed with it; its thresholds were chosen after an exploratory matching pass on
single groups (reported in `data.md`), not tuned on exclusions.

**Reference.** Each 0 dB file defines one candidate identity. The machine type,
machine ID and label (normal/anomalous) of a clip are those of its 0 dB file.

**Score.** s(a, b) is the Pearson correlation of the two 10-s waveforms at lag 0,
averaged over the 8 channels.

**Candidates.** A reference is compared with every file of the same machine type
at the other SNR, across all machine IDs and both labels, so that
correspondences crossing an ID or label boundary are detected.

**Score matrices.** At +6 dB: S⁺[i, j] = s(x⁰ᵢ, x⁺⁶ⱼ). At −6 dB, for identities
with an accepted +6 dB match π(i):
S⁻[i, j] = ½ [s(x⁰ᵢ, x⁻⁶ⱼ) + s(x⁺⁶_π(i), x⁻⁶ⱼ)].

**Acceptance.** A pair (i, j) is accepted if and only if:

1. it belongs to the one-to-one assignment maximizing the total score
   (Hungarian algorithm);
2. it is a mutual best match: j maximizes row i and i maximizes column j;
3. S[i, j] ≥ τ = 0.10;
4. S[i, j] − max(best other score in row i, best other score in column j)
   ≥ δ = 0.05;
5. at −6 dB only, both terms s(x⁰ᵢ, x⁻⁶ⱼ) and s(x⁺⁶_π(i), x⁻⁶ⱼ) are ≥ 0.05.
   The two reference versions carry independent noise, so a correlation caused
   by a noise segment shared with only one of them fails this test.

**Kept identities.** An identity is kept if and only if it has accepted matches at
both +6 dB and −6 dB. Otherwise it is excluded at all three SNRs, so that the test
sets of the three SNRs contain exactly the same recordings. A file that is not
the accepted match of a kept identity is never used.

**Validation.** Between 0 and +6 dB, names agree for 99.9 % of the clips
(exploratory pass), so the same-name file serves as ground truth there: the
precision (accepted pairs whose names agree) and recall (clips accepted with their
same-name file) of the rule are reported. At −6 dB, the internal consistency of
accepted pairs is reported: the two terms of S⁻, each computed from an
independently noised version, must point to the same −6 dB file.

## Train/test split

**Clusters.** Some distinct files share machine sound at the same SNR
([`data.md`](data.md)). Kept identities whose accepted +6 dB files score ≥ 0.5
(`split.cluster_min_score_p6`, same score as in "Clip identity") are linked, and
the connected components of these links form **clusters**. All identities of a
cluster go to the same side, training or test. The threshold was fixed on
2026-09-29, before the split was generated.

For each (machine type, machine ID), following the MIMII baseline protocol,
applied to the clusters of kept identities:

- **test set**: every anomalous clip, with its whole cluster, plus normal clips
  added cluster by cluster until the test set holds as many normal as anomalous
  clips. Normal-only clusters are visited in a random order drawn with a fixed
  seed (`split.seed`, one generator per machine ID); a cluster is added if it
  does not exceed the remaining need. If no combination reaches the exact count,
  the closest count is kept (the smallest remaining cluster is added if that
  brings the count closer) and the difference is reported per group;
- **training set**: all remaining normal clips.

The split is generated once and saved as a versioned CSV (with the cluster of
every identity); all experiments read that file. It is never regenerated as a side
effect of a run.

**Leakage.** A clip exists at three SNRs, so splitting by file (or by file name)
would let a test recording appear in training at another SNR. Automated tests check,
through the content-based identity mapping, that no recording used for training at
any SNR is used for testing at any SNR, that each kept identity has exactly one
file at each SNR, in the same partition, and that no cluster is split.

**Noise reuse check.** Factory-noise segments are reused across files
([`data.md`](data.md)). Before the split is saved, for each SNR, the share of
normal and of anomalous test files that share a noise segment with at least one
training file (score ≥ 0.5 between the two files at that SNR; pairs sharing
machine sound are in the same cluster, hence never across the split) is computed,
among the training files of the same machine ID and of the same machine type. If
the two shares differ by 5 percentage points or more at some SNR, the procedure
stops for review: a model with memory (kNN) could exploit the difference.

**Noise reuse across machine types.** The same measurement is made against the
training files of all machine types (the cross-type models' training data), for
the main and the 2×2 splits: for each SNR, share of normal and of anomalous test
files with a noise partner (same-SNR score ≥ 0.5) among training files of any
machine type, and among those of other machine types. A normal/anomalous
difference of 5 points or more stops the procedure for review.

**Small groups.** All groups are kept whatever their remaining size (the only
exception is the empty group valve `id_00`, see "Data"). The number of test clips
of each group is published, and each per-group AUC comes with a 95 % bootstrap
confidence interval: 1000 resamples of the test clips, with replacement and
stratified by label (normal and anomalous clips resampled separately), percentile
method, fixed seed (`evaluation.bootstrap`).

**Strict identity variant (evaluation only).** For 6.1 % of the accepted −6 dB
matches, the two score terms do not both select the matched file (`data.md`). The
rule is kept as pre-registered. As a sensitivity check, the models already trained
are also evaluated on the test set without these identities (removed at all three
SNRs, so that the test recordings stay identical across SNRs), with no
retraining.

**Sensitivity to the exclusion of clips.** Excluding clips that cannot be
identified at −6 dB may bias the test sets. A separate 2×2 experiment measures
it: training at 0 or +6 dB, testing at 0 and +6 dB, on **all clips identifiable
between 0 and +6 dB** (accepted +6 dB match; valve `id_00` excluded), with its own
cluster split built by the same procedure and seed (`sensitivity`). Its results
are compared with the corresponding cells of the main matrix. Its compute cost is
estimated in Phase 5.

## Experiments

Let S = {−6, 0, +6} dB.

1. **SNR mismatch matrix (3×3).** Train at s ∈ S, test at every s′ ∈ S.
2. **Multi-condition training.** Train on the union of the three SNR versions of
   the same training clips, test at every s′ ∈ S.

Each training configuration is repeated with 3 seeds (`training.seeds`). A trained
model is evaluated on the three test SNRs; test SNR never influences training.

The granularity of models (one model per machine type and ID, one per machine type,
or a single model) follows each original method and is documented per model in
[`deviations.md`](deviations.md).

**Training scope of the models.** Each model keeps the training scope of its
original method:

| Model | Trained on | Source |
|---|---|---|
| Autoencoder | one model per machine type and machine ID | MIMII paper, Sec. 4 |
| MobileNetV2 | one model per machine type, classifying its machine IDs | DCASE 2022 baseline code (`00_train.py`, commit `aaf3932`: `model_<machine_type>`) |
| EAT + kNN | no training (frozen encoder pre-trained on AudioSet); kNN reference set = training clips of the same machine ID | design choice of this study |
| Wilkinghoff (SSL/FeatEx) | one model on all machine types, classes = machine type × machine ID | Wilkinghoff 2024, Sec. 2 |
| OS-SCL | one model on all machine types, classes = machine IDs | Huang et al. 2025, Sec. III-A |

This asymmetry is part of what is compared: Wilkinghoff and OS-SCL see the
training clips of every machine type (more data, more varied noise, an auxiliary
task across types), while the autoencoder, MobileNetV2 and the kNN of EAT only
see clips of one machine type or ID. Differences between the two groups under
SNR mismatch therefore mix the effect of the method with that of the training
scope; the study compares methods as designed, not at equal training scope. One
specific risk is measured before any model is trained: factory-noise segments
shared between test files and training files of *other* machine types, a
shortcut available to the two cross-type models only (see "Noise reuse check").

**Training budget of the multi-condition model.** The multi-condition model is
trained for the **same number of optimization steps** as the single-SNR model of
the same method, machine and seed (that number is set by the single-SNR
configuration: original number of epochs × steps per epoch on one SNR). Its
training pool is three times larger, so each training example is seen about a
third as often. The question is whether noise diversity helps; with three times
more steps, a gain could come from longer training alone. At equal steps, only the
diversity of the noise changes.

EAT is not trained (frozen encoder, kNN scoring). Its multi-condition variant uses
a reference set holding the embeddings of the training clips at all three SNRs;
no step budget applies. Because EAT involves no random choice, its three seeds
give identical results; this is stated in the result tables.

**EAT + kNN configuration (fixed on 2026-09-30, before any embedding was
computed; `configs/models/eat_knn.yaml`).**

- *Encoder:* EAT-base, epoch 30, **pre-trained only** (self-supervised on the
  audio of AudioSet-2M; checkpoint `worstchan/EAT-base_epoch30_pretrain`, revision
  `6eb9593`, weights SHA-256 `8623072d…`). The checkpoint fine-tuned on AudioSet
  labels is not used: the encoder depends on no human annotation, which keeps the
  comparison within unsupervised anomalous sound detection (the only labels any
  model uses are the machine IDs, as auxiliary targets for the classifiers).
- *Input:* channel 0, as for every model; waveform in [−1, 1] with its mean
  removed; Kaldi filterbank of the model card (128 bins, 25 ms window, 10 ms
  shift, Hanning, HTK-compatible, no dither); zero-padded from 998 to 1,024
  frames; normalized as (x + 4.268) / (2 × 4.569).
- *Embedding (main):* last-layer output of the encoder (1 CLS token + 64 time ×
  8 frequency patch tokens); mean over the 64 time positions for each frequency
  position, concatenated: 6,144 dimensions ("temporal mean", as Saengthong &
  Shinozaki 2026 for EAT). The two time positions covering the padding are
  included, as in the frame-level output of the model card.
- *Scoring (main):* cosine distance (1 − cosine similarity) to the **nearest**
  embedding (K = 1) of the reference set: the normal training clips of the same
  machine ID at the training SNR (multi-condition: at all three SNRs).
- *Numerics:* float32 on GPU with TF32 disabled.

**Sensitivity analysis of the EAT scoring (secondary; rule fixed on 2026-09-30,
before any embedding was computed).** From the same cached embeddings, the
anomaly scores are also computed for the variants pooling ∈ {temporal mean
(main), CLS token} × K ∈ {1 (main), 5} (K = 5: mean cosine distance to the five
nearest reference embeddings). The variants are reported in a separate table and
never replace the main configuration, even where a variant scores higher: their
purpose is to show how much the absolute level of EAT depends on these choices,
not to select the best one. They cover every training condition and test SNR of
the main split with the Phase 6 runs; in Phase 4 they are computed for the
internal-control condition only (training at 0 dB).

**Input length of the Wilkinghoff model (decided on 2026-09-30).** In the study
(MIMII), the native 10 s clips (160,000 samples) are used, a deviation from the
paper (18 s, DCASE 2023). In the fidelity runs, each dataset uses the length the
author uses for it: every waveform is repeated (and randomly cropped) to the
length of the longest clip of the dataset (DCASE 2023 Task 2 technical report of
the author, Sec. 2.1), i.e. 160,000 samples (10 s) for DCASE 2022 and 288,000
(18 s) for DCASE 2023 (paper, Fig. 1). The two settings are never mixed.

## Metrics

Computed per (machine type, machine ID, training condition, test SNR, seed):

- **AUC**: `sklearn.metrics.roc_auc_score(y, s)`, anomalous = positive class,
  higher score = more anomalous.
- **pAUC**: `sklearn.metrics.roc_auc_score(y, s, max_fpr=0.1)`.

Note on pAUC: with `max_fpr`, scikit-learn returns the **McClish-standardized**
partial AUC, which maps chance level to 0.5 and a perfect detector to 1.0. This is
what the official DCASE evaluation code computes and what published DCASE tables
report. It differs from a literal reading of Eq. (3) in Dohi et al. (2022), whose
chance level is about `max_fpr / 2` (checked numerically: random scores give 0.50
with scikit-learn and 0.05 with the literal formula). Values reported here are
therefore comparable with published DCASE pAUCs.

**Main indicator.** For a training SNR s and a test SNR s′:

    ΔAUC(s → s′) = AUC(train s, test s′) − AUC(train s′, test s′)

i.e. the AUC change caused by the mismatch, relative to the matched model at the
test SNR (ΔAUC = 0 on the diagonal; negative = degradation). For multi-condition
training, the same definition is used with "train multi" in place of "train s".

## Aggregation

1. AUC and pAUC are computed per machine ID (for a given training condition, test
   SNR and seed).
2. The score of a machine type is the arithmetic mean over its machine IDs.
3. The overall score is the arithmetic mean over the four machine types.
4. **Seeds.** ΔAUC is computed seed by seed: for seed k,
   ΔAUC_k(s → s′) = AUC_k(train s, test s′) − AUC_k(train s′, test s′), where both
   terms are aggregated as above. Tables report the mean ± sample standard
   deviation (ddof = 1) of ΔAUC_k over the three seeds. AUC and pAUC entries are
   reported the same way (mean ± standard deviation over seeds of the aggregated
   value).

Pairing seed k with seed k is meaningful here: two models trained with the same
seed on different single SNRs start from the same initialization and see the
training clips in the same order (the training list is ordered by clip identity,
which is identical at every SNR), so they differ only in the noise level. Since
the means over IDs and machines are linear, aggregating then differencing equals
differencing per ID then aggregating.

## Hyperparameters

No hyperparameter is tuned on test data. Values come from the original paper or
code; when neither specifies one, a documented value is fixed once, before any
experiment, and used for all conditions. Every such choice is listed in
[`deviations.md`](deviations.md).

## Freeze of the protocol (rule adopted on 2026-10-02, before any study run)

From the moment the first results of the study exist, no protocol decision is
changed: no new threshold, no new variant, no rewording of a criterion, no
additional exclusion. If a problem is found that seems to require one, the work
stops and the problem is reported, without modifying anything. Any change made
after that point is dated, justified and presented as posterior to the results.
At the time this rule was written, no run of the study (Phase 6) had been
launched; the only AUCs computed on the study data were implementation checks of
Phase 4 (training at 0 dB, one seed), reported as such in
[`validation.md`](validation.md).

## Analyses added after the results

Each analysis below was decided after results of the study had been seen; none
changes a criterion, a threshold, an exclusion or the primary measure.

**Relative error (secondary analysis, decided on 2026-10-02, after the
stage-A results had been seen).** The primary measure of the effect of a
mismatch remains ΔAUC in points, as pre-registered; it is not modified. Reason
for the addition: AUCs close to the ceiling make equal ΔAUC incomparable
between models. Example seen in the stage-A results (training at +6 dB against
training at −6 dB, test at −6 dB): Wilkinghoff 97.6 → 85.3 (−12.3 points) and
the autoencoder 65.9 → 56.1 (−9.8 points) look alike in points, whereas the
error 1 − AUC is multiplied by 6.1 for the former and by 1.3 for the latter.
Definition (fixed on 2026-10-02, before any computation of the ratio beyond
this example): with the aggregated AUC of the protocol (mean over
IDs, then over machine types), for seed k, ratio_k = (1 − AUC_k(train s, test
s′)) / (1 − AUC_k(train s′, test s′)); reported as mean and sample standard
deviation over seeds, next to ΔAUC, never instead of it.
*Ratio of aggregates.* The ratio is computed on the aggregated AUCs; it is not
the mean over machine IDs of per-ID ratios, and the two are not equal. The
per-ID ratio is not used because several machine IDs have an AUC of 1.000 in
the matched condition (zero denominator).
*Instability near the ceiling.* The ratio becomes unstable when the matched
AUC approaches 1: at a matched AUC of 99.3, a difference of 0.3 point already
gives a ratio of 1.4. The standard deviation over seeds is therefore given
next to every value; the ratio is never reported alone.

*Note added on 2026-10-04 (the paragraph above is unchanged).* The illustrative
factor 6.1 was computed on the seed means of the aggregated AUCs; the defined
measure, the mean of the per-seed ratios, gives 6.3 ± 1.0 for the same cell
(Wilkinghoff, trained at +6 dB, tested at −6 dB).

**Descriptive analyses added on 2026-10-04, after all results had been seen.**
None changes a criterion or the primary measure; all are computed from the saved
scores (`results/study_report.md`): (i) for the hypothesis on the direction of
the loss, ΔAUC of the two extreme mismatches per seed, of the four 6 dB
mismatches, and the number of machine IDs whose loss is larger towards the clean
side; (ii) the model with the largest loss under each of the two measures, per
mismatched cell; (iii) the distribution of per-ID AUCs in the matched cells and
the width of the per-ID bootstrap intervals; (iv) the comparison of Wilkinghoff
and OS-SCL, trained and tested at −6 dB, with published DCASE 2020 values per
machine type; (v) in the figure of ΔAUC against the SNR gap, the two cells with
the same gap of ±6 dB are averaged seed by seed (all cells are in the tables).

**Interval on aggregated values (decided on 2026-10-04, after all results had
been seen).** The protocol gives intervals per machine ID only; aggregated values
and ΔAUC had the standard deviation over seeds alone. Added: a 95 % percentile
interval from a paired bootstrap of the test clips. For each machine ID, its
test identities are resampled with replacement, normal and anomalous clips
separately (counts kept), 1000 resamples, one generator per machine ID seeded
with `evaluation.bootstrap.seed` and the machine ID. The same draws are used for
every model, training condition, seed and test SNR (the test identities are the
same in all of them), so that differences between conditions and between models
are paired. For each resample the per-ID AUCs are recomputed, aggregated as in
"Aggregation", and the statistic is computed exactly as its point estimate
(seed-paired ΔAUC averaged over seeds, ratio of errors, differences between
models); the point estimate is that of the full test set
(`scripts/study_analysis.py bootstrap`). *Limitation:* this interval reflects the
sampling uncertainty of the test clips of these 15 machines, not the
generalization to other machines; machine IDs are not resampled, no second
bootstrap level over machines is added. The trained models are fixed: training
randomness is given separately by the standard deviation over seeds and is not
in the interval.

*Extension added on 2026-10-05, after the results (same method, same draws).*
For the article, the paired bootstrap above is also applied to the pAUC, to the
pre-registered robustness evaluations (strict variant, aggregation without
`fan id_00`), to the means per machine type, and to the values of the figure of
ΔAUC against the SNR gap (the two cells of a ±6 dB gap averaged seed by seed),
and, in pAUC as in AUC, to the paired differences between models, the direction
of the larger loss and the differences between models in the matched cells
(order of the models), to check whether the conclusions hold in pAUC.
Each statistic is computed on each resample exactly as its point estimate; the
AUC intervals of the main evaluation are unchanged
(`results/aggregated/bootstrap_intervals_<split>.csv`, columns `evaluation`,
`metric`, `statistic`).

## Known limitations

- **Unequal training sets created by the identity exclusions (slider `id_00`).**
  In the MIMII release, slider `id_00` has 1,068 normal and 356 anomalous clips
  per SNR, and `id_02` 1,068 and 267: the MIMII baseline split would train on
  712 and 801 normal clips. Content-based identity keeps 534 of the 1,068 normal
  clips of `id_00` (50 % excluded, [`data.md`](data.md)); 356 of them go to the
  test set to match its anomalous clips, which leaves 178 training clips, against
  708 for `id_02`, 301 for `id_04` and 396 for `id_06`. The imbalance is a
  product of this protocol, not of the dataset, and it may affect every model on
  this ID: the per-ID models (autoencoder, kNN reference set of EAT) have a small
  training or reference set, and the ID classifiers (MobileNetV2 within slider,
  Wilkinghoff and OS-SCL across types) see an under-represented class. Each model
  keeps the class weighting of its original method (none for MobileNetV2). The
  internal controls of MobileNetV2 show the effect: at the training SNR, 115 of
  the 356 normal test clips of slider `id_00` are predicted `id_02`
  ([`validation.md`](validation.md)). Results for slider `id_00` are reported
  with this caveat.
- Identities flagged by the strict variant also occur in the training sets
  (Amendment 2).
- **Validation against published results: three models of five.** The
  autoencoder (MIMII paper, original split) and MobileNetV2 (DCASE 2022 Task 2,
  pre-registered cells) are validated against published numbers. Wilkinghoff,
  re-implemented from the author's articles only, is validated partially: it
  reproduces the published DCASE 2022 results, but only 6 of the 12
  pre-registered cells agree under the official per-domain definition (10 of 12
  under the definition the author uses, clips of the domain only), and on DCASE
  2023 dev it stays 5 to 6 points below the published target-domain AUC, which is
  unexplained (the elements of the author's code that appear in none of his
  articles, squeeze-and-excitation blocks and L2 weight decay, are not
  implemented). Two are not validated against published numbers:
  **EAT + kNN**, because no per-machine result of a frozen EAT with kNN on
  MIMII-derived data is published, the two published aggregates on DCASE 2020
  Task 2 disagree by 5.7 points for the same checkpoint, and reproducing them would
  need the non-commercial ToyADMOS data; it is validated instead by the numerical
  equivalence of its encoder with the official inference code and by internal
  controls. **OS-SCL**, because its published results (DCASE 2020 Task 2) have no
  dispersion, so no cell can pass the pre-registered dispersion criterion, and
  its training set includes non-commercial data (ToyADMOS, DCASE 2020 additional
  training set); it is validated by unit tests of each component against the
  paper's formulas and by internal controls. See [`validation.md`](validation.md).
- **OS-SCL is a reconstruction, not a reproduction of the authors' system.** Its
  parameter count matches neither possible configuration: 1.49 M with the
  MobileFaceNet of its cited paper (used here, by the order of sources), 1.30 M
  with the configuration of the authors' repository, and 1.38 M reported in the
  OS-SCL paper (Table V). Together with the absence of an external validation,
  this means that our OS-SCL is faithful to the published formulas and
  architecture descriptions, but it cannot be guaranteed to reproduce the exact
  system of the authors. Its results are reported as those of this
  reconstruction.

## Amendments

Amendments are appended here; earlier sections keep their original wording.

**Amendment 1 — 2026-09-29 — clusters removed from the split.** The cluster rule
of "Train/test split" (introduced in commit `119b1a4`) is withdrawn: the split is
drawn per identity, with exact balance, as in the MIMII baseline protocol
(`make_split`; `split.cluster_min_score_p6: null`).
*Why:* the rule assumed that pairs of distinct files with a +6 dB score ≥ 0.5
share recorded machine sound. They do not: such pairs are as frequent with a
score ≤ −0.5 (fan 146 vs 144, pump 141 vs 107, slider 131 vs 103), i.e. they are
sign-symmetric phase coincidences of tonal sounds, whereas shared content can only
correlate positively (analysis in commit `6c14938`, [`data.md`](data.md)).
*Timing:* the change was made before any split was saved and before any model
was trained or evaluated; no result existed. The noise reuse check is kept as a
reported finding: its figures are published with the split
(`metadata/mimii_noise_check.csv`).

**Amendment 2 — 2026-09-29 — fan `id_00` kept; robustness aggregation; strict
variant extended.** Commit: the one adding this note (after `f633b63`).

- *fan `id_00` is kept* (15 machine IDs). A group-level check had flagged its
  anomalous clips: at −6 dB, 81 of 357 identities have a non-match of their row
  with a score ≤ −(median accepted score). Matching decisions are taken row by
  row, so the row decides: for every accepted identity of every group, the
  accepted score exceeds the most negative non-match of its row in absolute value
  (0 exceedances; one-sided binomial test in [`data.md`](data.md)). The group
  flag reflects heavy, sign-symmetric tails of tonal anomalous sounds.
- *Robustness aggregation "fan without `id_00`":* fan and overall means are
  also computed without fan `id_00` and reported next to the main results. It is
  a re-aggregation of the same per-ID results; no model is trained for it
  (`aggregation.robustness`).
- *Strict variant extended (evaluation only):* besides test identities whose
  two −6 dB score terms disagree, it removes test identities whose accepted +6 or
  −6 dB score is below the 99.9th percentile of |score| over the non-matches of
  their group (`p6_below_null_p999`, `m6_below_null_p999` in the identity table).
  In the main split this removes 368 of 5,582 test identities (203 normal, 165
  anomalous), 8 of them for the percentile criterion.
- *Limitation:* identities meeting these criteria are also found in the training
  sets (main split: 527 of 8,778 training identities, 6.0 %: 512 for disagreeing
  terms, 19 below the percentile, with overlap; 2×2 sensitivity split: 1 below
  the percentile at +6 dB). They cannot be neutralized without retraining and are
  not worked around; if one of their −6 dB matches were wrong, a training file
  could hold a recording of another identity. This bounds the residual risk; it
  is stated as a limitation.
- *Timing:* no split had been saved and no model trained or evaluated.
