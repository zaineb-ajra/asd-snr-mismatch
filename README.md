# noisemismatch

Code, data tables and results of the paper

> Zaineb Ajra and Andrea Daou. *Détection non supervisée de sons anormaux : effet
> d'un décalage de rapport signal sur bruit entre entraînement et test.*
> Extraction et Gestion des Connaissances (EGC 2027), RNTI, to appear.

Unsupervised anomalous sound detection (ASD) aims to detect machine faults from
their sound. In a plant, background noise varies, so a detector may be used at a
signal-to-noise ratio (SNR) different from the one it was trained at. This study
measures the consequences of such a mismatch for five detectors. The MIMII dataset
provides the same recordings at three SNRs (−6, 0 and +6 dB): each model is
trained at one of them (or at all three, multi-condition training) and evaluated
at all three.

| Family | Model | Original code |
|---|---|---|
| Reconstruction | Autoencoder (MIMII baseline) | [MIMII-hitachi/mimii_baseline](https://github.com/MIMII-hitachi/mimii_baseline) |
| Machine-ID classifier | MobileNetV2 (DCASE 2022 baseline) | [Kota-Dohi/dcase2022_task2_baseline_mobile_net_v2](https://github.com/Kota-Dohi/dcase2022_task2_baseline_mobile_net_v2) |
| Angular margin | Wilkinghoff, SSL / FeatEx | [wilkinghoff/ssl4asd](https://github.com/wilkinghoff/ssl4asd) (re-implemented from the paper, no code used) |
| Contrastive | OS-SCL | [huangswt/OS-SCL](https://github.com/huangswt/OS-SCL) |
| Frozen pretrained encoder + kNN | EAT + kNN | [cwx-worst-one/EAT](https://github.com/cwx-worst-one/EAT) |

## Citation

```bibtex
@inproceedings{ajra2027egc,
  author    = {Ajra, Zaineb and Daou, Andrea},
  title     = {D{\'e}tection non supervis{\'e}e de sons anormaux : effet d'un
               d{\'e}calage de rapport signal sur bruit entre entra{\^i}nement et test},
  booktitle = {Extraction et Gestion des Connaissances (EGC 2027)},
  series    = {Revue des Nouvelles Technologies de l'Information},
  note      = {to appear},
  publisher = {{\'E}ditions RNTI},
  year      = {2027},
}
```

## What this repository contains

| Item | Location |
|---|---|
| Code of the five detectors | [`noisemismatch/models/`](noisemismatch/models) (EAT encoder: [`noisemismatch/features/`](noisemismatch/features)) |
| Content-based pairing of MIMII files across SNRs | [`noisemismatch/data/matching.py`](noisemismatch/data/matching.py), [`pairing.py`](noisemismatch/data/pairing.py), [`identity.py`](noisemismatch/data/identity.py) |
| Splits: main, sensitivity, strict variant | [`noisemismatch/data/split.py`](noisemismatch/data/split.py), [`build_split.py`](noisemismatch/data/build_split.py); strict variant in [`configs/protocol.yaml`](configs/protocol.yaml) (`evaluation.strict_variant`) |
| Training, single SNR and multi-condition | [`noisemismatch/run.py`](noisemismatch/run.py), [`noisemismatch/data/dataset.py`](noisemismatch/data/dataset.py), [`scripts/run_study.py`](scripts/run_study.py) |
| Evaluation: AUC, pAUC, ΔAUC, paired bootstrap | [`noisemismatch/eval/`](noisemismatch/eval), [`scripts/study_analysis.py`](scripts/study_analysis.py) |
| Configuration of each model and of the protocol | [`configs/models/`](configs/models), [`configs/protocol.yaml`](configs/protocol.yaml) |
| Identity table (pairs the MIMII files across SNRs) | [`metadata/mimii_identity.csv`](metadata/mimii_identity.csv) |
| DCASE 2020 ↔ MIMII −6 dB correspondence, with CRC-32 | [`metadata/dcase2020_mimii_mapping.csv`](metadata/dcase2020_mimii_mapping.csv), [`metadata/dcase2020_other_snrs_check.csv`](metadata/dcase2020_other_snrs_check.csv) |
| Results (CSV): AUC and pAUC per model, condition, seed, test SNR and machine; 95 % bootstrap intervals | [`results/aggregated/`](results/aggregated) |
| Table 1 and Figure 1 of the paper, and the script that generates them | [`results/paper/`](results/paper), [`scripts/paper_figures.py`](scripts/paper_figures.py) |
| Protocol, data description, deviations from the original methods, validation of the implementations | [`docs/`](docs) |

Not included: MIMII audio (download it from Zenodo, below), trained models and
checkpoints, per-run outputs (per-clip scores, logs).

## Regenerate Table 1 and Figure 1 from the provided CSVs (no GPU, no data)

Needs Python ≥ 3.10; takes a few seconds on a CPU.

```bash
git clone https://github.com/zaineb-ajra/asd-snr-mismatch.git && cd asd-snr-mismatch
python3 -m venv .venv-paper
source .venv-paper/bin/activate
pip install -r requirements-paper.txt
pip install -e . --no-deps
python scripts/paper_figures.py
git status --short results/paper   # empty output: both files are regenerated identically
```

`scripts/paper_figures.py` reads `results/aggregated/aggregated.csv` and
`results/aggregated/bootstrap_intervals_main.csv` and writes
`results/paper/tab_auc.tex` (Table 1) and `results/paper/fig_pertes.pdf`
(Figure 1). The table and figure are in French, as in the paper.

## Full reproduction

### Requirements

- Linux with an NVIDIA GPU and a driver supporting CUDA 12.8 (driver ≥ 570).
  The study ran on two RTX 5000 Ada GPUs.
- Python ≥ 3.10 (developed and tested with Python 3.12.7).
- Disk: about 100 GB for the MIMII archives (read directly from the ZIP files,
  no extraction), about 15 GB for the channel-0 audio cache, plus the feature
  cache and run outputs.
- Compute: about 86 GPU-hours for the 90 study runs
  ([`docs/compute_budget.md`](docs/compute_budget.md)); the content-based
  matching needs about 47 GB of RAM.

### Installation

Exact versions are pinned in `requirements.txt` (runtime, including PyTorch
2.11.0 built for CUDA 12.8) and `requirements-dev.txt` (lint and test tools);
the top-level dependencies are in the `*.in` files.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e . --no-deps
python -m noisemismatch.envcheck   # versions, storage paths, GPU reproducibility check
```

With [uv](https://docs.astral.sh/uv/), replace the `pip install -r` line by
`uv pip install -r requirements.txt -r requirements-dev.txt --index-strategy unsafe-best-match`
(`requirements.txt` lists PyPI and the PyTorch CUDA 12.8 index).

### Storage paths

No path is hard-coded. Data, feature cache and run outputs go to the locations
of [`configs/paths.yaml`](configs/paths.yaml) (by default under `work/` in the
repository). To use other disks, create `configs/paths.local.yaml` (git-ignored)
with the keys to override, or point `NOISEMISMATCH_PATHS` to such a file:

```yaml
data_root: /big/disk/mimii
cache_root: /big/disk/cache
output_root: /big/disk/outputs
```

### MIMII dataset

MIMII (Purohit et al., 2019) is on Zenodo:
[record 3384388](https://zenodo.org/records/3384388), DOI
[10.5281/zenodo.3384388](https://doi.org/10.5281/zenodo.3384388), license
CC BY-SA 4.0. Its 12 archives (`-6_dB_fan.zip` to `6_dB_valve.zip`, about
100 GB) are downloaded and checked against their MD5 by:

```bash
python -m noisemismatch.data.download --check-manifest   # the Zenodo record still matches the manifest
python -m noisemismatch.data.download                    # resumable download + MD5 check
python -m noisemismatch.data.prepare                     # file index and inventory
```

The archives can also be downloaded by hand into `<data_root>/downloads/`; the
expected file names and MD5 digests are in
[`noisemismatch/data/mimii_zenodo_3384388.csv`](noisemismatch/data/mimii_zenodo_3384388.csv).
Details and checks: [`docs/data.md`](docs/data.md).

### Steps

The versioned files of `metadata/` and `results/` are the outputs of these
steps; rerunning a step regenerates the corresponding files.

```bash
# 1. Content-based pairing of the files across SNRs (GPU, ~40 min, ~47 GB RAM),
#    then the clip identities -> metadata/mimii_identity.csv
CUDA_VISIBLE_DEVICES=0 python -m noisemismatch.data.matching
python -m noisemismatch.data.identity
python scripts/make_data_tables.py                  # docs/data_tables.md

# 2. Splits. The committed splits (metadata/mimii_split.csv, main;
#    metadata/mimii_split_sensitivity.csv, sensitivity) are never overwritten:
#    this command rebuilds them from the identity table and the seed and checks
#    them (tests/test_committed_split.py does the same). The strict variant is a
#    filter applied at evaluation (configs/protocol.yaml, evaluation.strict_variant).
python -m noisemismatch.data.build_split

# 3. Inputs: channel-0 audio cache (~15 GB) and the EAT-base checkpoint
python -m noisemismatch.data.audio_cache
python -m noisemismatch.data.eat_checkpoint

# 4. Training and scoring: 90 runs (5 models x {main split: -6, 0, +6 dB, multi;
#    sensitivity split: 0, +6 dB} x 3 seeds), from a clean git work tree.
#    Stage A (autoencoder, MobileNetV2, EAT + kNN, Wilkinghoff), its checklist,
#    then stage B (OS-SCL) if every check passes.
python scripts/run_study.py --gpus 0 1
python scripts/check_runs.py --stage A
python scripts/check_runs.py --stage B
#    A single run: python -m noisemismatch.run --model <name> --train-snr {-6,0,6,multi} --seed <k> [--split sensitivity]

# 5. Evaluation from the per-clip scores (CPU, ~5 min on 16 cores):
#    per-machine AUC/pAUC with bootstrap intervals, EAT scoring variants,
#    paired bootstrap on the aggregated values, then the study report
python scripts/study_analysis.py per-id --workers 16
python scripts/study_analysis.py eat-variants
python scripts/study_analysis.py bootstrap --workers 15
python scripts/study_report.py                      # results/study_report.md, results/tables, results/figures

# 6. Table 1 and Figure 1 of the paper
python scripts/paper_figures.py                     # results/paper/
```

Model names for `--model`: `ae`, `mobilenetv2`, `wilkinghoff`, `osscl`,
`eat_knn`. The run outputs (configuration, seed, git commit, environment,
duration, per-clip scores) are written under `<output_root>/runs/`; see
[`docs/framework.md`](docs/framework.md). The checks applied to the runs are
listed in [`docs/phase6_checklist.md`](docs/phase6_checklist.md).

DCASE 2020 correspondence (CRC-32 of the central directory of the ZIP archives of
Zenodo [3678171](https://zenodo.org/records/3678171), read by HTTP range
requests; no DCASE audio is downloaded):

```bash
python -m noisemismatch.data.dcase2020                 # metadata/dcase2020_mimii_mapping.csv
python -m noisemismatch.data.dcase2020 --other-snrs    # metadata/dcase2020_other_snrs_check.csv
```

Validation of the implementations against published results (not part of the
study): `python -m noisemismatch.verification --model <name>` and the modules
`noisemismatch.verification_*`, `noisemismatch.controls`,
`noisemismatch.dcase2020_check`; criteria and results in
[`docs/validation.md`](docs/validation.md) and
[`results/verification/`](results/verification).

### Tests

```bash
scripts/ci.sh    # ruff, black --check, pytest
```

Tests that need a GPU or the MIMII data on disk are skipped when these are absent.

## Correspondence with the paper

| Paper | Files and scripts |
|---|---|
| §2 Related work: the MIMII part of the DCASE 2020 development set is channel 0 of MIMII at −6 dB (CRC-32) | `metadata/dcase2020_mimii_mapping.csv`, `metadata/dcase2020_other_snrs_check.csv`, `noisemismatch/data/dcase2020.py`, `docs/data.md` |
| §3 Data and recording identity: content-based pairing, 15,180 identities, exclusion of valve id_00 | `noisemismatch/data/matching.py`, `pairing.py`, `identity.py`, `metadata/mimii_identity.csv`, `metadata/mimii_pairing_content.csv`, `metadata/mimii_inventory.csv`, `docs/data.md`, `docs/data_tables.md` |
| §3 Split: main split (14,360 identities: 8,778 training, 5,582 test), sensitivity split, strict variant (368 test identities removed) | `noisemismatch/data/split.py`, `build_split.py`, `metadata/mimii_split.csv`, `metadata/mimii_split_sensitivity.csv`, `metadata/mimii_split_summary.csv`, `configs/protocol.yaml` |
| §3 Models and validation of the implementations | `noisemismatch/models/`, `noisemismatch/features/`, `configs/models/*.yaml`, `docs/deviations.md`, `docs/validation.md`, `results/verification/`, `results/dcase2020_check.md`, `metadata/published/` |
| §3 Conditions and measures: train/test matrix, multi-condition training, 90 runs, AUC, pAUC, ΔAUC, paired bootstrap | `noisemismatch/run.py`, `scripts/run_study.py`, `scripts/check_runs.py`, `noisemismatch/eval/`, `scripts/study_analysis.py`, `docs/protocol.md`, `results/phase6/` |
| §4 Results, Table 1 | `results/paper/tab_auc.tex` ← `scripts/paper_figures.py` ← `results/aggregated/aggregated.csv`, `bootstrap_intervals_main.csv` |
| §4 Results, Figure 1 (AUC points lost at 12 dB of mismatch and by the multi-condition model) | `results/paper/fig_pertes.pdf` ← `scripts/paper_figures.py` ← `results/aggregated/bootstrap_intervals_main.csv` |
| §4 Train/test matrix, direction of the mismatch, multi-condition training | `results/aggregated/bootstrap_intervals_main.csv`, `delta_main_*.csv`, `summary_main_*.csv`, `results/study_report.md` |
| §5 Fragility and robustness checks: error ratios, strict variant, without fan id_00, sensitivity split, pAUC | `results/aggregated/error_ratio_main_auc.csv`, `*_strict_*.csv`, `*_fan_without_id_00_*.csv`, `*_sensitivity_*.csv`, `bootstrap_intervals_*.csv`, `per_id_*.csv`, `results/study_report.md` |
| §6 Discussion and conclusion | `results/aggregated/bootstrap_intervals_main.csv`, `results/study_report.md`, `docs/validation.md`, `results/verification/` |

## Repository layout

```
noisemismatch/     Python package
  data/            MIMII download and MD5 check, inventory, content pairing, identities, splits
  features/        input features (log-Mel, EAT encoder) and on-disk cache
  models/          one module per detector, common fit/score interface
  eval/            metrics, aggregation, ΔAUC, paired bootstrap
  utils/           reproducibility, provenance, logging
configs/           protocol, storage paths, one file per model
metadata/          dataset-derived tables: inventory, identities, splits, DCASE 2020 mapping
results/           aggregated results (CSV), Table 1 and Figure 1, study report, validation
scripts/           study driver, checks, analysis, report, table and figure generation
tests/             pytest suite
docs/              protocol, data, deviations from the original methods, validation
papers/            list of the reference papers (PDFs not redistributed)
```

## License

The code is released under the **MIT License** ([`LICENSE`](LICENSE)). Code
ported or vendored from third parties keeps its own permissive license,
reproduced in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`LICENSES/`](LICENSES).

**Data-derived files.** The files of [`metadata/`](metadata) derived from MIMII
(identity tables, splits, checks, DCASE 2020 mapping) are distributed under
**CC BY-SA 4.0**, not under the MIT License; see
[`metadata/README.md`](metadata/README.md). No audio is redistributed.
`metadata/dcase2020_mimii_mapping.csv` holds only file names and CRC-32
checksums of the DCASE 2020 Task 2 development set (CC BY-NC-SA 4.0), from which
no audio file can be reconstructed.

### Third-party code

Licenses checked on 2026-09-30; notices and license texts in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`LICENSES/`](LICENSES).

| Component | Source | License | Use in this repository |
|---|---|---|---|
| MIMII baseline (autoencoder) | [MIMII-hitachi/mimii_baseline](https://github.com/MIMII-hitachi/mimii_baseline), commit `7086d21` | none declared | Re-implemented from the paper and the baseline's hyperparameter values; no code copied |
| DCASE 2022 MobileNetV2 baseline | [Kota-Dohi/dcase2022_task2_baseline_mobile_net_v2](https://github.com/Kota-Dohi/dcase2022_task2_baseline_mobile_net_v2), commit `aaf3932` | MIT, Copyright (c) 2022 Kota-Dohi | Ported to PyTorch |
| Keras MobileNetV2 | `keras/applications/mobilenet_v2.py`, Keras 2.8.0 | Apache-2.0, Copyright 2018 The TensorFlow Authors | Network translated to PyTorch |
| EAT | [cwx-worst-one/EAT](https://github.com/cwx-worst-one/EAT); Hugging Face `worstchan/EAT-base_epoch30_pretrain`, revision `6eb9593` | MIT, Copyright (c) 2024 Wenxi Chen | Encoder partly vendored; model-card preprocessing |
| data2vec 2.0 (fairseq) | [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq), through the EAT code | MIT, Copyright (c) Facebook, Inc. and its affiliates | Attention and block modules of the EAT encoder |
| EAT-base weights | Hugging Face `worstchan/EAT-base_epoch30_pretrain`, revision `6eb9593` | MIT (model card) | Downloaded by `python -m noisemismatch.data.eat_checkpoint`; not redistributed |

Deliberately not included: the parts of the upstream EAT code that come from
AudioMAE / MAE (CC BY-NC 4.0; the patch projection is written from its function
and the positional table is read from the checkpoint), and any code of
[wilkinghoff/ssl4asd](https://github.com/wilkinghoff/ssl4asd) (GPL-3.0; the
Wilkinghoff model is re-implemented from the paper, and every hyperparameter
value taken from that repository is listed in `docs/deviations.md`).
