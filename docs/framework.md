# Common framework

Every model goes through the same data, features, training loop contract and
evaluation. This page describes the pieces a reviewer needs to follow a run.

## Audio

`python -m noisemismatch.data.audio_cache` copies channel 0 (microphone 1, as the
MIMII baseline) of every file used by the main or the sensitivity split into one
int16 array per (machine type, SNR) under `<cache_root>/audio/ch0-v1/`, with an
index CSV and a manifest (channel, format, source archives and their MD5, git
commit). Samples are copied bit for bit; a test compares a random sample with the
archives. Models receive float32 waveforms, `int16 / 32768`. Built on
2026-09-29 (commit `44baa37`): 47,374 files (16,507 at 0 and +6 dB, 14,360 at
−6 dB), 15 GB, 25 minutes; the comparison with the archives passes on a random
sample of 24 files.

## Features and their cache

A feature extractor (`noisemismatch.features.base.FeatureExtractor`) declares
every parameter that affects its output (e.g. `LogMel`: FFT size, hop, window,
padding, mel scale, power, log offset). Features are computed once per
(machine type, SNR) and stored under
`<cache_root>/features/<name>-<key>/`, where `key` is the SHA-256 of the extractor
name, version and full parameter set, the cache format version and the audio cache
fingerprint. Changing any parameter selects another directory, so a stale cache is
never read; the configuration is also stored there and checked on every open.

## Clips

`noisemismatch.data.dataset.build_clip_table` turns a split into a clip table: one
row per (clip identity, SNR) with the file that holds the recording at that SNR
(content-based identity, `metadata/mimii_identity.csv`), the partition, the label
(`y` = 1 for anomalous) and the strict-variant flag. Rows are sorted by identity
then SNR, so runs of one seed at different SNRs see their clips in the same order.
A `ClipSet` wraps rows of that table and gives `waveforms()` and
`features(extractor)`.

## Models

A model subclasses `noisemismatch.models.base.AnomalyDetector`, declares its
`training_scope` (`machine_id`, `machine_type` or `all`, see the protocol) and
implements `fit(train_set)` and `score(test_set) -> np.ndarray` (higher = more
anomalous). It is registered with `@register` and configured by
`configs/models/<name>.yaml`. The step budget of the multi-condition model is
given by `training_steps` (single-SNR example count) and consumed with
`batch_indices`.

## Runs

```bash
python -m noisemismatch.run --model <name> --train-snr {-6,0,6,multi} --seed <k> [--split sensitivity]
```

The runner seeds every generator, refuses uncommitted changes unless
`--allow-dirty` (recorded), creates one model per group of the training scope
(re-seeding before each group, so a group's result does not depend on the others),
fits it on the group's training clips of the condition and scores its test clips at
every SNR of the split. Outputs in `<output_root>/runs/<split>/<model>/<condition>/seed<k>/`:
`scores.csv` (one row per test clip and SNR, with identity, file, label,
strict flag, run, seed, commit), `result.json` (configuration, protocol, seed,
commit and state, environment, duration, AUC/pAUC per machine ID and SNR),
`run.log` and `DONE`, written last. Finished runs are skipped.

## Evaluation

`noisemismatch.eval.evaluate` computes everything from `scores.csv`: per-ID AUC
and pAUC, bootstrap confidence intervals, the strict variant and the
aggregations of the protocol, including "fan without `id_00`". No model is
rerun for them.
