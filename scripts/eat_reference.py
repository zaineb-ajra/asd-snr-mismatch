"""Reference EAT embeddings through the official inference path (fidelity check).

Runs in a separate environment with ``transformers`` (not in the study
environment). Loads the pinned local snapshot of worstchan/EAT-base_epoch30_pretrain
with ``AutoModel.from_pretrained(<snapshot>, trust_remote_code=True)``, local
files only, and follows the preprocessing of its model card, one clip per forward
pass. Pooling is re-implemented here with NumPy, independently of the study code.

The preprocessing between the ``model card code`` markers is copied from the model
card of worstchan/EAT-base_epoch30_pretrain (revision 6eb9593; MIT License,
Copyright (c) 2024 Wenxi Chen; terms in THIRD_PARTY_NOTICES.md). The upstream
model code is executed at run time from the local snapshot and is not part of
this repository.

Usage::

    <reference venv>/bin/python scripts/eat_reference.py SNAPSHOT_DIR WAV_LIST OUT_NPZ
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModel
from transformers.utils import HF_MODULES_CACHE

TARGET_LENGTH = 1024
NORM_MEAN = -4.268
NORM_STD = 4.569


def main() -> int:
    """Writes ``fbank``, ``time_mean`` and ``cls`` arrays for the listed WAV files."""
    snapshot, wav_list, out = sys.argv[1:4]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # transformers 4.51.3 copies into its module cache only the files imported by
    # modeling_eat.py, not model_core.py (imported by eat_model.py): the snapshot's
    # Python files are copied there unchanged beforehand.
    modules = Path(HF_MODULES_CACHE) / "transformers_modules" / Path(snapshot).name
    modules.mkdir(parents=True, exist_ok=True)
    for source in Path(snapshot).glob("*.py"):
        shutil.copyfile(source, modules / source.name)
    model = AutoModel.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True
    )
    model = model.eval().cuda()
    fbanks, time_means, clss = [], [], []
    with open(wav_list, encoding="utf-8") as f:
        paths = f.read().split()
    for path in paths:
        # --- model card code (revision 6eb9593) ---
        wav, sr = sf.read(path)
        waveform = torch.tensor(wav).float().cuda()
        assert sr == 16000
        waveform = waveform - waveform.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0),
            htk_compat=True,
            sample_frequency=16000,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        ).unsqueeze(0)
        n_frames = mel.shape[1]
        if n_frames < TARGET_LENGTH:
            mel = torch.nn.ZeroPad2d((0, 0, 0, TARGET_LENGTH - n_frames))(mel)
        else:
            mel = mel[:, :TARGET_LENGTH, :]
        mel = (mel - NORM_MEAN) / (NORM_STD * 2)
        mel = mel.unsqueeze(0).cuda()
        with torch.no_grad():
            feat = model.extract_features(mel)
        feat = feat.squeeze(0).cpu().numpy()
        # --- end of model card code ---
        tokens = feat[1:].reshape(-1, 8, feat.shape[1])  # (time, freq, dim)
        fbanks.append(mel[0, 0].cpu().numpy())
        time_means.append(tokens.mean(axis=0).reshape(-1))
        clss.append(feat[0])
    np.savez(
        out,
        fbank=np.stack(fbanks),
        time_mean=np.stack(time_means),
        cls=np.stack(clss),
        transformers_version=__import__("transformers").__version__,
        torch_version=torch.__version__,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
