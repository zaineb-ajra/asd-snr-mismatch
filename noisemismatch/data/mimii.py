"""Access to the MIMII archives: file index, WAV headers and audio reading.

The archives are read in place (no extraction): each Zenodo archive holds one
machine type at one SNR, with members named
``<machine_type>/<machine_id>/<label>/<clip>.wav`` (e.g.
``pump/id_00/abnormal/00000000.wav``); the SNR is only encoded in the archive name.
This layout was checked on the downloaded archives (see docs/data.md); any member
that does not follow it raises an error.
"""

from __future__ import annotations

import re
import struct
import zipfile
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from noisemismatch.data.download import Archive

MEMBER_PATTERN: re.Pattern[str] = re.compile(
    r"(?P<machine_type>[a-z]+)/(?P<machine_id>id_\d{2})/"
    r"(?P<label>normal|abnormal)/(?P<clip>\d{8})\.wav"
)
WAVE_FORMAT_PCM: int = 0x0001
WAVE_FORMAT_EXTENSIBLE: int = 0xFFFE
_HEADER_BYTES: int = 1024


@dataclass(frozen=True)
class WavHeader:
    """Properties read from a RIFF/WAVE header.

    Attributes:
        format_tag: Sample format (1 = PCM; for WAVE_FORMAT_EXTENSIBLE files, the
            sub-format of the extension).
        extensible: True if the file uses WAVE_FORMAT_EXTENSIBLE.
        channels: Number of channels.
        sample_rate: Sampling rate in Hz.
        bits_per_sample: Bits per sample.
        data_offset: Byte offset of the sample data.
        data_bytes: Size of the sample data in bytes.
    """

    format_tag: int
    extensible: bool
    channels: int
    sample_rate: int
    bits_per_sample: int
    data_offset: int
    data_bytes: int

    @property
    def frames(self) -> int:
        """Number of samples per channel."""
        return self.data_bytes // (self.channels * self.bits_per_sample // 8)


def parse_wav_header(head: bytes) -> WavHeader:
    """Parses the RIFF chunks of a WAV file up to the start of the sample data.

    Args:
        head: The first bytes of the file (must include the ``data`` chunk header).

    Returns:
        The parsed header.

    Raises:
        ValueError: If the bytes are not a RIFF/WAVE header with ``fmt `` and
            ``data`` chunks.
    """
    if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    offset = 12
    fmt: tuple[int, bool, int, int, int] | None = None
    while offset + 8 <= len(head):
        chunk_id = head[offset : offset + 4]
        (size,) = struct.unpack_from("<I", head, offset + 4)
        body = offset + 8
        if chunk_id == b"fmt ":
            tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", head, body)
            extensible = tag == WAVE_FORMAT_EXTENSIBLE
            if extensible:
                # WAVEFORMATEXTENSIBLE: cbSize, validBits, channelMask, then the
                # sub-format GUID whose first two bytes are the format tag.
                (tag,) = struct.unpack_from("<H", head, body + 24)
            fmt = (tag, extensible, channels, rate, bits)
        elif chunk_id == b"data":
            if fmt is None:
                raise ValueError("'data' chunk before 'fmt ' chunk")
            return WavHeader(*fmt, data_offset=body, data_bytes=size)
        offset = body + size + (size % 2)  # chunks are word-aligned
    raise ValueError("no 'data' chunk in the header bytes")


def parse_member_name(name: str) -> dict[str, str]:
    """Splits an archive member path into its identity fields.

    Args:
        name: Member path, e.g. ``pump/id_00/abnormal/00000000.wav``.

    Returns:
        Mapping with keys ``machine_type``, ``machine_id``, ``label``, ``clip``.

    Raises:
        ValueError: If the path does not follow the MIMII layout.
    """
    match = MEMBER_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"unexpected archive member: {name!r}")
    return match.groupdict()


def index_archive(path: Path, archive: Archive) -> pd.DataFrame:
    """Lists the WAV files of one archive with their identity and header fields.

    Only the first bytes of each member are decompressed, so indexing is fast.

    Args:
        path: Archive on disk.
        archive: Manifest entry of the archive (gives the SNR and machine type).

    Returns:
        One row per WAV file with columns ``snr_db``, identity columns,
        ``archive``, ``member``, ``file_bytes``, ``format_tag``, ``extensible``,
        ``channels``, ``sample_rate``, ``bits_per_sample``, ``data_offset``,
        ``frames`` and ``duration_s``.

    Raises:
        ValueError: If a member does not follow the MIMII layout or belongs to
            another machine type than the archive.
    """
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            fields = parse_member_name(info.filename)
            if fields["machine_type"] != archive.machine_type:
                raise ValueError(f"{path.name}: unexpected member {info.filename!r}")
            with zf.open(info) as f:
                header = parse_wav_header(f.read(_HEADER_BYTES))
            rows.append(
                {
                    "snr_db": archive.snr_db,
                    **fields,
                    "archive": archive.filename,
                    "member": info.filename,
                    "file_bytes": info.file_size,
                    "format_tag": header.format_tag,
                    "extensible": header.extensible,
                    "channels": header.channels,
                    "sample_rate": header.sample_rate,
                    "bits_per_sample": header.bits_per_sample,
                    "data_offset": header.data_offset,
                    "frames": header.frames,
                    "duration_s": header.frames / header.sample_rate,
                }
            )
    return pd.DataFrame(rows)


def build_file_index(
    downloads_dir: Path, archives: Sequence[Archive], workers: int = 1
) -> pd.DataFrame:
    """Indexes several archives.

    Args:
        downloads_dir: Directory holding the archives.
        archives: Manifest entries to index.
        workers: Number of archives indexed in parallel (processes).

    Returns:
        The concatenated file index, sorted by SNR and identity.
    """
    paths = [downloads_dir / a.filename for a in archives]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(index_archive, paths, archives))
    else:
        frames = [index_archive(p, a) for p, a in zip(paths, archives, strict=True)]
    index = pd.concat(frames, ignore_index=True)
    order = ["snr_db", "machine_type", "machine_id", "label", "clip"]
    return index.sort_values(order, ignore_index=True)


def read_member(
    path: Path, member: str, channel: int | None = None
) -> tuple[np.ndarray, int]:
    """Reads a WAV file stored in an archive.

    Args:
        path: Archive on disk.
        member: Member path inside the archive.
        channel: If given, return only this channel.

    Returns:
        ``(samples, sample_rate)``; samples are float32 in [-1, 1], shaped
        ``(frames, channels)``, or ``(frames,)`` when ``channel`` is given.
    """
    with zipfile.ZipFile(path) as zf, zf.open(member) as f:
        samples, sample_rate = sf.read(f, dtype="float32", always_2d=True)
    if channel is not None:
        samples = samples[:, channel]
    return samples, sample_rate
