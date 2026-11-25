"""Download the MIMII archives from Zenodo and verify their MD5 checksums.

The list of archives (file name, size, MD5) is the committed manifest
``mimii_zenodo_3384388.csv``, copied from the Zenodo API for record 3384388 on
2026-09-29. Archives are downloaded with aria2c (resumable, parallel) when it is
installed, otherwise with a simple pure-Python fallback that restarts interrupted
files from scratch. Every archive is then verified independently with hashlib and
a verification report is written next to the archives.

Usage::

    python -m noisemismatch.data.download                   # download + verify all
    python -m noisemismatch.data.download --verify-only     # verify existing files
    python -m noisemismatch.data.download --check-manifest  # compare with live API
    python -m noisemismatch.data.download --only=-6_dB_fan.zip

Archives go to ``<data_root>/downloads`` and logs to ``<data_root>/logs``
(``data_root`` is set in ``configs/paths.yaml``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import subprocess
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from noisemismatch.config import load_paths
from noisemismatch.utils.logs import setup_logging

logger = logging.getLogger(__name__)

ZENODO_RECORD: int = 3384388
API_URL: str = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
MANIFEST_PATH: Path = Path(__file__).with_name(f"mimii_zenodo_{ZENODO_RECORD}.csv")
DOWNLOAD_SUBDIR: str = "downloads"
LOG_SUBDIR: str = "logs"
REPORT_NAME: str = "verification.json"
_CHUNK: int = 8 * 2**20


@dataclass(frozen=True)
class Archive:
    """One archive of the Zenodo record.

    Attributes:
        filename: Archive file name on Zenodo, e.g. ``-6_dB_fan.zip``.
        snr_db: Signal-to-noise ratio of the recordings in the archive.
        machine_type: Machine type of the recordings in the archive.
        size_bytes: Expected file size.
        md5: Expected MD5 digest (lowercase hexadecimal).
        record: Zenodo record holding the archive.
    """

    filename: str
    snr_db: int
    machine_type: str
    size_bytes: int
    md5: str
    record: int = ZENODO_RECORD

    @property
    def url(self) -> str:
        """Direct download URL of the archive (Zenodo API)."""
        return f"https://zenodo.org/api/records/{self.record}/files/{self.filename}/content"


@dataclass(frozen=True)
class Verification:
    """Result of checking one archive on disk.

    Attributes:
        filename: Archive file name.
        ok: True if size and MD5 both match the manifest.
        reason: ``"ok"`` or a short description of the problem.
        size_bytes: Size found on disk, or None if the file is missing.
        md5: MD5 computed on disk, or None if it was not computed.
    """

    filename: str
    ok: bool
    reason: str
    size_bytes: int | None
    md5: str | None


def load_manifest(
    path: Path = MANIFEST_PATH, record: int = ZENODO_RECORD
) -> list[Archive]:
    """Reads the archive manifest.

    Args:
        path: CSV file with columns ``filename, snr_db, machine_type, size_bytes, md5``.
        record: Zenodo record of the archives.

    Returns:
        The archives, in file order.
    """
    with path.open(newline="", encoding="utf-8") as f:
        return [
            Archive(
                filename=row["filename"],
                snr_db=int(row["snr_db"]),
                machine_type=row["machine_type"],
                size_bytes=int(row["size_bytes"]),
                md5=row["md5"].lower(),
                record=record,
            )
            for row in csv.DictReader(f)
        ]


def md5sum(path: Path, chunk_size: int = _CHUNK) -> str:
    """Computes the MD5 digest of a file, reading it in chunks.

    Args:
        path: File to hash.
        chunk_size: Number of bytes read at a time.

    Returns:
        The lowercase hexadecimal digest.
    """
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, archive: Archive) -> Verification:
    """Checks an archive on disk against the manifest.

    The size is checked first; the (slow) MD5 is only computed when the size
    matches and no aria2c control file indicates an unfinished download.

    Args:
        path: Location of the archive on disk.
        archive: Expected properties.

    Returns:
        The verification result.
    """
    if not path.exists():
        return Verification(archive.filename, False, "missing", None, None)
    size = path.stat().st_size
    if size != archive.size_bytes:
        reason = f"size {size} != expected {archive.size_bytes}"
        return Verification(archive.filename, False, reason, size, None)
    if path.with_name(path.name + ".aria2").exists():
        return Verification(archive.filename, False, "incomplete download", size, None)
    digest = md5sum(path)
    if digest != archive.md5:
        reason = f"md5 {digest} != expected {archive.md5}"
        return Verification(archive.filename, False, reason, size, digest)
    return Verification(archive.filename, True, "ok", size, digest)


def _needs_download(path: Path, archive: Archive) -> bool:
    """Tells whether an archive is absent or unfinished (without hashing it)."""
    if not path.exists() or path.with_name(path.name + ".aria2").exists():
        return True
    return path.stat().st_size != archive.size_bytes


def aria2c_input(archives: Sequence[Archive]) -> str:
    """Renders an aria2c input file with output names and MD5 checksums.

    Args:
        archives: Archives to download.

    Returns:
        The content of the input file.
    """
    lines: list[str] = []
    for archive in archives:
        lines += [
            archive.url,
            f"  out={archive.filename}",
            f"  checksum=md5={archive.md5}",
        ]
    return "\n".join(lines) + "\n"


def download_aria2c(
    archives: Sequence[Archive],
    dest_dir: Path,
    log_file: Path,
    jobs: int = 2,
    connections: int = 4,
) -> int:
    """Downloads archives with aria2c (resumable; MD5-checked by aria2c too).

    Args:
        archives: Archives to download.
        dest_dir: Output directory.
        log_file: aria2c log file.
        jobs: Number of archives downloaded concurrently.
        connections: Connections per archive.

    Returns:
        The aria2c exit code (0 on success).
    """
    input_file = dest_dir / "aria2c_input.txt"
    input_file.write_text(aria2c_input(archives), encoding="utf-8")
    cmd = [
        "aria2c",
        f"--input-file={input_file}",
        f"--dir={dest_dir}",
        "--continue=true",
        f"--max-concurrent-downloads={jobs}",
        f"--split={connections}",
        f"--max-connection-per-server={connections}",
        "--min-split-size=64M",
        "--file-allocation=none",
        "--check-integrity=true",
        "--auto-file-renaming=false",
        "--max-tries=20",
        "--retry-wait=60",
        "--timeout=120",
        "--summary-interval=300",
        f"--log={log_file}",
        "--log-level=notice",
        "--console-log-level=warn",
    ]
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def download_urllib(url: str, dest: Path, timeout: float = 120.0) -> None:
    """Downloads one file with the standard library (fallback when aria2c is absent).

    The file is written to ``<dest>.part`` and renamed on completion, so an
    interrupted download never leaves a truncated file under the final name. An
    interrupted download restarts from scratch.

    Args:
        url: Source URL.
        dest: Final file path.
        timeout: Socket timeout in seconds.
    """
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response, part.open("wb") as f:
        shutil.copyfileobj(response, f, length=_CHUNK)
    part.replace(dest)


def check_manifest_against_api(archives: Sequence[Archive]) -> bool:
    """Compares the committed manifest with the live Zenodo record.

    Args:
        archives: Archives from the committed manifest.

    Returns:
        True if file names, sizes and MD5 digests are identical.
    """
    with urllib.request.urlopen(API_URL, timeout=60) as response:
        record = json.load(response)
    live = {
        (f["key"], int(f["size"]), f["checksum"].removeprefix("md5:").lower())
        for f in record["files"]
    }
    committed = {(a.filename, a.size_bytes, a.md5) for a in archives}
    for item in sorted(live - committed):
        logger.error("Only on Zenodo: %s", item)
    for item in sorted(committed - live):
        logger.error("Only in manifest: %s", item)
    return live == committed


def main(argv: list[str] | None = None) -> int:
    """Downloads and verifies the MIMII archives.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 if every selected archive is verified.
    """
    parser = argparse.ArgumentParser(description="Download and verify MIMII.")
    parser.add_argument(
        "--only",
        action="append",
        metavar="FILENAME",
        help="Restrict to this archive (repeatable). Use --only=-6_dB_fan.zip "
        "for names starting with '-'.",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Do not download, only verify."
    )
    parser.add_argument(
        "--check-manifest",
        action="store_true",
        help="Only compare the committed manifest with the live Zenodo API.",
    )
    parser.add_argument("--jobs", type=int, default=2, help="Concurrent archives.")
    parser.add_argument(
        "--connections", type=int, default=4, help="Connections per archive."
    )
    args = parser.parse_args(argv)

    paths = load_paths()
    dest_dir = paths.data_root / DOWNLOAD_SUBDIR
    log_dir = paths.data_root / LOG_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir / "download.log")

    archives = load_manifest()
    if args.only:
        known = {a.filename for a in archives}
        unknown = sorted(set(args.only) - known)
        if unknown:
            parser.error(f"unknown archive(s): {unknown}")
        archives = [a for a in archives if a.filename in args.only]

    if args.check_manifest:
        same = check_manifest_against_api(archives)
        logger.info(
            "Manifest %s the live Zenodo record.", "matches" if same else "DIFFERS FROM"
        )
        return 0 if same else 1

    if not args.verify_only:
        todo = [a for a in archives if _needs_download(dest_dir / a.filename, a)]
        total_gb = sum(a.size_bytes for a in todo) / 1e9
        logger.info(
            "%d archive(s) to download (%.2f GB) into %s", len(todo), total_gb, dest_dir
        )
        if todo and shutil.which("aria2c"):
            code = download_aria2c(
                todo, dest_dir, log_dir / "aria2c.log", args.jobs, args.connections
            )
            if code != 0:
                logger.error("aria2c exited with code %d (see aria2c.log).", code)
        elif todo:
            logger.warning("aria2c not found: using the slower urllib fallback.")
            for archive in todo:
                logger.info("Downloading %s", archive.filename)
                download_urllib(archive.url, dest_dir / archive.filename)

    results: list[Verification] = []
    for archive in archives:
        logger.info("Verifying %s", archive.filename)
        result = verify_archive(dest_dir / archive.filename, archive)
        log = logger.info if result.ok else logger.error
        log("%s: %s", archive.filename, result.reason)
        results.append(result)

    report = {
        "zenodo_record": ZENODO_RECORD,
        "manifest": MANIFEST_PATH.name,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": [asdict(r) for r in results],
    }
    (dest_dir / REPORT_NAME).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    n_ok = sum(r.ok for r in results)
    logger.info("%d/%d archive(s) verified.", n_ok, len(results))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
