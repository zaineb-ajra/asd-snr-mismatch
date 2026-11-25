"""Read-only access to a remote ZIP archive through HTTP range requests.

Only the bytes needed are downloaded: the central directory to list members and
their CRC-32, or single members. Used to check the relation between MIMII and the
DCASE 2020 Task 2 data without downloading the latter.
"""

from __future__ import annotations

import io
import urllib.request


class HttpRangeFile(io.RawIOBase):
    """Seekable, read-only file over HTTP range requests.

    Args:
        url: URL of a server that honours ``Range`` headers.
        timeout: Socket timeout in seconds.
    """

    def __init__(self, url: str, timeout: float = 120.0) -> None:
        super().__init__()
        self.url = url
        self.timeout = timeout
        self.position = 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            self.size = int(response.headers["Content-Length"])
        self.downloaded = 0

    def readable(self) -> bool:
        """The file is readable."""
        return True

    def seekable(self) -> bool:
        """The file is seekable."""
        return True

    def tell(self) -> int:
        """Current position."""
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Moves the position (no request is sent)."""
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        else:
            self.position = self.size + offset
        return self.position

    def readinto(self, buffer: bytearray | memoryview) -> int:
        """Reads up to ``len(buffer)`` bytes with one range request."""
        if self.position >= self.size or len(buffer) == 0:
            return 0
        end = min(self.position + len(buffer), self.size) - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if response.status != 206:
                raise OSError(f"{self.url}: range requests not supported")
            data = response.read()
        buffer[: len(data)] = data
        self.position += len(data)
        self.downloaded += len(data)
        return len(data)


def open_remote_zip(url: str) -> tuple[io.BufferedReader, HttpRangeFile]:
    """Buffered reader over a remote file, for :class:`zipfile.ZipFile`.

    Args:
        url: URL of the archive.

    Returns:
        ``(reader, raw)``; ``raw.downloaded`` counts the bytes transferred.
    """
    raw = HttpRangeFile(url)
    return io.BufferedReader(raw, buffer_size=1 << 16), raw
