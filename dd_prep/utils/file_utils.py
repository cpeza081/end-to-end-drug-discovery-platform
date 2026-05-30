"""
utils/file_utils.py — Small file-handling helpers used by multiple steps.

Kept as pure functions (no classes, no state) so they are trivially
testable and importable without any side effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, TypeVar

T = TypeVar("T")


def count_lines(path: Path, has_header: bool = True) -> int:
    """
    Count the number of data lines in a text file.

    Uses buffered binary reads for speed — significantly faster than
    iterating line-by-line in Python for large files (100 M+ lines).

    Parameters
    ----------
    path : Path
        File to count.
    has_header : bool
        If True, subtract 1 from the raw line count to exclude the header row.
        The DD SMILES format always has a 'smiles idnumber' header.

    Returns
    -------
    int
        Number of data (non-header) lines.
    """
    count = 0
    buf_size = 1 << 20  # 1 MB read buffer
    with open(path, "rb") as fh:
        buf = fh.read(buf_size)
        while buf:
            count += buf.count(b"\n")
            buf = fh.read(buf_size)
    return max(0, count - (1 if has_header else 0))


def zero_pad(index: int, total: int) -> str:
    """
    Return a zero-padded string wide enough to sort correctly up to *total*.

    Examples
    --------
    >>> zero_pad(3, 1000)
    '0003'
    >>> zero_pad(3, 100)
    '003'
    """
    width = len(str(total))
    return str(index).zfill(width)


def iter_batches(items: list[T], size: int) -> Iterator[list[T]]:
    """
    Yield successive fixed-size sub-lists from *items*.

    The final batch may be smaller than *size*.

    Examples
    --------
    >>> list(iter_batches([1, 2, 3, 4, 5], 2))
    [[1, 2], [3, 4], [5]]
    """
    for start in range(0, len(items), size):
        yield items[start : start + size]


def smiles_files_in(directory: Path) -> list[Path]:
    """
    Return all .smi and .txt files in *directory*, sorted by name.

    Used by resume-logic in several steps to detect previously created output.
    """
    files = sorted(directory.glob("*.smi")) + sorted(directory.glob("*.txt"))
    return sorted(files)