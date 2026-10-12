from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd


MAX_CSV_BYTES = 10 * 1024 * 1024


class DatasetLoadError(ValueError):
    """Raised when a CSV cannot be safely loaded for analysis."""


def _read_bytes(source: bytes | bytearray | BinaryIO | Path | str) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (Path, str)):
        return Path(source).read_bytes()
    if hasattr(source, "getvalue"):
        return bytes(source.getvalue())

    position = source.tell() if hasattr(source, "tell") else None
    content = source.read()
    if position is not None and hasattr(source, "seek"):
        source.seek(position)
    return content


def load_csv_dataset(
    source: bytes | bytearray | BinaryIO | Path | str,
    *,
    max_bytes: int = MAX_CSV_BYTES,
) -> tuple[pd.DataFrame, str]:
    """Load and validate a CSV, returning its frame and content fingerprint."""
    try:
        content = _read_bytes(source)
    except OSError as exc:
        raise DatasetLoadError("The CSV file could not be read.") from exc

    if not content:
        raise DatasetLoadError("The CSV file is empty.")
    if len(content) > max_bytes:
        limit_mb = max_bytes // (1024 * 1024)
        raise DatasetLoadError(f"The CSV exceeds the {limit_mb} MB upload limit.")

    try:
        frame = pd.read_csv(BytesIO(content))
    except UnicodeDecodeError as exc:
        raise DatasetLoadError("The CSV must use UTF-8 compatible text encoding.") from exc
    except pd.errors.EmptyDataError as exc:
        raise DatasetLoadError("The CSV does not contain any columns.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetLoadError("The CSV structure is invalid or inconsistent.") from exc

    if frame.columns.empty:
        raise DatasetLoadError("The CSV does not contain any columns.")
    if frame.empty:
        raise DatasetLoadError("The CSV contains column headers but no data rows.")

    fingerprint = hashlib.sha256(content).hexdigest()
    return frame, fingerprint
