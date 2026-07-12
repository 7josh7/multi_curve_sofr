from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

import pandas as pd


def ensure_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"Unsupported date value: {value!r}")


def tenor_to_months(tenor: str) -> int:
    normalized = tenor.strip().upper()
    if normalized.endswith("M"):
        return int(normalized[:-1])
    if normalized.endswith("Y"):
        return int(normalized[:-1]) * 12
    raise ValueError(f"Unsupported tenor: {tenor}")


def annualize_bp(value: float) -> float:
    return value * 10_000.0


def project_root_from_here(file_path: str | Path) -> Path:
    return Path(file_path).resolve().parents[2]


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def dedupe_sorted_dates(dates: Iterable[date]) -> list[date]:
    return sorted({ensure_date(item) for item in dates})
