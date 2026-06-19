from __future__ import annotations

from datetime import date

from .utils import ensure_date

SUPPORTED_DAY_COUNTS = {"ACT/360", "ACT/365F", "30/360"}


def yearfrac(start: date, end: date, convention: str) -> float:
    start_date = ensure_date(start)
    end_date = ensure_date(end)
    if end_date < start_date:
        raise ValueError("End date must be on or after start date.")

    normalized = convention.upper()
    if normalized not in SUPPORTED_DAY_COUNTS:
        raise ValueError(f"Unsupported day count convention: {convention}")

    if normalized == "ACT/360":
        return (end_date - start_date).days / 360.0
    if normalized == "ACT/365F":
        return (end_date - start_date).days / 365.0

    d1 = min(start_date.day, 30)
    d2 = end_date.day
    if d1 == 30:
        d2 = min(d2, 30)
    numerator = (
        360 * (end_date.year - start_date.year)
        + 30 * (end_date.month - start_date.month)
        + (d2 - d1)
    )
    return numerator / 360.0
