from __future__ import annotations

from datetime import date, timedelta

from .utils import ensure_date


SUPPORTED_ROLLS = {"FOLLOWING", "MODIFIED FOLLOWING"}


def is_business_day(day: date, calendar: str = "WEEKEND") -> bool:
    current = ensure_date(day)
    if calendar.upper() != "WEEKEND":
        raise ValueError(f"Unsupported calendar: {calendar}")
    return current.weekday() < 5


def adjust_business_day(day: date, calendar: str = "WEEKEND", roll: str = "Modified Following") -> date:
    current = ensure_date(day)
    roll_name = roll.upper()
    if roll_name not in SUPPORTED_ROLLS:
        raise ValueError(f"Unsupported business-day roll: {roll}")

    if is_business_day(current, calendar):
        return current

    adjusted = current
    while not is_business_day(adjusted, calendar):
        adjusted += timedelta(days=1)

    if roll_name == "FOLLOWING":
        return adjusted

    if adjusted.month != current.month:
        adjusted = current
        while not is_business_day(adjusted, calendar):
            adjusted -= timedelta(days=1)
    return adjusted
