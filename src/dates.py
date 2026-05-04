from __future__ import annotations

import calendar as pycalendar
from datetime import date, timedelta

from .calendars import adjust_business_day
from .utils import ensure_date


FREQUENCY_TO_MONTHS = {
    "MONTHLY": 1,
    "QUARTERLY": 3,
    "SEMIANNUAL": 6,
    "ANNUAL": 12,
}


def add_months(day: date, months: int) -> date:
    current = ensure_date(day)
    month_index = current.month - 1 + months
    year = current.year + month_index // 12
    month = month_index % 12 + 1
    last_day = pycalendar.monthrange(year, month)[1]
    return date(year, month, min(current.day, last_day))


def generate_schedule(
    start: date,
    end: date,
    freq: str,
    calendar: str = "WEEKEND",
    roll: str = "Modified Following",
) -> list[date]:
    start_date = ensure_date(start)
    end_date = ensure_date(end)
    if end_date <= start_date:
        return [start_date, end_date]

    freq_name = freq.upper()
    if freq_name not in FREQUENCY_TO_MONTHS:
        raise ValueError(f"Unsupported frequency: {freq}")

    step_months = FREQUENCY_TO_MONTHS[freq_name]
    dates = [start_date]
    anchor = start_date
    step = 1
    adjusted_end = adjust_business_day(end_date, calendar=calendar, roll=roll)

    while True:
        candidate = add_months(anchor, step * step_months)
        if candidate >= end_date:
            break
        adjusted_candidate = adjust_business_day(candidate, calendar=calendar, roll=roll)
        if adjusted_candidate >= adjusted_end:
            break
        if adjusted_candidate != dates[-1]:
            dates.append(adjusted_candidate)
        step += 1

    if adjusted_end != dates[-1]:
        dates.append(adjusted_end)
    return dates


def is_imm_date(day: date) -> bool:
    current = ensure_date(day)
    return current.month in {3, 6, 9, 12} and current.weekday() == 2 and 15 <= current.day <= 21


def next_imm_date(day: date) -> date:
    current = ensure_date(day)
    probe = current
    while not is_imm_date(probe):
        probe += timedelta(days=1)
    return probe
