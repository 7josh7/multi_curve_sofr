from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from .dates import generate_schedule
from .daycount import yearfrac
from .utils import ensure_date


@dataclass(frozen=True)
class CashflowPeriod:
    start_date: date
    end_date: date
    accrual_factor: float


@dataclass(frozen=True)
class FuturesQuote:
    contract_type: Literal["SOFR_1M", "SOFR_3M"]
    contract_code: str
    start_date: date
    end_date: date
    price: float

    @property
    def implied_rate(self) -> float:
        return (100.0 - self.price) / 100.0


@dataclass(frozen=True)
class SwapQuote:
    tenor: str
    start_date: date
    end_date: date
    fixed_rate: float
    pay_freq: str
    day_count: str


@dataclass(frozen=True)
class FixedLeg:
    start_date: date
    end_date: date
    pay_freq: str
    day_count: str
    calendar: str = "WEEKEND"
    roll: str = "Modified Following"

    def periods(self) -> list[CashflowPeriod]:
        return build_periods(
            self.start_date,
            self.end_date,
            self.pay_freq,
            self.day_count,
            calendar=self.calendar,
            roll=self.roll,
        )


@dataclass(frozen=True)
class FloatingLeg(FixedLeg):
    pass


def build_periods(
    start_date: date,
    end_date: date,
    frequency: str,
    day_count: str,
    calendar: str = "WEEKEND",
    roll: str = "Modified Following",
) -> list[CashflowPeriod]:
    schedule = generate_schedule(start_date, end_date, frequency, calendar=calendar, roll=roll)
    periods: list[CashflowPeriod] = []
    for start, end in zip(schedule[:-1], schedule[1:], strict=True):
        periods.append(
            CashflowPeriod(
                start_date=ensure_date(start),
                end_date=ensure_date(end),
                accrual_factor=yearfrac(start, end, day_count),
            )
        )
    return periods
