from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np

from .daycount import yearfrac
from .interpolation import LogDFInterpolator
from .utils import ensure_date


@dataclass(frozen=True)
class CurveNode:
    pillar_date: date
    time: float
    df: float


class DiscountCurve:
    def __init__(
        self,
        valuation_date: date,
        pillar_dates: list[date],
        dfs: list[float],
        interpolator_cls: type[LogDFInterpolator] = LogDFInterpolator, 
        #type[] means the argument should be a class that is a subclass of LogDFInterpolator or like it, not an instance of it. That lets you swap interpolation implementations without changing DiscountCurve.
        label: str = "discount",
    ) -> None:
        self.valuation_date = ensure_date(valuation_date)
        if len(pillar_dates) != len(dfs):
            raise ValueError("Pillar dates and discount factors must have the same length.")

        sorted_pairs = sorted((ensure_date(pillar), float(df)) for pillar, df in zip(pillar_dates, dfs, strict=True))
        dates = [self.valuation_date]
        values = [1.0]
        for pillar, df in sorted_pairs:
            if pillar == self.valuation_date:
                values[0] = df
                continue
            dates.append(pillar)
            values.append(df)

        self.pillar_dates = dates
        self.dfs = np.asarray(values, dtype=float)
        self.times = np.asarray(
            [0.0] + [yearfrac(self.valuation_date, pillar, "ACT/365F") for pillar in dates[1:]],
            dtype=float,
        )
        self.label = label
        self.interpolator = interpolator_cls(self.times, self.dfs)

    @classmethod
    def from_zero_rates(
        cls,
        valuation_date: date,
        pillar_dates: list[date],
        zero_rates: list[float],
        label: str = "discount",
    ) -> DiscountCurve:
        valuation = ensure_date(valuation_date)
        dfs = []
        for pillar, zero in zip(pillar_dates, zero_rates, strict=True):
            t = yearfrac(valuation, pillar, "ACT/365F")
            dfs.append(math.exp(-float(zero) * t))
        return cls(valuation, pillar_dates, dfs, label=label)

    def time_from_date(self, target: date | float) -> float:
        if isinstance(target, (int, float)):
            return float(target)
        return yearfrac(self.valuation_date, ensure_date(target), "ACT/365F")

    def df(self, target: date | float) -> float:
        return self.interpolator.df(self.time_from_date(target))

    def zero_rate(self, target: date | float, comp: str = "cont") -> float:
        t = self.time_from_date(target)
        if t <= 0.0:
            t = self.times[1]
        cont_rate = self.interpolator.zero(t)
        if comp.lower() == "cont":
            return cont_rate
        if comp.lower() == "simple":
            return (math.exp(cont_rate * t) - 1.0) / t
        raise ValueError(f"Unsupported compounding: {comp}")

    def forward_df_ratio(self, start: date | float, end: date | float) -> float:
        return self.df(end) / self.df(start)

    def forward_rate(
        self,
        start: date,
        end: date,
        day_count: str = "ACT/360",
        rate_type: str = "simple",
    ) -> float:
        tau = yearfrac(start, end, day_count)
        ratio = self.df(start) / self.df(end)
        if rate_type.lower() == "simple":
            return (ratio - 1.0) / tau
        if rate_type.lower() == "cont":
            return -math.log(self.df(end) / self.df(start)) / tau
        raise ValueError(f"Unsupported rate type: {rate_type}")

    def pillar_zero_rates(self) -> list[float]:
        return [self.zero_rate(time) for time in self.times[1:]]

    def bump_zero_curve(self, bump_fn: Callable[[float], float], label: str | None = None) -> DiscountCurve:
        bumped_zeros = [zero + float(bump_fn(time)) for time, zero in zip(self.times[1:], self.pillar_zero_rates(), strict=True)]
        return DiscountCurve.from_zero_rates(
            valuation_date=self.valuation_date,
            pillar_dates=self.pillar_dates[1:],
            zero_rates=bumped_zeros,
            label=label or self.label,
        )

    def apply_discount_spread(self, spread: float, label: str | None = None) -> DiscountCurve:
        return self.bump_zero_curve(lambda _time: spread, label=label or f"{self.label}_spread")

    def nodes(self) -> list[CurveNode]:
        return [
            CurveNode(pillar_date=pillar, time=time, df=df)
            for pillar, time, df in zip(self.pillar_dates[1:], self.times[1:], self.dfs[1:], strict=True)
        ]


class ForwardCurve:
    def __init__(self, projection_curve: DiscountCurve, discount_curve: DiscountCurve | None = None) -> None:
        self.projection_curve = projection_curve
        self.discount_curve = discount_curve or projection_curve

    def forward_rate(self, start: date, end: date, day_count: str = "ACT/360") -> float:
        return self.projection_curve.forward_rate(start, end, day_count=day_count, rate_type="simple")
