from __future__ import annotations

from .curves import DiscountCurve
from .instruments import CashflowPeriod


def par_swap_rate(
    discount_curve: DiscountCurve,
    projection_curve: DiscountCurve,
    accrual_periods: list[CashflowPeriod],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for period in accrual_periods:
        tau = period.accrual_factor
        forward = projection_curve.forward_rate(period.start_date, period.end_date, day_count="ACT/360")
        discount_factor = discount_curve.df(period.end_date)
        numerator += tau * discount_factor * forward
        denominator += tau * discount_factor
    return numerator / denominator


def pv_fixed_leg(
    notional: float,
    fixed_rate: float,
    periods: list[CashflowPeriod],
    discount_curve: DiscountCurve,
) -> float:
    return sum(notional * fixed_rate * period.accrual_factor * discount_curve.df(period.end_date) for period in periods)


def pv_float_leg(
    notional: float,
    periods: list[CashflowPeriod],
    discount_curve: DiscountCurve,
    projection_curve: DiscountCurve,
    spread: float = 0.0,
) -> float:
    pv = 0.0
    for period in periods:
        forward = projection_curve.forward_rate(period.start_date, period.end_date, day_count="ACT/360") + spread
        pv += notional * period.accrual_factor * forward * discount_curve.df(period.end_date)
    return pv


def pv_swap(
    notional: float,
    fixed_rate: float,
    periods: list[CashflowPeriod],
    discount_curve: DiscountCurve,
    projection_curve: DiscountCurve,
    spread: float = 0.0,
) -> float:
    return pv_float_leg(notional, periods, discount_curve, projection_curve, spread=spread) - pv_fixed_leg(
        notional, fixed_rate, periods, discount_curve
    )
