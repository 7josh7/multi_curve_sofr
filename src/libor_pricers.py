from __future__ import annotations

from datetime import date

from .curves import DiscountCurve
from .instruments import CashflowPeriod
from .utils import ensure_date


def pv_libor_swap_with_sofr_fallback(
    notional: float,
    fixed_rate: float,
    periods: list[CashflowPeriod],
    discount_curve: DiscountCurve,
    projection_curve: DiscountCurve,
    libor_forwards: dict[date, float],
    fallback_spread: float,
    fallback_start_date: date,
) -> float:
    """Mercurio section 9 fixed-payer LIBOR swap with SOFR fallback."""

    fallback_start = ensure_date(fallback_start_date)
    float_leg = 0.0
    fixed_leg = 0.0
    for period in periods:
        discount = discount_curve.df(period.end_date)
        if period.end_date <= fallback_start:
            forward = _libor_forward_for_period(period, libor_forwards)
        else:
            forward = projection_curve.forward_rate(period.start_date, period.end_date, day_count="ACT/360") + fallback_spread
        float_leg += notional * period.accrual_factor * forward * discount
        fixed_leg += notional * fixed_rate * period.accrual_factor * discount
    return float_leg - fixed_leg


def pv_libor_sofr_basis_swap_to_libor_payer(
    notional: float,
    periods: list[CashflowPeriod],
    discount_curve: DiscountCurve,
    projection_curve: DiscountCurve,
    libor_forwards: dict[date, float],
    fallback_spread: float,
    fallback_start_date: date,
) -> float:
    """Mercurio section 10 basis swap value to the LIBOR payer."""

    fallback_start = ensure_date(fallback_start_date)
    sofr_leg = 0.0
    libor_or_fallback_leg = 0.0
    for period in periods:
        discount = discount_curve.df(period.end_date)
        sofr_forward = projection_curve.forward_rate(period.start_date, period.end_date, day_count="ACT/360")
        if period.end_date <= fallback_start:
            libor_forward = _libor_forward_for_period(period, libor_forwards)
        else:
            libor_forward = sofr_forward + fallback_spread
        sofr_leg += notional * period.accrual_factor * sofr_forward * discount
        libor_or_fallback_leg += notional * period.accrual_factor * libor_forward * discount
    return sofr_leg - libor_or_fallback_leg


def libor_forward_map_from_calibration_table(calibration_table) -> dict[date, float]:
    return {
        ensure_date(row.end_date): float(row.libor_forward)
        for row in calibration_table.itertuples(index=False)
    }


def _libor_forward_for_period(period: CashflowPeriod, libor_forwards: dict[date, float]) -> float:
    end = ensure_date(period.end_date)
    if end in libor_forwards:
        return float(libor_forwards[end])
    if not libor_forwards:
        raise ValueError("At least one LIBOR forward is required.")
    nearest = min(libor_forwards, key=lambda item: abs((item - end).days))
    return float(libor_forwards[nearest])
