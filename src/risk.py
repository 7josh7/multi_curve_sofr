from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .curves import DiscountCurve
from .instruments import CashflowPeriod
from .pricers import par_swap_rate, pv_swap


def pv01(
    notional: float,
    fixed_rate: float,
    periods: list[CashflowPeriod],
    discount_curve: DiscountCurve,
    projection_curve: DiscountCurve,
    bump_bp: float = 1.0,
) -> float:
    bump = bump_bp / 10_000.0
    discount_up = discount_curve.bump_zero_curve(lambda _time: bump)
    projection_up = projection_curve.bump_zero_curve(lambda _time: bump)
    discount_dn = discount_curve.bump_zero_curve(lambda _time: -bump)
    projection_dn = projection_curve.bump_zero_curve(lambda _time: -bump)
    pv_up = pv_swap(notional, fixed_rate, periods, discount_up, projection_up)
    pv_dn = pv_swap(notional, fixed_rate, periods, discount_dn, projection_dn)
    return (pv_dn - pv_up) / 2.0


def key_rate_dv01(
    notional: float,
    fixed_rate: float,
    periods: list[CashflowPeriod],
    discount_curve: DiscountCurve,
    projection_curve: DiscountCurve,
    key_rates_years: list[float],
    bump_bp: float = 1.0,
    width: float = 1.5,
) -> dict[float, float]:
    bump = bump_bp / 10_000.0
    sensitivities: dict[float, float] = {}
    for key in key_rates_years:
        def key_bump(time: float, anchor: float = key) -> float:
            return bump * max(0.0, 1.0 - abs(time - anchor) / width)

        discount_up = discount_curve.bump_zero_curve(key_bump)
        projection_up = projection_curve.bump_zero_curve(key_bump)
        discount_dn = discount_curve.bump_zero_curve(lambda time, anchor=key: -key_bump(time, anchor))
        projection_dn = projection_curve.bump_zero_curve(lambda time, anchor=key: -key_bump(time, anchor))
        pv_up = pv_swap(notional, fixed_rate, periods, discount_up, projection_up)
        pv_dn = pv_swap(notional, fixed_rate, periods, discount_dn, projection_dn)
        sensitivities[key] = (pv_dn - pv_up) / 2.0
    return sensitivities


def run_scenarios(
    notional: float,
    fixed_rate: float,
    periods: list[CashflowPeriod],
    discount_curve: DiscountCurve,
    projection_curve: DiscountCurve,
    alt_projection_curve: DiscountCurve | None = None,
) -> pd.DataFrame:
    def apply(
        name: str,
        discount_bump: Callable[[float], float],
        projection_bump: Callable[[float], float],
    ) -> dict[str, float | str]:
        bumped_discount = discount_curve.bump_zero_curve(discount_bump, label=f"{name}_discount")
        bumped_projection = projection_curve.bump_zero_curve(projection_bump, label=f"{name}_projection")
        pv = pv_swap(notional, fixed_rate, periods, bumped_discount, bumped_projection)
        return {
            "Scenario": name,
            "Swap PV": pv,
            "DV01": pv01(notional, fixed_rate, periods, bumped_discount, bumped_projection),
            "Par Rate": par_swap_rate(bumped_discount, bumped_projection, periods),
            "Curve End DF": bumped_projection.dfs[-1],
        }

    scenarios = [
        apply("Base", lambda _t: 0.0, lambda _t: 0.0),
        apply("Parallel +25bp", lambda _t: 0.0025, lambda _t: 0.0025),
        apply("Parallel -25bp", lambda _t: -0.0025, lambda _t: -0.0025),
        apply("2s10s Steepener", lambda t: -0.0010 if t <= 2.0 else 0.0015, lambda t: -0.0010 if t <= 2.0 else 0.0015),
        apply("2s10s Flattener", lambda t: 0.0010 if t <= 2.0 else -0.0015, lambda t: 0.0010 if t <= 2.0 else -0.0015),
        apply("Front-End Stress", lambda t: 0.0025 * max(0.0, 1.0 - t / 3.0), lambda t: 0.0025 * max(0.0, 1.0 - t / 3.0)),
        apply("Basis Widening", lambda _t: 0.0, lambda _t: 0.0010),
        apply("Funding Spread +10bp", lambda _t: 0.0010, lambda _t: 0.0),
        apply("Funding Spread +25bp", lambda _t: 0.0025, lambda _t: 0.0),
    ]

    if alt_projection_curve is not None:
        pv = pv_swap(notional, fixed_rate, periods, discount_curve, alt_projection_curve)
        scenarios.append(
            {
                "Scenario": "Convexity Off",
                "Swap PV": pv,
                "DV01": pv01(notional, fixed_rate, periods, discount_curve, alt_projection_curve),
                "Par Rate": par_swap_rate(discount_curve, alt_projection_curve, periods),
                "Curve End DF": alt_projection_curve.dfs[-1],
            }
        )

    return pd.DataFrame(scenarios)
