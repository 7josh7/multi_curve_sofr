from __future__ import annotations

import math
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
    """Signed PV change per +1 bp parallel move, estimated centrally.

    The project prices a fixed-payer swap as floating PV minus fixed PV, so a
    conventional payer position normally has positive PV01 when rates rise.
    ``bump_bp`` controls the finite-difference stencil and the result is always
    normalized back to one basis point.
    """

    if not math.isfinite(float(bump_bp)) or bump_bp <= 0.0:
        raise ValueError("bump_bp must be finite and positive.")
    bump = bump_bp / 10_000.0
    discount_up = discount_curve.bump_zero_curve(lambda _time: bump)
    projection_up = projection_curve.bump_zero_curve(lambda _time: bump)
    discount_dn = discount_curve.bump_zero_curve(lambda _time: -bump)
    projection_dn = projection_curve.bump_zero_curve(lambda _time: -bump)
    pv_up = pv_swap(notional, fixed_rate, periods, discount_up, projection_up)
    pv_dn = pv_swap(notional, fixed_rate, periods, discount_dn, projection_dn)
    return (pv_up - pv_dn) / (2.0 * bump_bp)


def _partition_weight(time: float, anchor: float, keys: list[float]) -> float:
    """Piecewise-linear key-rate weight; weights sum to one at every time."""

    if len(keys) == 1:
        return 1.0
    if time <= keys[0]:
        return 1.0 if anchor == keys[0] else 0.0
    if time >= keys[-1]:
        return 1.0 if anchor == keys[-1] else 0.0
    for left, right in zip(keys[:-1], keys[1:], strict=True):
        if left <= time <= right:
            if anchor == left:
                return (right - time) / (right - left)
            if anchor == right:
                return (time - left) / (right - left)
            return 0.0
    return 0.0


def key_rate_dv01(
    notional: float,
    fixed_rate: float,
    periods: list[CashflowPeriod],
    discount_curve: DiscountCurve,
    projection_curve: DiscountCurve,
    key_rates_years: list[float],
    bump_bp: float = 1.0,
    width: float | None = None,
) -> dict[float, float]:
    """Signed key-rate DV01s normalized to one basis point.

    By default, piecewise-linear partition-of-unity bumps make the key-rate
    ladder reconcile closely to parallel PV01 (up to finite-difference
    cross-gamma). Passing ``width`` retains the earlier independent triangular
    bumps for callers that explicitly need that non-reconciling shape.
    """

    if not math.isfinite(float(bump_bp)) or bump_bp <= 0.0:
        raise ValueError("bump_bp must be finite and positive.")
    keys = sorted(float(value) for value in key_rates_years)
    if not keys or any(not math.isfinite(value) or value <= 0.0 for value in keys):
        raise ValueError("key_rates_years must contain finite positive maturities.")
    if len(set(keys)) != len(keys):
        raise ValueError("key_rates_years must not contain duplicates.")
    if width is not None and (not math.isfinite(float(width)) or width <= 0.0):
        raise ValueError("width must be finite and positive when provided.")

    bump = bump_bp / 10_000.0
    sensitivities: dict[float, float] = {}
    for key in keys:
        def key_bump(time: float, anchor: float = key) -> float:
            if width is not None:
                return bump * max(0.0, 1.0 - abs(time - anchor) / width)
            return bump * _partition_weight(float(time), anchor, keys)

        def negative_key_bump(time: float, anchor: float = key) -> float:
            return -key_bump(time, anchor)

        discount_up = discount_curve.bump_zero_curve(key_bump)
        projection_up = projection_curve.bump_zero_curve(key_bump)
        discount_dn = discount_curve.bump_zero_curve(negative_key_bump)
        projection_dn = projection_curve.bump_zero_curve(negative_key_bump)
        pv_up = pv_swap(notional, fixed_rate, periods, discount_up, projection_up)
        pv_dn = pv_swap(notional, fixed_rate, periods, discount_dn, projection_dn)
        sensitivities[key] = (pv_up - pv_dn) / (2.0 * bump_bp)
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
