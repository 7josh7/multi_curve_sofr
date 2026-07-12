from __future__ import annotations

import math
from datetime import date

import pytest

from sofr_curve_engine.curves import DiscountCurve
from sofr_curve_engine.hw_model import (
    A_const_sigma,
    SwaptionQuote,
    U_j_const_sigma,
    atm_normal_vol,
    calibrate_hw,
    convexity_1m,
    implied_forward_from_3m_future,
    swaption_price_hw,
    zcb_call,
    zcb_put,
)

# ─── OIS Discount Curve (matches ois_curve.csv) ───────────────────────────────

_VALUATION_DATE = date(2025, 4, 15)

_OIS_DATA = [
    (date(2025, 5, 15),  0.041996),
    (date(2025, 7, 15),  0.042657),
    (date(2025, 10, 15), 0.043561),
    (date(2026, 4, 15),  0.045068),
    (date(2027, 4, 15),  0.047282),
    (date(2028, 4, 17),  0.048838),
    (date(2030, 4, 15),  0.050965),
    (date(2032, 4, 15),  0.052597),
    (date(2035, 4, 16),  0.054786),
]


@pytest.fixture(scope="module")
def ois_curve() -> DiscountCurve:
    return DiscountCurve.from_zero_rates(
        _VALUATION_DATE,
        [d for d, _ in _OIS_DATA],
        [z for _, z in _OIS_DATA],
    )


# ─── Legacy primitive tests ───────────────────────────────────────────────────

def test_convexity_zero_sigma() -> None:
    assert abs(convexity_1m(a=0.03, sigma=0.0, T_start=0.25, T_end=1.0 / 3.0)) < 1e-12


def test_three_month_u_term_zero_sigma() -> None:
    assert abs(U_j_const_sigma(a=0.03, sigma=0.0, T_prev=0.5, T_curr=0.75)) < 1e-12


def test_three_month_implied_forward_matches_bootstrap_identity() -> None:
    future_rate = 0.05
    tau = 0.25
    adjustment = U_j_const_sigma(a=0.03, sigma=0.01, T_prev=0.75, T_curr=1.0)

    implied = implied_forward_from_3m_future(future_rate, tau, adjustment)

    assert 1.0 + tau * implied == pytest.approx((1.0 + tau * future_rate) * math.exp(adjustment))


def test_a_term_is_one_when_sigma_is_zero() -> None:
    assert abs(A_const_sigma(a=0.03, sigma=0.0, t=0.0, T=2.0) - 1.0) < 1e-12


def test_a_term_matches_constant_sigma_integral() -> None:
    a = 0.03
    sigma = 0.01
    t = 0.25
    maturity = 2.0
    horizon = maturity - t
    integrated_b_squared = (
        horizon
        - 2.0 * (1.0 - math.exp(-a * horizon)) / a
        + (1.0 - math.exp(-2.0 * a * horizon)) / (2.0 * a)
    ) / a**2
    expected = math.exp(0.5 * sigma**2 * integrated_b_squared)

    assert A_const_sigma(a=a, sigma=sigma, t=t, T=maturity) == pytest.approx(expected, rel=1e-9)


# ─── ZCB Option Tests ─────────────────────────────────────────────────────────

def test_zcb_put_call_parity() -> None:
    """Put-call parity: Call - Put = P(0,T_bond) - K * P(0,T_exp)."""
    p0_exp, p0_bond, strike, sigma_p = 0.95, 0.88, 0.93, 0.015
    call = zcb_call(p0_exp, p0_bond, strike, sigma_p)
    put  = zcb_put(p0_exp, p0_bond, strike, sigma_p)
    assert abs((call - put) - (p0_bond - strike * p0_exp)) < 1e-12


def test_zcb_option_zero_sigma_intrinsic() -> None:
    """Zero sigma → option collapses to intrinsic value."""
    p0_exp, p0_bond, strike = 0.95, 0.88, 0.93
    call = zcb_call(p0_exp, p0_bond, strike, sigma_p=0.0)
    put  = zcb_put(p0_exp, p0_bond, strike, sigma_p=0.0)
    assert abs(call - max(p0_bond - strike * p0_exp, 0.0)) < 1e-12
    assert abs(put  - max(strike * p0_exp - p0_bond, 0.0)) < 1e-12


# ─── Swaption Pricing Tests ───────────────────────────────────────────────────

def test_swaption_zero_sigma_returns_zero(ois_curve: DiscountCurve) -> None:
    T_exp = 1.0
    payment_times = [2.0, 3.0]
    coupons = [0.05, 1.05]
    price = swaption_price_hw(0.03, 0.0, ois_curve, T_exp, payment_times, coupons, is_payer=True)
    assert price == 0.0


def test_payer_receiver_put_call_parity(ois_curve: DiscountCurve) -> None:
    """
    Payer - Receiver = value of forward-start swap = P(0,T_exp) - CB(0).

    CB(0) = sum_i c_i * P(0, T_i) is the bond price at t=0.
    """
    a, sigma = 0.03, 0.012
    T_exp = 1.0
    payment_times = [2.0, 3.0, 4.0, 5.0, 6.0]
    tau_i = [1.0] * 5
    annuity = sum(ois_curve.df(T) for T in payment_times)
    K = (ois_curve.df(T_exp) - ois_curve.df(T_exp + 5.0)) / annuity

    coupons = [K * t for t in tau_i]
    coupons[-1] += 1.0

    payer    = swaption_price_hw(a, sigma, ois_curve, T_exp, payment_times, coupons, is_payer=True)
    receiver = swaption_price_hw(a, sigma, ois_curve, T_exp, payment_times, coupons, is_payer=False)

    # At t=0, forward swap value for the fixed-rate receiver = CB(0) - P(0,T_exp)
    cb0 = sum(c * ois_curve.df(T) for c, T in zip(coupons, payment_times, strict=True))
    fwd_swap_recv = cb0 - ois_curve.df(T_exp)

    assert abs((receiver - payer) - fwd_swap_recv) < 1e-8


def test_atm_normal_vol_level(ois_curve: DiscountCurve) -> None:
    """ATM normal vol should be roughly sigma * sqrt(T_exp) order of magnitude."""
    a, sigma = 0.03, 0.012
    T_exp = 2.0
    payment_times = [3.0, 4.0, 5.0, 6.0, 7.0]
    tau_i = [1.0] * 5
    annuity = sum(ois_curve.df(T) for T in payment_times)
    K = (ois_curve.df(T_exp) - ois_curve.df(T_exp + 5.0)) / annuity

    vol = atm_normal_vol(a, sigma, ois_curve, T_exp, payment_times, tau_i, K)

    # Normal vol ≈ 100-130 bp/yr for sigma=120bp, T_exp=2Y; must be positive and finite
    assert 0.005 < vol < 0.03


# ─── Calibration Round-Trip ───────────────────────────────────────────────────

def _make_quotes(
    ois_curve: DiscountCurve,
    a_true: float,
    sigma_true: float,
    expiries: list[float],
    tenors: list[float],
) -> list[SwaptionQuote]:
    """Generate SwaptionQuote objects at true (a, sigma) for calibration testing."""
    quotes = []
    for T_exp in expiries:
        for tenor in tenors:
            n = int(round(tenor))
            payment_times = [T_exp + k for k in range(1, n + 1)]
            tau_i = [1.0] * n
            annuity = sum(ois_curve.df(T) for T in payment_times)
            K = (ois_curve.df(T_exp) - ois_curve.df(T_exp + tenor)) / annuity
            vol = atm_normal_vol(a_true, sigma_true, ois_curve, T_exp, payment_times, tau_i, K)
            quotes.append(SwaptionQuote(T_exp=T_exp, payment_times=payment_times,
                                        tau_i=tau_i, fixed_rate=K, market_normal_vol=vol))
    return quotes


def test_calibrate_hw_recovers_sigma(ois_curve: DiscountCurve) -> None:
    """
    Round-trip: generate vol surface at known (a, sigma), calibrate, recover sigma.

    With a_true fixed we only fit sigma; tolerance is tight (<0.1 bp).
    """
    a_true, sigma_true = 0.03, 0.012
    quotes = _make_quotes(ois_curve, a_true, sigma_true,
                          expiries=[1.0, 2.0, 3.0, 5.0], tenors=[2.0, 5.0])

    # Fix a at the true value and solve only for sigma
    a_cal, sigma_cal = calibrate_hw(
        quotes, ois_curve,
        a_init=a_true, sigma_init=0.01,
        a_bounds=(a_true - 1e-6, a_true + 1e-6),  # effectively pin a
        sigma_bounds=(1e-5, 0.10),
    )
    assert abs(sigma_cal - sigma_true) < 1e-5, (
        f"sigma: expected {sigma_true:.6f}, got {sigma_cal:.6f}"
    )


def test_calibrate_hw_joint_round_trip(ois_curve: DiscountCurve) -> None:
    """
    Round-trip calibrating both (a, sigma) from a 4×2 grid of swaption vols.

    Uses a wider tolerance because the HW surface with constant sigma has
    limited curvature; parameter identification is approximate.
    """
    a_true, sigma_true = 0.03, 0.012
    quotes = _make_quotes(ois_curve, a_true, sigma_true,
                          expiries=[1.0, 2.0, 3.0, 5.0], tenors=[2.0, 5.0])

    a_cal, sigma_cal = calibrate_hw(
        quotes, ois_curve,
        a_init=0.05, sigma_init=0.01,
        a_bounds=(1e-4, 0.5),
        sigma_bounds=(1e-5, 0.10),
    )

    # Both parameters should be recovered within reasonable tolerance
    assert abs(sigma_cal - sigma_true) < 5e-4, (
        f"sigma: expected {sigma_true:.4f}, got {sigma_cal:.4f}"
    )
    assert abs(a_cal - a_true) < 0.01, (
        f"a: expected {a_true:.4f}, got {a_cal:.4f}"
    )
