from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import norm as _norm

from .calibration import CalibrationDiagnostics, boundary_hits

if TYPE_CHECKING:
    from .curves import DiscountCurve


# ─── HW Primitives ─────────────────────────────────────────────────────────────

def B(a: float, t: float, T: float) -> float:
    """(1 - exp(-a*(T-t))) / a; the duration-like factor in Hull-White."""
    if T < t:
        raise ValueError("Bond maturity must be on or after the evaluation time.")
    if abs(a) < 1e-12:
        return T - t
    return (1.0 - math.exp(-a * (T - t))) / a


def A_const_sigma(a: float, sigma: float, t: float, T: float, n_grid: int = 2000) -> float:
    if T < t:
        raise ValueError("Bond maturity must be on or after the evaluation time.")
    grid = np.linspace(t, T, n_grid)
    values = sigma**2 * np.array([B(a, point, T) ** 2 for point in grid])
    integral = float(np.trapz(values, grid))
    return math.exp(0.5 * integral)


def convexity_1m(a: float, sigma: float, T_start: float, T_end: float) -> float:
    if sigma == 0.0:
        return 0.0
    delta = T_end - T_start
    if delta <= 0.0:
        raise ValueError("Futures end time must be after start time.")
    if abs(a) < 1e-8:
        a = 1e-8
    numerator = (
        delta
        + 2.0 / a * math.exp(-a * T_end) * (1.0 - math.exp(a * delta))
        - 1.0 / (2.0 * a) * math.exp(-2.0 * a * T_end) * (1.0 - math.exp(2.0 * a * delta))
    )
    return sigma**2 / (2.0 * delta * a**2) * numerator


def U_j_const_sigma(a: float, sigma: float, T_prev: float, T_curr: float) -> float:
    if sigma == 0.0:
        return 0.0
    if T_curr <= T_prev:
        raise ValueError("Current futures boundary must be after previous boundary.")
    if abs(a) < 1e-8:
        a = 1e-8
    return (sigma**2 / (2.0 * a**3)) * (
        math.exp(-a * (T_curr + T_prev))
        - math.exp(-2.0 * a * T_curr)
        + math.exp(-a * (T_curr - T_prev))
        + 2.0 * a * (T_curr - T_prev)
        - 1.0
        + 2.0 * math.exp(-a * T_curr)
        - 2.0 * math.exp(-a * T_prev)
    )


def implied_forward_from_3m_future(future_rate: float, tau: float, U: float) -> float:
    """Forward consistent with the 3m futures bootstrap identity."""

    if not all(math.isfinite(value) for value in (future_rate, tau, U)) or tau <= 0.0:
        raise ValueError("Future rate and adjustment must be finite and tau must be positive.")
    return ((1.0 + tau * future_rate) * math.exp(U) - 1.0) / tau


# ─── Zero-Coupon Bond Option Pricing ──────────────────────────────────────────

def _zcb_sigma_p(a: float, sigma: float, T_exp: float, T_bond: float) -> float:
    """
    Volatility of ln P(T_exp, T_bond) under the risk-neutral measure in HW.

    sigma_P = (sigma/a) * (1 - exp(-a*(T_bond-T_exp))) * sqrt((1-exp(-2*a*T_exp))/(2*a))
            = B(a, T_exp, T_bond) * sigma_r
    where sigma_r = sigma * sqrt((1-exp(-2*a*T_exp))/(2*a)) is the std dev of r(T_exp).
    """
    if abs(a) < 1e-8:
        a = 1e-8
    sigma_r = sigma * math.sqrt((1.0 - math.exp(-2.0 * a * T_exp)) / (2.0 * a))
    return B(a, T_exp, T_bond) * sigma_r


def zcb_call(p0_exp: float, p0_bond: float, strike: float, sigma_p: float) -> float:
    """European call on P(T_exp, T_bond) struck at `strike`, valued at t=0."""
    if sigma_p < 1e-12:
        return max(p0_bond - strike * p0_exp, 0.0)
    h = math.log(p0_bond / (strike * p0_exp)) / sigma_p + sigma_p / 2.0
    return p0_bond * _norm.cdf(h) - strike * p0_exp * _norm.cdf(h - sigma_p)


def zcb_put(p0_exp: float, p0_bond: float, strike: float, sigma_p: float) -> float:
    """European put on P(T_exp, T_bond) struck at `strike`, valued at t=0."""
    if sigma_p < 1e-12:
        return max(strike * p0_exp - p0_bond, 0.0)
    h = math.log(p0_bond / (strike * p0_exp)) / sigma_p + sigma_p / 2.0
    return strike * p0_exp * _norm.cdf(-(h - sigma_p)) - p0_bond * _norm.cdf(-h)


# ─── Swaption Pricing via Jamshidian Decomposition ────────────────────────────

def swaption_price_hw(
    a: float,
    sigma: float,
    discount_curve: DiscountCurve,
    T_exp: float,
    payment_times: list[float],
    coupons: list[float],
    is_payer: bool = True,
) -> float:
    """
    European swaption price using Jamshidian (1989) decomposition.

    The coupon bond CB(T_exp) = sum_i c_i * P(T_exp, T_i) is decomposed into
    a portfolio of ZCB options sharing the same critical rate r*.

    payment_times : T_1, ..., T_n in years from the valuation date.
    coupons       : c_i at T_i — for a par swap: K*tau_i for i < n,
                    and 1 + K*tau_n at the final payment (includes notional).
    is_payer      : True → right to pay fixed (put on coupon bond).
    """
    if sigma == 0.0:
        return 0.0
    if abs(a) < 1e-8:
        a = 1e-8

    p0_exp = discount_curve.df(T_exp)
    p0_bonds = [discount_curve.df(T) for T in payment_times]

    # Forward bond prices F_i = P(0, T_i) / P(0, T_exp)
    fwd_bonds = [pb / p0_exp for pb in p0_bonds]

    # sigma_P_i = B(a, T_exp, T_i) * sigma_r
    sigma_r = sigma * math.sqrt((1.0 - math.exp(-2.0 * a * T_exp)) / (2.0 * a))
    sigma_ps = [B(a, T_exp, T) * sigma_r for T in payment_times]

    # Jamshidian: find z* such that sum_i c_i * F_i * exp(-sigma_P_i * z) = 1.
    # The function is strictly decreasing in z (all c_i, F_i, sigma_P_i > 0).
    def coupon_bond_value(z: float) -> float:
        return sum(c * f * math.exp(-sp * z) for c, f, sp in zip(coupons, fwd_bonds, sigma_ps, strict=True)) - 1.0

    z_star = brentq(coupon_bond_value, -30.0, 30.0, xtol=1e-12)

    # Per-component ZCB option strikes
    strikes = [f * math.exp(-sp * z_star) for f, sp in zip(fwd_bonds, sigma_ps, strict=True)]

    price = 0.0
    for c, p0b, k, sp in zip(coupons, p0_bonds, strikes, sigma_ps, strict=True):
        if is_payer:
            price += c * zcb_put(p0_exp, p0b, k, sp)
        else:
            price += c * zcb_call(p0_exp, p0b, k, sp)
    return price


# ─── Normal Vol Conversion ─────────────────────────────────────────────────────

def atm_normal_vol(
    a: float,
    sigma: float,
    discount_curve: DiscountCurve,
    T_exp: float,
    payment_times: list[float],
    tau_i: list[float],
    fixed_rate: float,
) -> float:
    """
    HW ATM payer swaption price back-solved to Bachelier normal vol.

    fixed_rate : ATM par swap rate = (P(0,T_exp) - P(0,T_n)) / annuity.
    Returns normal vol in decimal units (e.g. 0.012 = 120 bp/yr).

    Bachelier ATM: price = annuity * sigma_n * sqrt(T_exp) / sqrt(2*pi).
    """
    n = len(payment_times)
    coupons = [fixed_rate * t for t in tau_i]
    coupons[n - 1] += 1.0  # notional at final payment

    price = swaption_price_hw(a, sigma, discount_curve, T_exp, payment_times, coupons, is_payer=True)

    annuity = sum(t * discount_curve.df(T) for t, T in zip(tau_i, payment_times, strict=True))
    return price * math.sqrt(2.0 * math.pi) / (annuity * math.sqrt(T_exp))


# ─── Calibration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SwaptionQuote:
    T_exp: float               # option expiry in years
    payment_times: list[float] # swap payment times T_1, ..., T_n (years)
    tau_i: list[float]         # accrual fractions for each period
    fixed_rate: float          # ATM par swap rate
    market_normal_vol: float   # Bachelier normal vol in decimal


_CALIBRATION_PENALTY = 1e10


def _validate_hw_calibration_inputs(
    quotes: list[SwaptionQuote],
    a_init: float,
    sigma_init: float,
    a_bounds: tuple[float, float],
    sigma_bounds: tuple[float, float],
) -> None:
    if not quotes:
        raise ValueError("Hull-White calibration requires at least one swaption quote.")

    bounds_by_name = {"a": a_bounds, "sigma": sigma_bounds}
    initial_by_name = {"a": float(a_init), "sigma": float(sigma_init)}
    for name, bounds in bounds_by_name.items():
        if len(bounds) != 2:
            raise ValueError(f"{name} bounds must contain exactly two values.")
        lower, upper = float(bounds[0]), float(bounds[1])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0.0 or upper <= lower:
            raise ValueError(f"{name} bounds must be finite and satisfy 0 < lower < upper.")
        initial = initial_by_name[name]
        if not math.isfinite(initial) or not lower <= initial <= upper:
            raise ValueError(f"Initial {name} must be finite and within its calibration bounds.")

    for index, quote in enumerate(quotes):
        scalar_values = {
            "expiry": quote.T_exp,
            "fixed_rate": quote.fixed_rate,
            "market_normal_vol": quote.market_normal_vol,
        }
        if any(not math.isfinite(float(value)) for value in scalar_values.values()):
            raise ValueError(f"Swaption quote {index} contains a non-finite scalar input.")
        if quote.T_exp <= 0.0:
            raise ValueError(f"Swaption quote {index} expiry must be positive.")
        if quote.market_normal_vol < 0.0:
            raise ValueError(f"Swaption quote {index} normal volatility cannot be negative.")
        if not quote.payment_times or len(quote.payment_times) != len(quote.tau_i):
            raise ValueError(f"Swaption quote {index} requires matching non-empty payment times and accruals.")
        payment_times = [float(value) for value in quote.payment_times]
        accruals = [float(value) for value in quote.tau_i]
        if any(not math.isfinite(value) for value in payment_times + accruals):
            raise ValueError(f"Swaption quote {index} contains a non-finite schedule input.")
        if any(value <= quote.T_exp for value in payment_times):
            raise ValueError(f"Swaption quote {index} payment times must be after expiry.")
        if any(right <= left for left, right in zip(payment_times[:-1], payment_times[1:], strict=True)):
            raise ValueError(f"Swaption quote {index} payment times must be strictly increasing.")
        if any(value <= 0.0 for value in accruals):
            raise ValueError(f"Swaption quote {index} accrual fractions must be positive.")


def calibrate_hw_with_diagnostics(
    quotes: list[SwaptionQuote],
    discount_curve: DiscountCurve,
    a_init: float = 0.05,
    sigma_init: float = 0.01,
    a_bounds: tuple[float, float] = (1e-4, 1.0),
    sigma_bounds: tuple[float, float] = (1e-5, 0.10),
) -> tuple[float, float, CalibrationDiagnostics]:
    """Fit Hull-White parameters and return validated optimizer diagnostics."""

    _validate_hw_calibration_inputs(quotes, a_init, sigma_init, a_bounds, sigma_bounds)

    def mean_squared_error(params: np.ndarray) -> float:
        a_p, sigma_p = float(params[0]), float(params[1])
        if a_p <= 0.0 or sigma_p <= 0.0:
            return _CALIBRATION_PENALTY
        sq = 0.0
        for q in quotes:
            try:
                mv = atm_normal_vol(
                    a_p, sigma_p, discount_curve,
                    q.T_exp, q.payment_times, q.tau_i, q.fixed_rate,
                )
                sq += (mv - q.market_normal_vol) ** 2
            except Exception:
                return _CALIBRATION_PENALTY
        return sq / len(quotes)

    result = minimize(
        mean_squared_error,
        x0=np.array([a_init, sigma_init]),
        method="L-BFGS-B",
        bounds=[a_bounds, sigma_bounds],
        options={"ftol": 1e-16, "gtol": 1e-10, "maxiter": 1000},
    )
    values = np.asarray(getattr(result, "x", []), dtype=float)
    objective = float(getattr(result, "fun", float("nan")))
    if not bool(getattr(result, "success", False)):
        raise RuntimeError(f"Hull-White calibration failed: {getattr(result, 'message', 'unknown optimizer error')}")
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise RuntimeError("Hull-White calibration returned non-finite parameters.")
    if not math.isfinite(objective) or objective < 0.0 or objective >= _CALIBRATION_PENALTY:
        raise RuntimeError("Hull-White calibration returned an invalid objective value.")

    a_calibrated, sigma_calibrated = float(values[0]), float(values[1])
    parameter_values = {"a": a_calibrated, "sigma": sigma_calibrated}
    parameter_bounds = {
        "a": (float(a_bounds[0]), float(a_bounds[1])),
        "sigma": (float(sigma_bounds[0]), float(sigma_bounds[1])),
    }
    for name, value in parameter_values.items():
        lower, upper = parameter_bounds[name]
        if not lower <= value <= upper:
            raise RuntimeError(f"Hull-White calibration returned {name} outside its bounds.")

    diagnostics = CalibrationDiagnostics(
        method="swaption_surface",
        converged=True,
        objective_value=objective,
        message=str(getattr(result, "message", "optimizer converged")),
        iterations=int(result.nit) if getattr(result, "nit", None) is not None else None,
        function_evaluations=int(result.nfev) if getattr(result, "nfev", None) is not None else None,
        parameter_values=parameter_values,
        parameter_bounds=parameter_bounds,
        boundary_hits=boundary_hits(parameter_values, parameter_bounds),
        is_regularization_proxy=False,
    )
    return a_calibrated, sigma_calibrated, diagnostics


def calibrate_hw(
    quotes: list[SwaptionQuote],
    discount_curve: DiscountCurve,
    a_init: float = 0.05,
    sigma_init: float = 0.01,
    a_bounds: tuple[float, float] = (1e-4, 1.0),
    sigma_bounds: tuple[float, float] = (1e-5, 0.10),
) -> tuple[float, float]:
    """Compatibility wrapper returning only calibrated ``(a, sigma)``."""

    a_calibrated, sigma_calibrated, _ = calibrate_hw_with_diagnostics(
        quotes,
        discount_curve,
        a_init=a_init,
        sigma_init=sigma_init,
        a_bounds=a_bounds,
        sigma_bounds=sigma_bounds,
    )
    return a_calibrated, sigma_calibrated
