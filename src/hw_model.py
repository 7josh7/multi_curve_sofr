from __future__ import annotations

import math

import numpy as np


def B(a: float, t: float, T: float) -> float:
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
    return ((1.0 + tau * future_rate) * math.exp(-U) - 1.0) / tau
