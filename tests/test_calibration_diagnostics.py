from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest

from sofr_curve_engine.bootstrap import (
    calibrate_sigma_regularization_proxy,
    load_market_data,
)
from sofr_curve_engine.curves import DiscountCurve
from sofr_curve_engine.hw_model import (
    SwaptionQuote,
    calibrate_hw,
    calibrate_hw_with_diagnostics,
)


def _discount_curve() -> DiscountCurve:
    return DiscountCurve.from_zero_rates(
        date(2025, 4, 15),
        [date(2026, 4, 15), date(2027, 4, 15), date(2028, 4, 15)],
        [0.04, 0.042, 0.044],
    )


def _quote() -> SwaptionQuote:
    return SwaptionQuote(
        T_exp=1.0,
        payment_times=[2.0, 3.0],
        tau_i=[1.0, 1.0],
        fixed_rate=0.04,
        market_normal_vol=0.01,
    )


def test_smoothness_proxy_exposes_lower_boundary_diagnostics() -> None:
    market = load_market_data(".")

    sigma, diagnostics = calibrate_sigma_regularization_proxy(
        market,
        a=market.config.model.mean_reversion,
        sigma_bounds=market.config.model.sigma_bounds,
    )

    assert sigma == pytest.approx(market.config.model.sigma_bounds[0], abs=1e-6)
    assert diagnostics.converged
    assert diagnostics.is_regularization_proxy
    assert diagnostics.boundary_hits == {"sigma": "lower"}
    assert "not option-implied volatility" in diagnostics.message


def test_smoothness_proxy_rejects_empty_market_inputs() -> None:
    market = load_market_data(".")
    empty = replace(market, futures_1m=[], futures_3m=[])

    with pytest.raises(ValueError, match="futures quote"):
        calibrate_sigma_regularization_proxy(empty, a=0.03, sigma_bounds=(0.001, 0.05))


def test_smoothness_proxy_rejects_optimizer_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    market = load_market_data(".")
    failed = SimpleNamespace(
        success=False,
        message="forced failure",
        x=0.01,
        fun=1.0,
        nit=1,
        nfev=1,
    )
    monkeypatch.setattr("sofr_curve_engine.bootstrap.minimize_scalar", lambda *args, **kwargs: failed)

    with pytest.raises(RuntimeError, match="forced failure"):
        calibrate_sigma_regularization_proxy(market, a=0.03, sigma_bounds=(0.001, 0.05))


def test_hw_calibration_rejects_empty_and_nonfinite_inputs() -> None:
    curve = _discount_curve()
    with pytest.raises(ValueError, match="at least one"):
        calibrate_hw([], curve)

    invalid = replace(_quote(), market_normal_vol=float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        calibrate_hw([invalid], curve)


def test_hw_calibration_rejects_optimizer_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    failed = SimpleNamespace(
        success=False,
        message="forced failure",
        x=[0.03, 0.01],
        fun=1.0,
        nit=1,
        nfev=1,
    )
    monkeypatch.setattr("sofr_curve_engine.hw_model.minimize", lambda *args, **kwargs: failed)

    with pytest.raises(RuntimeError, match="forced failure"):
        calibrate_hw([_quote()], _discount_curve())


def test_hw_calibration_reports_boundary_solution(monkeypatch: pytest.MonkeyPatch) -> None:
    converged = SimpleNamespace(
        success=True,
        message="forced boundary",
        x=[0.0001, 0.00001],
        fun=0.0,
        nit=2,
        nfev=3,
    )
    monkeypatch.setattr("sofr_curve_engine.hw_model.minimize", lambda *args, **kwargs: converged)

    _, _, diagnostics = calibrate_hw_with_diagnostics(
        [_quote()],
        _discount_curve(),
        a_init=0.03,
        sigma_init=0.01,
        a_bounds=(0.0001, 0.5),
        sigma_bounds=(0.00001, 0.1),
    )

    assert diagnostics.boundary_hits == {"a": "lower", "sigma": "lower"}
    assert diagnostics.boundary_hit
