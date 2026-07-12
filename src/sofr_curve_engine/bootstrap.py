from __future__ import annotations  #postpones evaluation of type hints

import math
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .calibration import CalibrationDiagnostics, boundary_hits, configured_diagnostics
from .config import EngineConfig, load_engine_config
from .curves import DiscountCurve
from .data_input import CsvMarketDataSource, MarketData, MarketDataSource
from .daycount import yearfrac
from .hw_model import SwaptionQuote, U_j_const_sigma, calibrate_hw_with_diagnostics, convexity_1m
from .instruments import FuturesQuote, SwapQuote, build_periods
from .pricers import par_swap_rate
from .utils import annualize_bp, ensure_date, tenor_to_months

'''
CSV market data
   ↓
MarketData object
   ↓
OIS discount curve
   ↓
SOFR projection curve:
   valuation-date DF = 1
   → 1M futures bootstrap
   → 3M futures bootstrap
   → swap bootstrap
   ↓
Repricing tables
   ↓
Diagnostics
   ↓
CurveBuildResult
'''
@dataclass(frozen=True)
class CurveBuildResult:
    config: EngineConfig
    discount_curve: DiscountCurve
    projection_curve: DiscountCurve
    mean_reversion: float
    sigma: float
    futures_repricing: pd.DataFrame
    swap_repricing: pd.DataFrame
    oos_swap_repricing: pd.DataFrame
    diagnostics: dict[str, float | bool]
    calibration_diagnostics: CalibrationDiagnostics | None = None


def load_market_data(
    project_root: str | Path | None = None,
    data_source: MarketDataSource | None = None,
) -> MarketData:
    config = load_engine_config(project_root)
    source = data_source or CsvMarketDataSource(config.data_dir)
    return source.load(config)


def build_discount_curve(valuation_date: date, ois_curve: pd.DataFrame) -> DiscountCurve:
    pillar_dates = [ensure_date(value) for value in ois_curve["end_date"]]
    zero_rates = [float(value) for value in ois_curve["zero_rate"]]
    return DiscountCurve.from_zero_rates(valuation_date, pillar_dates, zero_rates, label="ois_discount")


def bootstrap_sofr_short_end(valuation_date: date) -> dict[date, float]:
    return {ensure_date(valuation_date): 1.0}


def bootstrap_from_1m_futures(
    quotes: list[FuturesQuote],
    valuation_date: date,
    a: float,
    sigma: float,
    seed_dfs: dict[date, float] | None = None,
) -> dict[date, float]:
    dfs = dict(seed_dfs or bootstrap_sofr_short_end(valuation_date))
    valuation = ensure_date(valuation_date)
    for quote in sorted(quotes, key=lambda item: item.start_date):
        start = ensure_date(quote.start_date)
        end = ensure_date(quote.end_date)
        if start not in dfs:
            raise KeyError(f"Missing start discount factor for {quote.contract_code}: {start.isoformat()}")
        delta = yearfrac(start, end, "ACT/360")
        t_start = yearfrac(valuation, start, "ACT/365F")
        t_end = yearfrac(valuation, end, "ACT/365F")
        correction = convexity_1m(a, sigma, t_start, t_end)
        dfs[end] = dfs[start] * np.exp(-delta * (quote.implied_rate - correction))
    return dfs


def bootstrap_from_3m_futures(
    quotes: list[FuturesQuote],
    valuation_date: date,
    a: float,
    sigma: float,
    seed_dfs: dict[date, float],
) -> dict[date, float]:
    dfs = dict(seed_dfs)
    valuation = ensure_date(valuation_date)
    for quote in sorted(quotes, key=lambda item: item.start_date):
        start = ensure_date(quote.start_date)
        end = ensure_date(quote.end_date)
        if start not in dfs:
            raise KeyError(f"Missing start discount factor for {quote.contract_code}: {start.isoformat()}")
        tau = yearfrac(start, end, "ACT/360")
        t_start = yearfrac(valuation, start, "ACT/365F")
        t_end = yearfrac(valuation, end, "ACT/365F")
        correction = U_j_const_sigma(a, sigma, t_start, t_end)
        dfs[end] = dfs[start] / ((1.0 + tau * quote.implied_rate) * np.exp(correction))
    return dfs


def bootstrap_from_swaps(
    swap_quotes: list[SwapQuote],
    discount_curve: DiscountCurve,
    projection_dfs: dict[date, float],
    calendar: str,
    roll: str,
) -> dict[date, float]:
    dfs = dict(projection_dfs)
    for quote in sorted(swap_quotes, key=lambda item: tenor_to_months(item.tenor)):
        maturity = ensure_date(quote.end_date)
        if maturity in dfs:
            continue
        periods = build_periods(
            quote.start_date,
            maturity,
            quote.pay_freq,
            quote.day_count,
            calendar=calendar,
            roll=roll,
        )
        known_float = 0.0
        annuity = 0.0
        last_period = periods[-1]
        for period in periods[:-1]:
            if period.end_date not in dfs or period.start_date not in dfs:
                raise KeyError(f"Missing projection pillar needed to bootstrap {quote.tenor}: {period.end_date.isoformat()}")
            forward = (dfs[period.start_date] / dfs[period.end_date] - 1.0) / period.accrual_factor
            discount_factor = discount_curve.df(period.end_date)
            known_float += period.accrual_factor * discount_factor * forward
            annuity += period.accrual_factor * discount_factor

        last_discount = discount_curve.df(last_period.end_date)
        annuity += last_period.accrual_factor * last_discount
        previous_df = dfs[last_period.start_date]
        denominator = quote.fixed_rate * annuity - known_float + last_discount
        if denominator <= 0.0:
            raise ValueError(f"Invalid bootstrap denominator for swap {quote.tenor}.")
        dfs[last_period.end_date] = last_discount * previous_df / denominator
    return dfs


def _projection_curve_from_sigma(market: MarketData, sigma: float, a: float) -> DiscountCurve:
    valuation_date = market.config.market.valuation_date
    projection_dfs = bootstrap_sofr_short_end(valuation_date)
    projection_dfs = bootstrap_from_1m_futures(market.futures_1m, valuation_date, a=a, sigma=sigma, seed_dfs=projection_dfs)
    projection_dfs = bootstrap_from_3m_futures(market.futures_3m, valuation_date, a=a, sigma=sigma, seed_dfs=projection_dfs)
    discount_curve = build_discount_curve(valuation_date, market.ois_curve)
    projection_dfs = bootstrap_from_swaps(
        market.swaps,
        discount_curve,
        projection_dfs,
        calendar=market.config.market.calendar,
        roll=market.config.market.business_day_roll,
    )
    dates = sorted(projection_dfs)
    dfs = [projection_dfs[item] for item in dates]
    return DiscountCurve(valuation_date, dates, dfs, label="sofr_projection")


def sofr_curve_smoothness_objective(market: MarketData, sigma: float, a: float) -> float:
    """Roughness of the bootstrapped SOFR projection curve for a constant HW sigma.

    This implements the no-OIS-option-data alternative described by Mercurio:
    fix the mean reversion ``a`` and choose ``sigma`` to make the SOFR curve
    built from futures and swaps as smooth as possible.
    """

    if sigma < 0.0:
        return 1e12
    try:
        curve = _projection_curve_from_sigma(market, sigma=sigma, a=a)
    except Exception:
        return 1e12

    times = np.asarray(curve.times, dtype=float)
    dfs = np.asarray(curve.dfs, dtype=float)
    if times.size < 4 or np.any(dfs <= 0.0) or np.any(np.diff(times) <= 0.0):
        return 1e12

    log_dfs = np.log(dfs)
    forwards = -np.diff(log_dfs) / np.diff(times)
    if not np.all(np.isfinite(forwards)):
        return 1e12

    # Penalize curvature of adjacent piecewise-constant forwards. Scaling by
    # interval length keeps long sparse swap intervals from dominating solely
    # because they are farther apart than the futures intervals.
    midpoint_times = 0.5 * (times[:-1] + times[1:])
    forward_jumps = np.diff(forwards)
    midpoint_gaps = np.diff(midpoint_times)
    curvature = forward_jumps / np.maximum(midpoint_gaps, 1e-12)
    objective = float(np.mean(curvature**2))

    monotonicity_penalty = float(np.sum(np.maximum(np.diff(dfs), 0.0) ** 2)) * 1e8
    return objective + monotonicity_penalty


_SMOOTHNESS_PENALTY = 1e12


def _validate_smoothness_proxy_inputs(
    market: MarketData,
    a: float,
    sigma_bounds: tuple[float, float],
) -> tuple[float, float]:
    if not math.isfinite(float(a)) or float(a) < 0.0:
        raise ValueError("Mean reversion must be finite and non-negative.")
    if len(sigma_bounds) != 2:
        raise ValueError("Sigma bounds must contain exactly two values.")
    lower, upper = float(sigma_bounds[0]), float(sigma_bounds[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower < 0.0 or upper <= lower:
        raise ValueError("Sigma bounds must be finite and satisfy 0 <= lower < upper.")
    if not market.futures_1m and not market.futures_3m:
        raise ValueError("Smoothness regularization requires at least one SOFR futures quote.")
    if not market.swaps:
        raise ValueError("Smoothness regularization requires at least one SOFR swap quote.")
    if market.ois_curve.empty:
        raise ValueError("Smoothness regularization requires a non-empty OIS curve.")

    numeric_inputs = [quote.price for quote in market.futures_1m + market.futures_3m]
    numeric_inputs.extend(quote.fixed_rate for quote in market.swaps)
    numeric_inputs.extend(float(value) for value in market.ois_curve["zero_rate"])
    if any(not math.isfinite(float(value)) for value in numeric_inputs):
        raise ValueError("Smoothness regularization inputs must all be finite.")
    return lower, upper


def calibrate_sigma_regularization_proxy(
    market: MarketData,
    a: float,
    sigma_bounds: tuple[float, float],
) -> tuple[float, CalibrationDiagnostics]:
    """Choose sigma as a curve-smoothness regularization proxy.

    This is deliberately not described as an option-implied volatility
    calibration. A boundary solution is valid optimizer output but is surfaced
    explicitly because it indicates weak identification by curve smoothness.
    """

    lower, upper = _validate_smoothness_proxy_inputs(market, a, sigma_bounds)

    result = minimize_scalar(
        lambda sigma: sofr_curve_smoothness_objective(market, sigma=float(sigma), a=a),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )
    sigma = float(getattr(result, "x", float("nan")))
    objective = float(getattr(result, "fun", float("nan")))
    if not bool(getattr(result, "success", False)):
        raise RuntimeError(
            f"SOFR smoothness regularization failed: {getattr(result, 'message', 'unknown optimizer error')}"
        )
    if not math.isfinite(sigma) or not lower <= sigma <= upper:
        raise RuntimeError("SOFR smoothness regularization returned an invalid sigma.")
    if not math.isfinite(objective) or objective < 0.0 or objective >= _SMOOTHNESS_PENALTY:
        raise RuntimeError("SOFR smoothness regularization returned an invalid objective value.")

    parameter_values = {"sigma": sigma, "a_fixed": float(a)}
    parameter_bounds = {"sigma": (lower, upper)}
    hits = boundary_hits({"sigma": sigma}, parameter_bounds)
    message = str(getattr(result, "message", "optimizer converged"))
    if hits.get("sigma") == "lower":
        message += "; sigma is pinned to the lower bound and is a regularization proxy, not option-implied volatility"
    diagnostics = CalibrationDiagnostics(
        method="sofr_curve_smoothness_regularization_proxy",
        converged=True,
        objective_value=objective,
        message=message,
        iterations=int(result.nit) if getattr(result, "nit", None) is not None else None,
        function_evaluations=int(result.nfev) if getattr(result, "nfev", None) is not None else None,
        parameter_values=parameter_values,
        parameter_bounds=parameter_bounds,
        boundary_hits=hits,
        is_regularization_proxy=True,
    )
    return sigma, diagnostics


def calibrate_sigma_from_sofr_curve_smoothness(
    market: MarketData,
    a: float,
    sigma_bounds: tuple[float, float],
) -> float:
    """Compatibility wrapper returning only the smoothness-proxy sigma."""

    sigma, _ = calibrate_sigma_regularization_proxy(market, a, sigma_bounds)
    return sigma


def build_projection_curve(
    project_root: str | Path | None = None,
    sigma_override: float | None = None,
    mean_reversion_override: float | None = None,
    sigma_calibration_method: str | None = None,
    data_source: MarketDataSource | None = None,
) -> DiscountCurve:
    market = load_market_data(project_root, data_source=data_source)
    method = (
        sigma_calibration_method
        if sigma_calibration_method is not None
        else market.config.model.sigma_calibration_method
    )
    mean_reversion, sigma, _ = _resolve_model_parameters(
        market,
        sigma_override=sigma_override,
        mean_reversion_override=mean_reversion_override,
        should_calibrate=market.config.model.calibrate_sigma,
        method=method,
    )
    return _projection_curve_from_sigma(market, sigma=sigma, a=mean_reversion)


def _build_swaption_quotes(
    discount_curve: DiscountCurve,
    surface: pd.DataFrame,
) -> list[SwaptionQuote]:
    """Build SwaptionQuote objects from a swaption vol surface DataFrame.

    Assumes annual payment frequency and OIS-implied ATM rates.
    """
    required_columns = {"expiry_y", "tenor_y", "atm_normal_vol"}
    missing = sorted(required_columns - set(surface.columns))
    if missing:
        raise ValueError(f"Swaption surface missing required columns: {', '.join(missing)}")
    if surface.empty:
        raise ValueError("Swaption surface must contain at least one quote.")
    for column in sorted(required_columns):
        values = pd.to_numeric(surface[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Swaption surface column {column!r} must contain only finite numbers.")
    if (surface["expiry_y"].astype(float) <= 0.0).any():
        raise ValueError("Swaption expiries must be positive.")
    if (surface["tenor_y"].astype(float) <= 0.0).any():
        raise ValueError("Swaption tenors must be positive.")
    if (surface["atm_normal_vol"].astype(float) < 0.0).any():
        raise ValueError("Swaption normal volatilities cannot be negative.")

    quotes: list[SwaptionQuote] = []
    for _, row in surface.iterrows():
        T_exp = float(row["expiry_y"])
        tenor = float(row["tenor_y"])
        n = int(round(tenor))
        if n < 1 or not math.isclose(tenor, float(n), rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("The annual swaption schedule currently requires whole-year tenors.")
        payment_times = [T_exp + k for k in range(1, n + 1)]
        tau_i = [1.0] * n
        annuity = sum(discount_curve.df(T) for T in payment_times)
        if not math.isfinite(annuity) or annuity <= 0.0:
            raise ValueError("Swaption annuity must be finite and positive.")
        fixed_rate = (discount_curve.df(T_exp) - discount_curve.df(T_exp + tenor)) / annuity
        quotes.append(
            SwaptionQuote(
                T_exp=T_exp,
                payment_times=payment_times,
                tau_i=tau_i,
                fixed_rate=fixed_rate,
                market_normal_vol=float(row["atm_normal_vol"]),
            )
        )
    return quotes


def calibrate_hw_from_surface_with_diagnostics(
    discount_curve: DiscountCurve,
    swaption_vols_path: str,
    a_init: float = 0.05,
    sigma_init: float = 0.01,
    a_bounds: tuple[float, float] = (1e-4, 1.0),
    sigma_bounds: tuple[float, float] = (1e-5, 0.10),
) -> tuple[float, float, CalibrationDiagnostics]:
    """Load a validated swaption surface and calibrate HW with diagnostics."""

    surface = pd.read_csv(swaption_vols_path)
    quotes = _build_swaption_quotes(discount_curve, surface)
    return calibrate_hw_with_diagnostics(
        quotes,
        discount_curve,
        a_init=a_init,
        sigma_init=sigma_init,
        a_bounds=a_bounds,
        sigma_bounds=sigma_bounds,
    )


def calibrate_hw_from_surface(
    discount_curve: DiscountCurve,
    swaption_vols_path: str,
    a_init: float = 0.05,
    sigma_init: float = 0.01,
    a_bounds: tuple[float, float] = (1e-4, 1.0),
    sigma_bounds: tuple[float, float] = (1e-5, 0.10),
) -> tuple[float, float]:
    """Compatibility wrapper returning only calibrated ``(a, sigma)``."""

    a, sigma, _ = calibrate_hw_from_surface_with_diagnostics(
        discount_curve,
        swaption_vols_path,
        a_init=a_init,
        sigma_init=sigma_init,
        a_bounds=a_bounds,
        sigma_bounds=sigma_bounds,
    )
    return a, sigma


def _resolve_model_parameters(
    market: MarketData,
    *,
    sigma_override: float | None,
    mean_reversion_override: float | None,
    should_calibrate: bool,
    method: str,
) -> tuple[float, float, CalibrationDiagnostics]:
    a = float(mean_reversion_override) if mean_reversion_override is not None else float(market.config.model.mean_reversion)
    sigma = float(sigma_override) if sigma_override is not None else float(market.config.model.sigma)
    if not math.isfinite(a) or a < 0.0:
        raise ValueError("Mean reversion must be finite and non-negative.")
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("Sigma must be finite and non-negative.")

    parameter_values = {"a": a, "sigma": sigma}
    if sigma_override is not None:
        return a, sigma, configured_diagnostics(
            method="explicit_override",
            parameter_values=parameter_values,
            message="Model parameters include an explicit sigma override; no optimizer was run.",
        )
    if not should_calibrate:
        return a, sigma, configured_diagnostics(
            method="configured_parameters",
            parameter_values=parameter_values,
            message="Configured model parameters were used; calibration is disabled.",
        )

    normalized_method = method.lower().replace("-", "_").replace(" ", "_")
    if normalized_method in {"swaption_surface", "swaption", "ois_option", "ois_options"}:
        discount_curve = build_discount_curve(market.config.market.valuation_date, market.ois_curve)
        swaption_path = str(market.config.data_dir / "market" / market.config.model.swaption_vols_file)
        return calibrate_hw_from_surface_with_diagnostics(
            discount_curve,
            swaption_path,
            a_init=a,
            sigma_init=sigma,
            a_bounds=market.config.model.a_bounds,
            sigma_bounds=market.config.model.sigma_bounds,
        )
    if normalized_method in {
        "sofr_curve_smoothness",
        "curve_smoothness",
        "smoothness",
        "sofr_curve_smoothness_regularization_proxy",
    }:
        sigma, diagnostics = calibrate_sigma_regularization_proxy(
            market,
            a=a,
            sigma_bounds=market.config.model.sigma_bounds,
        )
        return a, sigma, diagnostics
    raise ValueError(f"Unsupported sigma calibration method: {method}")


def futures_repricing_table(
    projection_curve: DiscountCurve,
    valuation_date: date,
    one_m_quotes: list[FuturesQuote],
    three_m_quotes: list[FuturesQuote],
    a: float,
    sigma: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    valuation = ensure_date(valuation_date)
    for quote in one_m_quotes:
        delta = yearfrac(quote.start_date, quote.end_date, "ACT/360")
        t_start = yearfrac(valuation, quote.start_date, "ACT/365F")
        t_end = yearfrac(valuation, quote.end_date, "ACT/365F")
        forward = -np.log(projection_curve.df(quote.end_date) / projection_curve.df(quote.start_date)) / delta
        model_rate = forward + convexity_1m(a, sigma, t_start, t_end)
        rows.append(
            {
                "Instrument": quote.contract_code,
                "Type": "SOFR_1M",
                "Market Rate": quote.implied_rate,
                "Model Rate": model_rate,
                "Error (bp)": annualize_bp(model_rate - quote.implied_rate),
            }
        )

    for quote in three_m_quotes:
        tau = yearfrac(quote.start_date, quote.end_date, "ACT/360")
        t_start = yearfrac(valuation, quote.start_date, "ACT/365F")
        t_end = yearfrac(valuation, quote.end_date, "ACT/365F")
        correction = U_j_const_sigma(a, sigma, t_start, t_end)
        model_rate = (projection_curve.df(quote.start_date) / (projection_curve.df(quote.end_date) * np.exp(correction)) - 1.0) / tau
        rows.append(
            {
                "Instrument": quote.contract_code,
                "Type": "SOFR_3M",
                "Market Rate": quote.implied_rate,
                "Model Rate": model_rate,
                "Error (bp)": annualize_bp(model_rate - quote.implied_rate),
            }
        )

    return pd.DataFrame(rows)


def swap_repricing_table(
    market: MarketData,
    discount_curve: DiscountCurve,
    projection_curve: DiscountCurve,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for quote in market.swaps:
        periods = build_periods(
            quote.start_date,
            quote.end_date,
            quote.pay_freq,
            quote.day_count,
            calendar=market.config.market.calendar,
            roll=market.config.market.business_day_roll,
        )
        model_rate = par_swap_rate(discount_curve, projection_curve, periods)
        rows.append(
            {
                "Tenor": quote.tenor,
                "Market Rate": quote.fixed_rate,
                "Model Rate": model_rate,
                "Error (bp)": annualize_bp(model_rate - quote.fixed_rate),
            }
        )
    return pd.DataFrame(rows)


def swap_holdout_repricing_table(
    market: MarketData,
    a: float,
    sigma: float,
    holdout_tenors: list[str] | None = None,
    perturbation_bp: float = 1.0,
) -> pd.DataFrame:
    """Reprice held-out swaps and perturb training quotes for stability checks."""

    holdouts = holdout_tenors or [market.swaps[-1].tenor]
    heldout_swaps = [quote for quote in market.swaps if quote.tenor in set(holdouts)]
    training_swaps = [quote for quote in market.swaps if quote.tenor not in set(holdouts)]
    if not heldout_swaps or not training_swaps:
        return pd.DataFrame()

    rows: list[dict[str, float | str]] = []
    discount_curve = build_discount_curve(market.config.market.valuation_date, market.ois_curve)
    base_market = replace(market, swaps=training_swaps)
    base_curve = _projection_curve_from_sigma(base_market, sigma=sigma, a=a)
    base_errors: dict[str, float] = {}

    for shock_bp in [0.0, -perturbation_bp, perturbation_bp]:
        if shock_bp == 0.0:
            projection_curve = base_curve
        else:
            shifted_swaps = [
                replace(quote, fixed_rate=quote.fixed_rate + shock_bp / 10000.0)
                for quote in training_swaps
            ]
            projection_curve = _projection_curve_from_sigma(
                replace(base_market, swaps=shifted_swaps),
                sigma=sigma,
                a=a,
            )
        for quote in heldout_swaps:
            periods = build_periods(
                quote.start_date,
                quote.end_date,
                quote.pay_freq,
                quote.day_count,
                calendar=market.config.market.calendar,
                roll=market.config.market.business_day_roll,
            )
            model_rate = par_swap_rate(discount_curve, projection_curve, periods)
            error_bp = annualize_bp(model_rate - quote.fixed_rate)
            if shock_bp == 0.0:
                base_errors[quote.tenor] = error_bp
            rows.append(
                {
                    "Tenor": quote.tenor,
                    "Validation Type": "held_out_swap",
                    "Training Quote Shock (bp)": shock_bp,
                    "Market Rate": quote.fixed_rate,
                    "Model Rate": model_rate,
                    "Error (bp)": error_bp,
                    "Error Change vs Base (bp)": error_bp - base_errors.get(quote.tenor, error_bp),
                }
            )

    return pd.DataFrame(rows)


def diagnostics_summary(
    discount_curve: DiscountCurve,
    projection_curve: DiscountCurve,
    futures_table: pd.DataFrame,
    swap_table: pd.DataFrame,
    oos_swap_table: pd.DataFrame | None = None,
) -> dict[str, float | bool]:
    projection_dfs = projection_curve.dfs
    discount_dfs = discount_curve.dfs
    diagnostics = {
        "avg_futures_error_bp": float(futures_table["Error (bp)"].abs().mean()),
        "max_futures_error_bp": float(futures_table["Error (bp)"].abs().max()),
        "avg_swap_error_bp": float(swap_table["Error (bp)"].abs().mean()),
        "max_swap_error_bp": float(swap_table["Error (bp)"].abs().max()),
        "projection_monotone": bool(np.all(np.diff(projection_dfs) <= 1e-12)),
        "discount_monotone": bool(np.all(np.diff(discount_dfs) <= 1e-12)),
        "positive_projection_dfs": bool(np.all(projection_dfs > 0.0)),
        "positive_discount_dfs": bool(np.all(discount_dfs > 0.0)),
    }
    if oos_swap_table is not None and not oos_swap_table.empty:
        base = oos_swap_table[oos_swap_table["Training Quote Shock (bp)"] == 0.0]
        shocked = oos_swap_table[oos_swap_table["Training Quote Shock (bp)"] != 0.0]
        diagnostics.update(
            {
                "oos_swap_mae_bp": float(base["Error (bp)"].abs().mean()),
                "oos_swap_max_error_bp": float(base["Error (bp)"].abs().max()),
                "oos_swap_max_1bp_shock_error_change_bp": float(
                    shocked["Error Change vs Base (bp)"].abs().max()
                ) if not shocked.empty else 0.0,
            }
        )
    return diagnostics


def build_full_curves(
    project_root: str | Path | None = None,
    calibrate_sigma_override: bool | None = None,
    sigma_override: float | None = None,
    mean_reversion_override: float | None = None,
    sigma_calibration_method: str | None = None,
    data_source: MarketDataSource | None = None,
    holdout_swap_tenors: list[str] | None = None,
) -> CurveBuildResult:
    market = load_market_data(project_root, data_source=data_source)
    valuation_date = market.config.market.valuation_date
    discount_curve = build_discount_curve(valuation_date, market.ois_curve)
    should_calibrate = calibrate_sigma_override if calibrate_sigma_override is not None else market.config.model.calibrate_sigma
    method = (
        sigma_calibration_method
        if sigma_calibration_method is not None
        else market.config.model.sigma_calibration_method
    )
    a, sigma, calibration_diagnostics = _resolve_model_parameters(
        market,
        sigma_override=sigma_override,
        mean_reversion_override=mean_reversion_override,
        should_calibrate=bool(should_calibrate),
        method=method,
    )
    projection_curve = _projection_curve_from_sigma(market, sigma=sigma, a=a)
    futures_table = futures_repricing_table(
        projection_curve,
        valuation_date,
        market.futures_1m,
        market.futures_3m,
        a=a,
        sigma=sigma,
    )
    swap_table = swap_repricing_table(market, discount_curve, projection_curve)
    oos_swap_table = swap_holdout_repricing_table(
        market,
        a=a,
        sigma=sigma,
        holdout_tenors=holdout_swap_tenors,
    ) if holdout_swap_tenors is not None else pd.DataFrame()
    diagnostics = diagnostics_summary(
        discount_curve,
        projection_curve,
        futures_table,
        swap_table,
        oos_swap_table,
    )
    return CurveBuildResult(
        config=market.config,
        discount_curve=discount_curve,
        projection_curve=projection_curve,
        mean_reversion=a,
        sigma=sigma,
        futures_repricing=futures_table,
        swap_repricing=swap_table,
        oos_swap_repricing=oos_swap_table,
        diagnostics=diagnostics,
        calibration_diagnostics=calibration_diagnostics,
    )
