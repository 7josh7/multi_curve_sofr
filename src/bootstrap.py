from __future__ import annotations #postpones evaluation of type hints

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .config import EngineConfig, load_engine_config
from .curves import DiscountCurve
from .data_input import CsvMarketDataSource, MarketData, MarketDataSource
from .daycount import yearfrac
from .hw_model import U_j_const_sigma, convexity_1m
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
    sigma: float
    futures_repricing: pd.DataFrame
    swap_repricing: pd.DataFrame
    oos_swap_repricing: pd.DataFrame
    diagnostics: dict[str, float | bool]


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


def build_projection_curve(
    project_root: str | Path | None = None,
    sigma_override: float | None = None,
    mean_reversion_override: float | None = None,
    data_source: MarketDataSource | None = None,
) -> DiscountCurve:
    market = load_market_data(project_root, data_source=data_source)
    sigma = float(sigma_override) if sigma_override is not None else market.config.model.sigma
    mean_reversion = (
        float(mean_reversion_override) if mean_reversion_override is not None else market.config.model.mean_reversion
    )
    return _projection_curve_from_sigma(market, sigma=sigma, a=mean_reversion)


def calibrate_sigma_value(
    market: MarketData,
    a: float,
    sigma_bounds: tuple[float, float],
) -> float:
    def objective(sigma: float) -> float:
        curve = _projection_curve_from_sigma(market, sigma=sigma, a=a)
        zeros = np.array(curve.pillar_zero_rates())
        second_differences = np.diff(zeros, n=2)
        return float(np.sum(second_differences**2))

    result = minimize_scalar(objective, bounds=sigma_bounds, method="bounded")
    return float(result.x)


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
    data_source: MarketDataSource | None = None,
    holdout_swap_tenors: list[str] | None = None,
) -> CurveBuildResult:
    market = load_market_data(project_root, data_source=data_source)
    valuation_date = market.config.market.valuation_date
    discount_curve = build_discount_curve(valuation_date, market.ois_curve)
    a = float(mean_reversion_override) if mean_reversion_override is not None else market.config.model.mean_reversion
    sigma = float(sigma_override) if sigma_override is not None else market.config.model.sigma
    should_calibrate = calibrate_sigma_override if calibrate_sigma_override is not None else market.config.model.calibrate_sigma
    if should_calibrate and sigma_override is None:
        sigma = calibrate_sigma_value(market, a=a, sigma_bounds=market.config.model.sigma_bounds)
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
        sigma=sigma,
        futures_repricing=futures_table,
        swap_repricing=swap_table,
        oos_swap_repricing=oos_swap_table,
        diagnostics=diagnostics,
    )
