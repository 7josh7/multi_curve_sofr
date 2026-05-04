from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .bootstrap import CurveBuildResult, build_full_curves, load_market_data
from .config import EngineConfig, load_engine_config
from .curves import DiscountCurve
from .daycount import yearfrac
from .hw_model import B
from .utils import ensure_date


@dataclass(frozen=True)
class LiborCalibrationPoint:
    tenor: str
    start_date: object
    end_date: object
    option_maturity: float
    end_time: float
    accrual_factor: float
    libor_forward: float
    ois_forward: float
    sofr_forward: float
    shift: float
    atm_caplet_vol: float
    correlation_to_ois: float
    shifted_lmm_vol: float
    multiplicative_basis: float
    libor_q_drift: float
    basis_optimal_hw_sigma: float
    basis_vol_at_curve_sigma: float
    basis_vol_at_optimal_sigma: float


@dataclass(frozen=True)
class JointModelCalibrationResult:
    config: EngineConfig
    curve_build: CurveBuildResult
    calibration_table: pd.DataFrame
    target_row: dict[str, float | str]
    diagnostics: dict[str, float | str | bool]


def load_libor_calibration_data(project_root: str | Path | None = None) -> pd.DataFrame:
    config = load_engine_config(project_root)
    path = config.data_dir / "market" / config.joint_model.libor_calibration_file
    frame = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    frame["start_date"] = frame["start_date"].dt.date
    frame["end_date"] = frame["end_date"].dt.date
    return frame


def shifted_lmm_vol_from_atm(
    libor_forward: float,
    shift: float,
    atm_caplet_vol: float,
    option_maturity: float,
) -> float:
    """Mercurio eq. 15 constant-vol calibration for shifted-lognormal LIBOR."""

    maturity = float(option_maturity)
    if maturity <= 1e-12:
        return float(atm_caplet_vol)
    forward = float(libor_forward)
    displacement = float(shift)
    if forward + displacement <= 0.0:
        raise ValueError("LIBOR forward plus shift must be positive.")

    sqrt_maturity = math.sqrt(maturity)
    black_term = 2.0 * norm.cdf(0.5 * float(atm_caplet_vol) * sqrt_maturity) - 1.0
    probability = 0.5 + forward / (2.0 * (forward + displacement)) * black_term
    clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
    return float(2.0 / sqrt_maturity * norm.ppf(clipped))


def multiplicative_basis(libor_forward: float, ois_forward: float, accrual_factor: float) -> float:
    """Mercurio multiplicative LIBOR-OIS basis B_j."""

    tau = float(accrual_factor)
    return float((float(libor_forward) - float(ois_forward)) / (1.0 + tau * float(ois_forward)))


def libor_drift_under_q(
    libor_forward: float,
    shift: float,
    shifted_lmm_vol: float,
    hw_sigma: float,
    correlation_to_ois: float,
    mean_reversion: float,
    maturity: float,
) -> float:
    """Mercurio eq. 13 drift of L_j under the OIS risk-neutral measure Q."""

    return float(
        correlation_to_ois
        * shifted_lmm_vol
        * hw_sigma
        * B(mean_reversion, 0.0, maturity)
        * (libor_forward + shift)
    )


def basis_instantaneous_vol(
    libor_forward: float,
    ois_forward: float,
    shift: float,
    shifted_lmm_vol: float,
    hw_sigma: float,
    correlation_to_ois: float,
    mean_reversion: float,
    start_time: float,
    end_time: float,
    accrual_factor: float,
) -> float:
    """Instantaneous volatility of Mercurio's multiplicative LIBOR-OIS basis."""

    tau = float(accrual_factor)
    basis = multiplicative_basis(libor_forward, ois_forward, tau)
    libor_loading = shifted_lmm_vol * (libor_forward + shift) / (libor_forward + 1.0 / tau)
    ois_loading = (B(mean_reversion, 0.0, end_time) - B(mean_reversion, 0.0, start_time)) * hw_sigma
    variance = libor_loading**2 + ois_loading**2 - 2.0 * correlation_to_ois * libor_loading * ois_loading
    return float((basis + 1.0 / tau) * math.sqrt(max(variance, 0.0)))


def hw_sigma_from_basis_minimization(
    libor_forward: float,
    shift: float,
    shifted_lmm_vol: float,
    correlation_to_ois: float,
    mean_reversion: float,
    start_time: float,
    end_time: float,
    accrual_factor: float,
) -> float:
    """Mercurio eq. 18 constant-sigma basis-vol minimizer."""

    b_diff = B(mean_reversion, 0.0, end_time) - B(mean_reversion, 0.0, start_time)
    if abs(b_diff) < 1e-12:
        raise ValueError("Degenerate Hull-White B difference for basis calibration.")
    tau = float(accrual_factor)
    return float(
        correlation_to_ois
        * shifted_lmm_vol
        / b_diff
        * (libor_forward + shift)
        / (libor_forward + 1.0 / tau)
    )


def build_joint_model_calibration(
    project_root: str | Path | None = None,
    basis_calibration_tenor_years: float | None = None,
    sigma_override: float | None = None,
) -> JointModelCalibrationResult:
    config = load_engine_config(project_root)
    curve_build = build_full_curves(project_root=project_root, sigma_override=sigma_override)
    libor_data = load_libor_calibration_data(project_root)
    target_tenor = (
        float(basis_calibration_tenor_years)
        if basis_calibration_tenor_years is not None
        else config.joint_model.basis_calibration_tenor_years
    )

    rows: list[dict[str, float | str]] = []
    for raw in libor_data.itertuples(index=False):
        point = _calibrate_row(
            config=config,
            discount_curve=curve_build.discount_curve,
            projection_curve=curve_build.projection_curve,
            raw=raw,
            hw_sigma=float(curve_build.sigma),
        )
        rows.append(point.__dict__)

    table = pd.DataFrame(rows)
    target_index = (table["end_time"] - target_tenor).abs().idxmin()
    target_row = table.loc[target_index].to_dict()
    diagnostics = {
        "model_scope": "OIS/SOFR Hull-White + shifted-lognormal forward LIBOR calibration",
        "basis_calibration_tenor_years": float(target_tenor),
        "curve_hw_sigma": float(curve_build.sigma),
        "basis_implied_hw_sigma": float(target_row["basis_optimal_hw_sigma"]),
        "mean_shifted_lmm_vol": float(table["shifted_lmm_vol"].mean()),
        "max_shifted_lmm_vol": float(table["shifted_lmm_vol"].max()),
        "mean_basis_vol_at_curve_sigma": float(table["basis_vol_at_curve_sigma"].mean()),
        "mean_basis_vol_at_optimal_sigma": float(table["basis_vol_at_optimal_sigma"].mean()),
        "synthetic_calibration_rows": int(len(table)),
    }
    return JointModelCalibrationResult(
        config=config,
        curve_build=curve_build,
        calibration_table=table,
        target_row=target_row,
        diagnostics=diagnostics,
    )


def _calibrate_row(
    config: EngineConfig,
    discount_curve: DiscountCurve,
    projection_curve: DiscountCurve,
    raw,
    hw_sigma: float,
) -> LiborCalibrationPoint:
    start = ensure_date(raw.start_date)
    end = ensure_date(raw.end_date)
    valuation = config.market.valuation_date
    option_maturity = yearfrac(valuation, start, "ACT/365F")
    end_time = yearfrac(valuation, end, "ACT/365F")
    tau = yearfrac(start, end, "ACT/360")
    libor_forward = float(raw.libor_forward)
    shift = float(getattr(raw, "shift", config.joint_model.default_shift))
    correlation = float(getattr(raw, "correlation_to_ois", config.joint_model.default_correlation_to_ois))
    atm_vol = float(raw.atm_caplet_vol)
    lmm_vol = shifted_lmm_vol_from_atm(libor_forward, shift, atm_vol, option_maturity)
    ois_forward = discount_curve.forward_rate(start, end, day_count="ACT/360")
    sofr_forward = projection_curve.forward_rate(start, end, day_count="ACT/360")
    basis = multiplicative_basis(libor_forward, ois_forward, tau)
    optimal_sigma = hw_sigma_from_basis_minimization(
        libor_forward,
        shift,
        lmm_vol,
        correlation,
        config.model.mean_reversion,
        option_maturity,
        end_time,
        tau,
    )
    basis_vol_curve = basis_instantaneous_vol(
        libor_forward,
        ois_forward,
        shift,
        lmm_vol,
        hw_sigma,
        correlation,
        config.model.mean_reversion,
        option_maturity,
        end_time,
        tau,
    )
    basis_vol_optimal = basis_instantaneous_vol(
        libor_forward,
        ois_forward,
        shift,
        lmm_vol,
        optimal_sigma,
        correlation,
        config.model.mean_reversion,
        option_maturity,
        end_time,
        tau,
    )
    drift = libor_drift_under_q(
        libor_forward,
        shift,
        lmm_vol,
        hw_sigma,
        correlation,
        config.model.mean_reversion,
        end_time,
    )
    return LiborCalibrationPoint(
        tenor=str(raw.tenor),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        option_maturity=float(option_maturity),
        end_time=float(end_time),
        accrual_factor=float(tau),
        libor_forward=libor_forward,
        ois_forward=float(ois_forward),
        sofr_forward=float(sofr_forward),
        shift=shift,
        atm_caplet_vol=atm_vol,
        correlation_to_ois=correlation,
        shifted_lmm_vol=float(lmm_vol),
        multiplicative_basis=float(basis),
        libor_q_drift=float(drift),
        basis_optimal_hw_sigma=float(optimal_sigma),
        basis_vol_at_curve_sigma=float(basis_vol_curve),
        basis_vol_at_optimal_sigma=float(basis_vol_optimal),
    )
