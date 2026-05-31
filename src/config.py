from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .utils import ensure_date, project_root_from_here


@dataclass(frozen=True)
class ModelConfig:
    mean_reversion: float
    sigma: float
    sigma_bounds: tuple[float, float]
    a_bounds: tuple[float, float]
    calibrate_sigma: bool
    swaption_vols_file: str


@dataclass(frozen=True)
class JointModelConfig:
    libor_calibration_file: str
    basis_calibration_tenor_years: float
    default_shift: float
    default_correlation_to_ois: float


@dataclass(frozen=True)
class MarketConfig:
    valuation_date: date
    calendar: str
    business_day_roll: str
    futures_day_count: str
    swap_fixed_day_count: str
    swap_float_day_count: str
    swap_pay_freq: str
    synthetic_data_default: bool


@dataclass(frozen=True)
class RiskConfig:
    scenario_bump_bp: float
    key_rates_years: tuple[float, ...]


@dataclass(frozen=True)
class EngineConfig:
    project_root: Path
    data_dir: Path
    output_dir: Path
    market: MarketConfig
    model: ModelConfig
    joint_model: JointModelConfig
    risk: RiskConfig


def load_engine_config(project_root: str | Path | None = None) -> EngineConfig:
    root = Path(project_root).resolve() if project_root else project_root_from_here(__file__)
    config_path = root / "data" / "metadata" / "conventions.yaml"
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    market_section = raw["market"]
    model_section = raw["model"]
    joint_model_section = raw.get("joint_model", {})
    risk_section = raw["risk"]

    market = MarketConfig(
        valuation_date=ensure_date(market_section["valuation_date"]),
        calendar=market_section["calendar"],
        business_day_roll=market_section["business_day_roll"],
        futures_day_count=market_section["futures_day_count"],
        swap_fixed_day_count=market_section["swap_fixed_day_count"],
        swap_float_day_count=market_section["swap_float_day_count"],
        swap_pay_freq=market_section["swap_pay_freq"],
        synthetic_data_default=bool(raw["project"]["synthetic_data_default"]),
    )
    model = ModelConfig(
        mean_reversion=float(model_section["mean_reversion"]),
        sigma=float(model_section["sigma"]),
        sigma_bounds=tuple(float(value) for value in model_section["sigma_bounds"]),
        a_bounds=tuple(float(value) for value in model_section.get("a_bounds", [0.0001, 1.0])),
        calibrate_sigma=bool(model_section["calibrate_sigma"]),
        swaption_vols_file=str(model_section.get("swaption_vols_file", "swaption_vols.csv")),
    )
    joint_model = JointModelConfig(
        libor_calibration_file=str(joint_model_section.get("libor_calibration_file", "libor_lmm_calibration.csv")),
        basis_calibration_tenor_years=float(joint_model_section.get("basis_calibration_tenor_years", 5.0)),
        default_shift=float(joint_model_section.get("default_shift", 0.02)),
        default_correlation_to_ois=float(joint_model_section.get("default_correlation_to_ois", 0.65)),
    )
    risk = RiskConfig(
        scenario_bump_bp=float(risk_section["scenario_bump_bp"]),
        key_rates_years=tuple(float(value) for value in risk_section["key_rates_years"]),
    )
    return EngineConfig(
        project_root=root,
        data_dir=root / "data",
        output_dir=root / "outputs",
        market=market,
        model=model,
        joint_model=joint_model,
        risk=risk,
    )
