"""Collateral-aware multi-curve SOFR engine."""

__version__ = "0.2.0"

from .bootstrap import (
    CurveBuildResult,
    build_full_curves,
    build_projection_curve,
    calibrate_sigma_from_sofr_curve_smoothness,
    calibrate_sigma_regularization_proxy,
    load_market_data,
    sofr_curve_smoothness_objective,
)
from .calibration import CalibrationDiagnostics
from .curves import DiscountCurve, ForwardCurve
from .data_input import (
    CsvLiborCalibrationDataSource,
    CsvMarketDataSource,
    DataInputError,
    DataSchemaError,
    DataValidationError,
    LiborCalibrationDataSource,
    MarketData,
    MarketDataSource,
)
from .export import build_market_snapshot, export_market_snapshot, validate_market_snapshot, write_market_snapshot
from .libor_model import JointModelCalibrationResult, build_joint_model_calibration
from .libor_pricers import pv_libor_sofr_basis_swap_to_libor_payer, pv_libor_swap_with_sofr_fallback
from .pricers import par_swap_rate, pv_swap

__all__ = [
    "__version__",
    "CurveBuildResult",
    "CalibrationDiagnostics",
    "CsvLiborCalibrationDataSource",
    "CsvMarketDataSource",
    "DataInputError",
    "DataSchemaError",
    "DataValidationError",
    "DiscountCurve",
    "ForwardCurve",
    "JointModelCalibrationResult",
    "LiborCalibrationDataSource",
    "MarketData",
    "MarketDataSource",
    "build_full_curves",
    "build_market_snapshot",
    "build_joint_model_calibration",
    "build_projection_curve",
    "calibrate_sigma_from_sofr_curve_smoothness",
    "calibrate_sigma_regularization_proxy",
    "export_market_snapshot",
    "load_market_data",
    "par_swap_rate",
    "pv_libor_sofr_basis_swap_to_libor_payer",
    "pv_libor_swap_with_sofr_fallback",
    "pv_swap",
    "sofr_curve_smoothness_objective",
    "validate_market_snapshot",
    "write_market_snapshot",
]
