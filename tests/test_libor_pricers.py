from datetime import date

from sofr_curve_engine.bootstrap import build_full_curves
from sofr_curve_engine.dates import generate_schedule
from sofr_curve_engine.daycount import yearfrac
from sofr_curve_engine.instruments import CashflowPeriod
from sofr_curve_engine.libor_model import build_joint_model_calibration
from sofr_curve_engine.libor_pricers import (
    libor_forward_map_from_calibration_table,
    pv_libor_sofr_basis_swap_to_libor_payer,
    pv_libor_swap_with_sofr_fallback,
)


def _quarterly_periods(start: date, end: date) -> list[CashflowPeriod]:
    schedule = generate_schedule(start, end, "Quarterly")
    return [
        CashflowPeriod(start_date=left, end_date=right, accrual_factor=yearfrac(left, right, "ACT/360"))
        for left, right in zip(schedule[:-1], schedule[1:], strict=True)
    ]


def test_libor_swap_with_sofr_fallback_prices() -> None:
    curves = build_full_curves(project_root=".")
    calibration = build_joint_model_calibration(project_root=".")
    forwards = libor_forward_map_from_calibration_table(calibration.calibration_table)
    periods = _quarterly_periods(date(2025, 4, 15), date(2030, 4, 15))
    pv = pv_libor_swap_with_sofr_fallback(
        notional=100_000_000,
        fixed_rate=0.055,
        periods=periods,
        discount_curve=curves.discount_curve,
        projection_curve=curves.projection_curve,
        libor_forwards=forwards,
        fallback_spread=0.0025,
        fallback_start_date=date(2028, 4, 15),
    )
    assert abs(pv) > 1.0


def test_libor_sofr_basis_swap_to_libor_payer_is_negative_with_positive_fallback_spread() -> None:
    curves = build_full_curves(project_root=".")
    calibration = build_joint_model_calibration(project_root=".")
    forwards = libor_forward_map_from_calibration_table(calibration.calibration_table)
    periods = _quarterly_periods(date(2025, 4, 15), date(2030, 4, 15))
    pv = pv_libor_sofr_basis_swap_to_libor_payer(
        notional=100_000_000,
        periods=periods,
        discount_curve=curves.discount_curve,
        projection_curve=curves.projection_curve,
        libor_forwards=forwards,
        fallback_spread=0.0025,
        fallback_start_date=date(2028, 4, 15),
    )
    assert pv < 0.0
