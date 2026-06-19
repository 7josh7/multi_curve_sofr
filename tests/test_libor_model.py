from multi_curve_sofr.libor_model import (
    basis_instantaneous_vol,
    build_joint_model_calibration,
    hw_sigma_from_basis_minimization,
    shifted_lmm_vol_from_atm,
)


def test_shifted_lmm_vol_reduces_to_black_vol_when_shift_is_zero() -> None:
    vol = shifted_lmm_vol_from_atm(
        libor_forward=0.05,
        shift=0.0,
        atm_caplet_vol=0.30,
        option_maturity=2.0,
    )
    assert abs(vol - 0.30) < 1e-12


def test_basis_minimization_sigma_reduces_basis_vol() -> None:
    sigma = hw_sigma_from_basis_minimization(
        libor_forward=0.055,
        shift=0.02,
        shifted_lmm_vol=0.20,
        correlation_to_ois=0.65,
        mean_reversion=0.03,
        start_time=4.75,
        end_time=5.0,
        accrual_factor=0.25,
    )
    optimal_vol = basis_instantaneous_vol(
        libor_forward=0.055,
        ois_forward=0.045,
        shift=0.02,
        shifted_lmm_vol=0.20,
        hw_sigma=sigma,
        correlation_to_ois=0.65,
        mean_reversion=0.03,
        start_time=4.75,
        end_time=5.0,
        accrual_factor=0.25,
    )
    zero_hw_vol = basis_instantaneous_vol(
        libor_forward=0.055,
        ois_forward=0.045,
        shift=0.02,
        shifted_lmm_vol=0.20,
        hw_sigma=0.0,
        correlation_to_ois=0.65,
        mean_reversion=0.03,
        start_time=4.75,
        end_time=5.0,
        accrual_factor=0.25,
    )
    assert sigma > 0.0
    assert optimal_vol < zero_hw_vol


def test_joint_model_calibration_uses_synthetic_data() -> None:
    result = build_joint_model_calibration(project_root=".")
    assert len(result.calibration_table) == 20
    assert result.diagnostics["basis_implied_hw_sigma"] > 0.0
    assert result.calibration_table["shifted_lmm_vol"].min() > 0.0
    assert result.calibration_table["basis_vol_at_optimal_sigma"].mean() < result.calibration_table[
        "basis_vol_at_curve_sigma"
    ].mean()
