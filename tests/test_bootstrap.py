from src.bootstrap import (
    build_full_curves,
    calibrate_sigma_from_sofr_curve_smoothness,
    load_market_data,
    sofr_curve_smoothness_objective,
)


def test_bootstrap_reprices_futures_and_swaps() -> None:
    result = build_full_curves(project_root=".")
    assert result.futures_repricing["Error (bp)"].abs().max() < 1e-6
    # Swaps bootstrapped from the swap quotes reprice to machine precision.
    # The 2Y swap may show a small residual (<1 bp) because its maturity
    # falls inside the futures strip and is anchored by the futures DF.
    assert result.swap_repricing["Error (bp)"].abs().max() < 1.0


def test_bootstrap_diagnostics_show_monotone_positive_curves() -> None:
    result = build_full_curves(project_root=".")
    assert result.diagnostics["projection_monotone"]
    assert result.diagnostics["discount_monotone"]
    assert result.diagnostics["positive_projection_dfs"]
    assert result.diagnostics["positive_discount_dfs"]


def test_sofr_smoothness_sigma_calibration_respects_bounds_and_reduces_roughness() -> None:
    market = load_market_data(".")
    a = market.config.model.mean_reversion
    lower, upper = market.config.model.sigma_bounds

    sigma = calibrate_sigma_from_sofr_curve_smoothness(market, a=a, sigma_bounds=(lower, upper))

    assert lower <= sigma <= upper
    assert sofr_curve_smoothness_objective(market, sigma=sigma, a=a) <= sofr_curve_smoothness_objective(
        market,
        sigma=upper,
        a=a,
    )


def test_build_full_curves_uses_sofr_smoothness_sigma_calibration() -> None:
    result = build_full_curves(project_root=".", sigma_calibration_method="sofr_curve_smoothness")

    lower, upper = result.config.model.sigma_bounds
    assert lower <= result.sigma <= upper
    assert result.mean_reversion == result.config.model.mean_reversion


def test_build_full_curves_still_supports_swaption_surface_calibration() -> None:
    result = build_full_curves(project_root=".", sigma_calibration_method="swaption_surface")

    lower, upper = result.config.model.sigma_bounds
    assert lower <= result.sigma <= upper
    assert result.mean_reversion > 0.0


def test_swap_holdout_repricing_reports_oos_and_stability_metrics() -> None:
    result = build_full_curves(project_root=".", holdout_swap_tenors=["10Y"])
    assert not result.oos_swap_repricing.empty
    assert "oos_swap_mae_bp" in result.diagnostics
    assert "oos_swap_max_1bp_shock_error_change_bp" in result.diagnostics
