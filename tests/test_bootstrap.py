from src.bootstrap import build_full_curves


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


def test_swap_holdout_repricing_reports_oos_and_stability_metrics() -> None:
    result = build_full_curves(project_root=".", holdout_swap_tenors=["10Y"])
    assert not result.oos_swap_repricing.empty
    assert "oos_swap_mae_bp" in result.diagnostics
    assert "oos_swap_max_1bp_shock_error_change_bp" in result.diagnostics
