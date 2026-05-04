from src.bootstrap import build_full_curves


def test_bootstrap_reprices_futures_and_swaps() -> None:
    result = build_full_curves(project_root=".")
    assert result.futures_repricing["Error (bp)"].abs().max() < 1e-6
    assert result.swap_repricing["Error (bp)"].abs().max() < 1e-2


def test_bootstrap_diagnostics_show_monotone_positive_curves() -> None:
    result = build_full_curves(project_root=".")
    assert result.diagnostics["projection_monotone"]
    assert result.diagnostics["discount_monotone"]
    assert result.diagnostics["positive_projection_dfs"]
    assert result.diagnostics["positive_discount_dfs"]
