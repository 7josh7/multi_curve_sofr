from __future__ import annotations

import pytest

from sofr_curve_engine.bootstrap import build_full_curves, load_market_data
from sofr_curve_engine.instruments import build_periods
from sofr_curve_engine.risk import key_rate_dv01, pv01, run_scenarios


@pytest.fixture(scope="module")
def payer_swap_inputs():
    market = load_market_data(".")
    curves = build_full_curves(".")
    quote = market.swaps[-1]
    periods = build_periods(
        quote.start_date,
        quote.end_date,
        quote.pay_freq,
        quote.day_count,
        calendar=market.config.market.calendar,
        roll=market.config.market.business_day_roll,
    )
    return market, curves, quote, periods


def test_payer_swap_pv01_is_positive_for_upward_rate_move(payer_swap_inputs) -> None:
    _, curves, quote, periods = payer_swap_inputs

    sensitivity = pv01(
        100_000_000,
        quote.fixed_rate,
        periods,
        curves.discount_curve,
        curves.projection_curve,
    )

    assert sensitivity > 0.0


def test_pv01_is_normalized_across_reasonable_bump_sizes(payer_swap_inputs) -> None:
    _, curves, quote, periods = payer_swap_inputs
    values = [
        pv01(
            100_000_000,
            quote.fixed_rate,
            periods,
            curves.discount_curve,
            curves.projection_curve,
            bump_bp=bump,
        )
        for bump in (0.5, 1.0, 2.0)
    ]

    assert max(values) - min(values) < 0.001 * abs(values[1])


def test_partitioned_key_rates_reconcile_to_parallel_pv01(payer_swap_inputs) -> None:
    market, curves, quote, periods = payer_swap_inputs
    parallel = pv01(
        100_000_000,
        quote.fixed_rate,
        periods,
        curves.discount_curve,
        curves.projection_curve,
    )
    ladder = key_rate_dv01(
        100_000_000,
        quote.fixed_rate,
        periods,
        curves.discount_curve,
        curves.projection_curve,
        list(market.config.risk.key_rates_years),
    )

    assert all(value > 0.0 for value in ladder.values())
    assert sum(ladder.values()) == pytest.approx(parallel, rel=0.005)


def test_parallel_scenarios_have_expected_payer_sign(payer_swap_inputs) -> None:
    _, curves, quote, periods = payer_swap_inputs
    scenarios = run_scenarios(
        100_000_000,
        quote.fixed_rate,
        periods,
        curves.discount_curve,
        curves.projection_curve,
    ).set_index("Scenario")

    assert scenarios.loc["Parallel +25bp", "Swap PV"] > scenarios.loc["Base", "Swap PV"]
    assert scenarios.loc["Parallel -25bp", "Swap PV"] < scenarios.loc["Base", "Swap PV"]
    assert scenarios.loc["Base", "DV01"] > 0.0


def test_risk_rejects_invalid_bumps_and_key_grids(payer_swap_inputs) -> None:
    _, curves, quote, periods = payer_swap_inputs
    with pytest.raises(ValueError, match="bump_bp"):
        pv01(1.0, quote.fixed_rate, periods, curves.discount_curve, curves.projection_curve, bump_bp=0.0)
    with pytest.raises(ValueError, match="duplicates"):
        key_rate_dv01(
            1.0,
            quote.fixed_rate,
            periods,
            curves.discount_curve,
            curves.projection_curve,
            [2.0, 2.0],
        )
