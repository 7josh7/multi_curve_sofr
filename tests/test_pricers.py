from sofr_curve_engine.bootstrap import build_full_curves, load_market_data
from sofr_curve_engine.instruments import build_periods
from sofr_curve_engine.pricers import par_swap_rate, pv_swap


def test_market_par_swap_has_near_zero_pv() -> None:
    result = build_full_curves(project_root=".")
    market = load_market_data(".")
    quote = market.swaps[-1]
    periods = build_periods(
        quote.start_date,
        quote.end_date,
        quote.pay_freq,
        quote.day_count,
        calendar=market.config.market.calendar,
        roll=market.config.market.business_day_roll,
    )
    pv = pv_swap(100_000_000, quote.fixed_rate, periods, result.discount_curve, result.projection_curve)
    assert abs(pv) < 1.0


def test_par_swap_rate_matches_market_quote() -> None:
    result = build_full_curves(project_root=".")
    market = load_market_data(".")
    quote = market.swaps[1]
    periods = build_periods(
        quote.start_date,
        quote.end_date,
        quote.pay_freq,
        quote.day_count,
        calendar=market.config.market.calendar,
        roll=market.config.market.business_day_roll,
    )
    model_rate = par_swap_rate(result.discount_curve, result.projection_curve, periods)
    assert abs(model_rate - quote.fixed_rate) < 1e-10
