from datetime import date

from multi_curve_sofr.curves import DiscountCurve


def test_curve_df_at_time_zero_is_one() -> None:
    curve = DiscountCurve.from_zero_rates(
        valuation_date=date(2025, 4, 15),
        pillar_dates=[date(2026, 4, 15), date(2027, 4, 15)],
        zero_rates=[0.04, 0.045],
    )
    assert curve.df(date(2025, 4, 15)) == 1.0


def test_curve_discount_factors_are_decreasing() -> None:
    curve = DiscountCurve.from_zero_rates(
        valuation_date=date(2025, 4, 15),
        pillar_dates=[date(2026, 4, 15), date(2027, 4, 15), date(2030, 4, 15)],
        zero_rates=[0.04, 0.045, 0.05],
    )
    assert curve.df(date(2026, 4, 15)) > curve.df(date(2027, 4, 15)) > curve.df(date(2030, 4, 15))
