from datetime import date

from multi_curve_sofr.daycount import yearfrac


def test_act_360() -> None:
    assert yearfrac(date(2025, 4, 15), date(2025, 5, 15), "ACT/360") == 30 / 360


def test_act_365f() -> None:
    assert yearfrac(date(2025, 4, 15), date(2026, 4, 15), "ACT/365F") == 365 / 365


def test_thirty_360() -> None:
    assert yearfrac(date(2025, 1, 30), date(2025, 2, 28), "30/360") == 28 / 360
