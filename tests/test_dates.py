from datetime import date

from multi_curve_sofr.dates import generate_schedule, is_imm_date, next_imm_date


def test_next_imm_date_from_regular_day() -> None:
    assert next_imm_date(date(2025, 4, 1)) == date(2025, 6, 18)


def test_is_imm_date_flags_third_wednesday() -> None:
    assert is_imm_date(date(2025, 6, 18))
    assert not is_imm_date(date(2025, 6, 17))


def test_generate_schedule_avoids_duplicate_adjusted_end_dates() -> None:
    schedule = generate_schedule(date(2025, 4, 15), date(2028, 4, 17), "Annual")
    assert schedule == [date(2025, 4, 15), date(2026, 4, 15), date(2027, 4, 15), date(2028, 4, 17)]
