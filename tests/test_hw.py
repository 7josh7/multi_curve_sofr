from src.hw_model import A_const_sigma, U_j_const_sigma, convexity_1m


def test_convexity_zero_sigma() -> None:
    adjustment = convexity_1m(a=0.03, sigma=0.0, T_start=0.25, T_end=1.0 / 3.0)
    assert abs(adjustment) < 1e-12


def test_three_month_u_term_zero_sigma() -> None:
    assert abs(U_j_const_sigma(a=0.03, sigma=0.0, T_prev=0.5, T_curr=0.75)) < 1e-12


def test_a_term_is_one_when_sigma_is_zero() -> None:
    assert abs(A_const_sigma(a=0.03, sigma=0.0, t=0.0, T=2.0) - 1.0) < 1e-12
