from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from sofr_curve_engine.bootstrap import build_full_curves, load_market_data
from sofr_curve_engine.daycount import yearfrac
from sofr_curve_engine.export import build_market_snapshot, write_market_snapshot
from sofr_curve_engine.hw_model import U_j_const_sigma, convexity_1m
from sofr_curve_engine.instruments import build_periods
from sofr_curve_engine.libor_model import build_joint_model_calibration
from sofr_curve_engine.quantlib_bridge import available as quantlib_available
from sofr_curve_engine.quantlib_bridge import compare_swaps
from sofr_curve_engine.risk import key_rate_dv01, run_scenarios

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def main() -> int:
    market = load_market_data(ROOT)
    result = build_full_curves(ROOT)
    holdout = build_full_curves(ROOT, holdout_swap_tenors=["10Y"])
    no_convexity = build_full_curves(ROOT, sigma_override=0.0)

    write_market_snapshot(build_market_snapshot(result), OUTPUTS / "curves" / "sofr_market_snapshot.json")
    result.futures_repricing.to_csv(OUTPUTS / "tables" / "futures_repricing.csv", index=False)
    result.swap_repricing.to_csv(OUTPUTS / "tables" / "swap_repricing.csv", index=False)
    holdout.oos_swap_repricing.to_csv(OUTPUTS / "tables" / "oos_swap_repricing.csv", index=False)

    quote = market.swaps[-1]
    periods = build_periods(
        quote.start_date,
        quote.end_date,
        quote.pay_freq,
        quote.day_count,
        calendar=market.config.market.calendar,
        roll=market.config.market.business_day_roll,
    )
    scenarios = run_scenarios(
        100_000_000,
        quote.fixed_rate,
        periods,
        result.discount_curve,
        result.projection_curve,
        alt_projection_curve=no_convexity.projection_curve,
    )
    scenarios.to_csv(OUTPUTS / "tables" / "scenario_table.csv", index=False)
    key_rates = key_rate_dv01(
        100_000_000,
        quote.fixed_rate,
        periods,
        result.discount_curve,
        result.projection_curve,
        list(market.config.risk.key_rates_years),
    )
    _write_json(
        OUTPUTS / "reports" / "diagnostics.json",
        {
            "sigma": result.sigma,
            "calibration": result.calibration_diagnostics.to_dict(),
            "diagnostics": holdout.diagnostics,
            "key_rate_dv01": {str(key): value for key, value in key_rates.items()},
        },
    )

    joint = build_joint_model_calibration(ROOT)
    joint.calibration_table.to_csv(OUTPUTS / "tables" / "joint_model_calibration.csv", index=False)
    _write_json(
        OUTPUTS / "reports" / "joint_model_diagnostics.json",
        {
            **joint.diagnostics,
            "curve_calibration": joint.curve_build.calibration_diagnostics.to_dict(),
        },
    )

    _write_figures(result, market, scenarios)
    if quantlib_available():
        _write_quantlib_outputs()
    return 0


def _write_figures(result, market, scenarios) -> None:
    figures = OUTPUTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(result.discount_curve.times[1:], result.discount_curve.dfs[1:], marker="o", label="OIS discount")
    axes[0].plot(
        result.projection_curve.times[1:],
        result.projection_curve.dfs[1:],
        marker="s",
        label="SOFR projection",
    )
    axes[0].set(title="Discount Factors", xlabel="Years")
    axes[0].legend()
    axes[1].plot(
        result.discount_curve.times[1:],
        result.discount_curve.pillar_zero_rates(),
        marker="o",
        label="OIS zeros",
    )
    axes[1].plot(
        result.projection_curve.times[1:],
        result.projection_curve.pillar_zero_rates(),
        marker="s",
        label="SOFR zeros",
    )
    axes[1].set(title="Zero Rates", xlabel="Years")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "discount_curves.png", dpi=120)
    plt.close(fig)

    valuation = market.config.market.valuation_date
    a = result.mean_reversion
    sigma = result.sigma
    one_month = []
    for future in market.futures_1m:
        start = yearfrac(valuation, future.start_date, "ACT/365F")
        end = yearfrac(valuation, future.end_date, "ACT/365F")
        one_month.append((end, 10_000.0 * convexity_1m(a, sigma, start, end)))
    three_month = []
    for future in market.futures_3m:
        start = yearfrac(valuation, future.start_date, "ACT/365F")
        end = yearfrac(valuation, future.end_date, "ACT/365F")
        three_month.append((end, 10_000.0 * U_j_const_sigma(a, sigma, start, end)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([x for x, _ in one_month], [y for _, y in one_month], marker="o", label="1m convexity")
    ax.plot([x for x, _ in three_month], [y for _, y in three_month], marker="s", label="3m U_j term")
    ax.set(title="Hull-White Convexity Diagnostics", xlabel="Years from valuation", ylabel="Adjustment (bp)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "convexity_adjustments.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(scenarios["Scenario"], scenarios["Swap PV"])
    ax.set_title("10Y SOFR Swap Scenario PV")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figures / "swap_pv_scenarios.png", dpi=120)
    plt.close(fig)


def _write_quantlib_outputs() -> None:
    snapshot = OUTPUTS / "curves" / "sofr_market_snapshot.json"
    rows = compare_swaps(snapshot, ROOT / "data" / "market" / "sofr_swaps.csv")
    table = OUTPUTS / "tables" / "quantlib_comparison.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([row["tenor"] for row in rows], [row["diff_bp"] for row in rows])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set(title="SOFR swap par-rate alignment", ylabel="QuantLib - engine (bp)")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "figures" / "quantlib_par_rate_diff.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
