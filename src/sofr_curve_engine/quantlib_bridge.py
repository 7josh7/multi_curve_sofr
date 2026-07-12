"""QuantLib swap-pricing cross-check for the SOFR curve/risk engine.

Reuses the engine's exported market snapshot and reprices the same SOFR swaps
with QuantLib's own OIS machinery. This checks pricing/convention alignment on
the engine's curves; it is not an independent curve bootstrap.

QuantLib is an optional dependency: import it lazily so the core engine still
runs when it is absent (this is the "when available" comparison).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

try:
    import QuantLib as ql
except ImportError:  # ponytail: optional dep. Comparison is skipped, engine unaffected.
    ql = None

# Conventions mirror data/metadata/conventions.json (WEEKEND calendar, ACT/360).
_VAL = (2025, 4, 15)


def available() -> bool:
    return ql is not None


def _curve(nodes, val, cal, dc):
    """A QuantLib discount curve from the snapshot's (date, df) nodes."""
    dates, dfs = [val], [1.0]  # snapshot omits the t=0 anchor; QuantLib needs it.
    for n in nodes:
        y, m, d = map(int, n["date"].split("-"))
        dates.append(ql.Date(d, m, y))
        dfs.append(n["df"])
    curve = ql.DiscountCurve(dates, dfs, dc, cal)
    curve.enableExtrapolation()
    return curve


def compare_swaps(snapshot_path: str | Path, swaps_csv: str | Path) -> list[dict]:
    """Reprice each SOFR swap in QuantLib off the engine's curves.

    Returns one row per tenor with the engine's market rate, QuantLib's fair
    rate, the difference in bp, NPV at the market rate, and DV01.
    """
    if ql is None:
        raise RuntimeError("QuantLib is not installed; run `pip install QuantLib`.")

    val = ql.Date(*reversed(_VAL))
    ql.Settings.instance().evaluationDate = val
    cal, dc = ql.WeekendsOnly(), ql.Actual360()

    snap = json.loads(Path(snapshot_path).read_text())
    ois = ql.YieldTermStructureHandle(_curve(snap["discount_curve"]["nodes"], val, cal, dc))
    proj = ql.YieldTermStructureHandle(_curve(snap["projection_curve"]["nodes"], val, cal, dc))
    index = ql.Sofr(proj)  # forecasts on the engine's SOFR projection curve

    rows = []
    for r in csv.DictReader(Path(swaps_csv).open()):
        years = int(r["tenor"][:-1])
        market = float(r["fixed_rate"])
        # ponytail: MakeOIS builds the schedule from the val date; tiny roll
        # differences vs the data's end_dates land in diff_bp (sub-bp). Build
        # explicit schedules only if exact-date matching is ever needed.
        swap = ql.MakeOIS(ql.Period(years, ql.Years), index, market,
                          nominal=1e8, discountingTermStructure=ois)
        fair = swap.fairRate()
        rows.append({
            "tenor": r["tenor"],
            "market_rate": market,
            "quantlib_fair_rate": fair,
            "diff_bp": (fair - market) * 1e4,
            "npv_per_100mm": swap.NPV(),
            "dv01": swap.fixedLegBPS(),
        })
    return rows


def _main():
    root = Path(__file__).resolve().parents[2]
    rows = compare_swaps(root / "outputs/curves/sofr_market_snapshot.json",
                         root / "data/market/sofr_swaps.csv")
    out = root / "outputs/tables/quantlib_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    worst = max(abs(r["diff_bp"]) for r in rows)
    print(f"Wrote {out}  (max |diff| = {worst:.3f} bp vs QuantLib)")

    try:  # optional plot; absence of matplotlib must not fail the comparison
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.bar([r["tenor"] for r in rows], [r["diff_bp"] for r in rows], color="#3b6ea5")
        plt.axhline(0, color="black", lw=0.8)
        plt.ylabel("QuantLib - engine par rate (bp)")
        plt.title("SOFR swap par-rate agreement vs QuantLib")
        plt.tight_layout()
        fig = root / "outputs/figures/quantlib_par_rate_diff.png"
        plt.savefig(fig, dpi=120)
        print(f"Wrote {fig}")
    except ImportError:
        pass


if __name__ == "__main__":
    _main()
