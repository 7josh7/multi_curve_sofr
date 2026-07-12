"""One check: the engine's SOFR swap par rates agree with QuantLib's.

Skips cleanly when QuantLib isn't installed, so it never blocks the core suite.
"""
from pathlib import Path

import pytest

pytestmark = pytest.mark.quantlib

ql = pytest.importorskip("QuantLib")

from sofr_curve_engine.export import export_market_snapshot
from sofr_curve_engine.quantlib_bridge import compare_swaps

ROOT = Path(__file__).resolve().parent.parent
TOL_BP = 0.5  # engine uses a projection-ratio float leg vs QuantLib's exact compounding


def test_par_rates_match_quantlib(tmp_path):
    snap = export_market_snapshot(project_root=str(ROOT), output_path=str(tmp_path / "snap.json"))
    rows = compare_swaps(snap, ROOT / "data/market/sofr_swaps.csv")
    worst = max(abs(r["diff_bp"]) for r in rows)
    assert worst < TOL_BP, f"max par-rate gap {worst:.3f} bp exceeds {TOL_BP} bp"
