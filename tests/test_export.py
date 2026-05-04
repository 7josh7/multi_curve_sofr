from __future__ import annotations

import copy

from src.bootstrap import build_full_curves
from src.export import build_market_snapshot, validate_market_snapshot


def test_export_market_snapshot_schema() -> None:
    result = build_full_curves(project_root=".")
    snapshot = build_market_snapshot(result)

    assert "valuation_date" in snapshot
    assert "discount_curve" in snapshot
    assert "projection_curve" in snapshot
    assert len(snapshot["discount_curve"]["nodes"]) > 0
    assert len(snapshot["projection_curve"]["nodes"]) > 0


def test_validate_market_snapshot_rejects_bad_df() -> None:
    result = build_full_curves(project_root=".")
    snapshot = build_market_snapshot(result)
    bad_snapshot = copy.deepcopy(snapshot)
    bad_snapshot["discount_curve"]["nodes"][0]["df"] = -1.0

    try:
        validate_market_snapshot(bad_snapshot)
    except ValueError:
        return
    raise AssertionError("validate_market_snapshot should reject negative discount factors.")
