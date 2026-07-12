from __future__ import annotations

import copy

import pytest

from sofr_curve_engine.bootstrap import build_full_curves
from sofr_curve_engine.export import SCHEMA_VERSION, build_market_snapshot, validate_market_snapshot


def test_export_market_snapshot_schema() -> None:
    result = build_full_curves(project_root=".")
    snapshot = build_market_snapshot(result)

    assert snapshot["schema_version"] == SCHEMA_VERSION == "1.0"
    assert snapshot["source"] == "sofr_curve_engine"
    assert snapshot["source_revision"]
    assert snapshot["conventions"]["time_basis"] == "ACT/365F"
    assert snapshot["conventions"]["zero_rate_compounding"] == "continuous"
    assert snapshot["conventions"]["canonical_representation"] == "discount_factor"
    assert snapshot["units"]["rates"] == "decimal"
    assert "discount_curve" in snapshot
    assert "projection_curve" in snapshot
    assert len(snapshot["discount_curve"]["nodes"]) > 0
    assert len(snapshot["projection_curve"]["nodes"]) > 0
    assert snapshot["model"]["calibration"]["converged"] is True
    assert snapshot["model"]["calibration"]["is_regularization_proxy"] is True
    validate_market_snapshot(snapshot)


def test_export_uses_effective_model_parameters() -> None:
    result = build_full_curves(
        project_root=".",
        mean_reversion_override=0.07,
        sigma_override=0.01,
    )

    snapshot = build_market_snapshot(result)

    assert snapshot["model"]["mean_reversion"] == 0.07
    assert snapshot["model"]["sigma"] == 0.01
    assert snapshot["model"]["calibration"]["method"] == "explicit_override"


def test_validate_market_snapshot_rejects_bad_df() -> None:
    result = build_full_curves(project_root=".")
    snapshot = build_market_snapshot(result)
    bad_snapshot = copy.deepcopy(snapshot)
    bad_snapshot["discount_curve"]["nodes"][0]["df"] = -1.0

    with pytest.raises(ValueError):
        validate_market_snapshot(bad_snapshot)


def test_validate_market_snapshot_rejects_noncanonical_zero_rate() -> None:
    snapshot = build_market_snapshot(build_full_curves(project_root="."))
    snapshot["projection_curve"]["nodes"][0]["zero_rate"] += 1e-5

    with pytest.raises(ValueError, match="reconcile"):
        validate_market_snapshot(snapshot)


def test_validate_market_snapshot_rejects_nonfinite_values() -> None:
    snapshot = build_market_snapshot(build_full_curves(project_root="."))
    snapshot["discount_curve"]["nodes"][0]["df"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        validate_market_snapshot(snapshot)


def test_validate_market_snapshot_rejects_schema_or_convention_drift() -> None:
    snapshot = build_market_snapshot(build_full_curves(project_root="."))
    bad_version = copy.deepcopy(snapshot)
    bad_version["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="schema_version"):
        validate_market_snapshot(bad_version)

    bad_convention = copy.deepcopy(snapshot)
    bad_convention["conventions"]["canonical_representation"] = "zero_rate"
    with pytest.raises(ValueError, match="canonical_representation"):
        validate_market_snapshot(bad_convention)
