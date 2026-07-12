from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

from .bootstrap import CurveBuildResult, build_full_curves
from .calibration import configured_diagnostics

SCHEMA_VERSION = "1.0"
SOURCE_NAME = "sofr_curve_engine"
SOURCE_REVISION = "0.2.0"
CANONICAL_CONVENTIONS = {
    "time_basis": "ACT/365F",
    "zero_rate_compounding": "continuous",
    "canonical_representation": "discount_factor",
}
CANONICAL_UNITS = {
    "rates": "decimal",
    "volatility": "decimal_annualized",
    "discount_factors": "unitless",
    "time": "years",
    "errors": "basis_points",
}


def curve_to_nodes(curve) -> list[dict[str, float | str]]:
    """Convert a DiscountCurve into serializable curve nodes."""

    nodes: list[dict[str, float | str]] = []
    for node in curve.nodes():
        nodes.append(
            {
                "date": node.pillar_date.isoformat(),
                "time": float(node.time),
                "df": float(node.df),
                "zero_rate": float(curve.zero_rate(node.time)),
            }
        )
    return nodes


def build_market_snapshot(result: CurveBuildResult) -> dict[str, Any]:
    """Convert CurveBuildResult into a JSON-serializable market snapshot."""

    calibration = result.calibration_diagnostics or configured_diagnostics(
        method="unspecified_legacy_result",
        parameter_values={"a": float(result.mean_reversion), "sigma": float(result.sigma)},
        message="CurveBuildResult did not carry optimizer diagnostics.",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "valuation_date": result.config.market.valuation_date.isoformat(),
        "source": SOURCE_NAME,
        "source_revision": SOURCE_REVISION,
        "conventions": {
            **CANONICAL_CONVENTIONS,
            "calendar": result.config.market.calendar,
            "business_day_roll": result.config.market.business_day_roll,
            "futures_day_count": result.config.market.futures_day_count,
            "swap_fixed_day_count": result.config.market.swap_fixed_day_count,
            "swap_float_day_count": result.config.market.swap_float_day_count,
            "swap_pay_frequency": result.config.market.swap_pay_freq,
        },
        "units": dict(CANONICAL_UNITS),
        "discount_curve": {
            "label": result.discount_curve.label,
            "nodes": curve_to_nodes(result.discount_curve),
        },
        "projection_curve": {
            "label": result.projection_curve.label,
            "nodes": curve_to_nodes(result.projection_curve),
        },
        "model": {
            "mean_reversion": float(result.mean_reversion),
            "sigma": float(result.sigma),
            "calibration": calibration.to_dict(),
        },
        "diagnostics": _jsonable_mapping(result.diagnostics),
    }


def validate_market_snapshot(snapshot: dict[str, Any]) -> None:
    """Validate exported curve snapshot before writing it to disk."""

    required_sections = [
        "schema_version",
        "valuation_date",
        "source",
        "source_revision",
        "conventions",
        "units",
        "discount_curve",
        "projection_curve",
        "model",
        "diagnostics",
    ]
    missing = [section for section in required_sections if section not in snapshot]
    if missing:
        raise ValueError(f"Market snapshot missing required section(s): {', '.join(missing)}")
    if snapshot["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported market snapshot schema_version: {snapshot['schema_version']!r}.")
    if not isinstance(snapshot["source"], str) or not snapshot["source"].strip():
        raise ValueError("Market snapshot source must be a non-empty string.")
    if not isinstance(snapshot["source_revision"], str) or not snapshot["source_revision"].strip():
        raise ValueError("Market snapshot source_revision must be a non-empty string.")
    try:
        valuation_date = date.fromisoformat(str(snapshot["valuation_date"]))
    except ValueError as exc:
        raise ValueError("Market snapshot valuation_date must be an ISO date.") from exc

    conventions = snapshot["conventions"]
    if not isinstance(conventions, dict):
        raise ValueError("Market snapshot conventions must be an object.")
    for field, expected in CANONICAL_CONVENTIONS.items():
        if conventions.get(field) != expected:
            raise ValueError(f"Market snapshot convention {field!r} must be {expected!r}.")
    units = snapshot["units"]
    if not isinstance(units, dict) or any(units.get(field) != expected for field, expected in CANONICAL_UNITS.items()):
        raise ValueError("Market snapshot units do not match the schema 1.0 canonical units.")

    _validate_model(snapshot["model"])

    for curve_name in ["discount_curve", "projection_curve"]:
        curve = snapshot[curve_name]
        if not isinstance(curve, dict) or not isinstance(curve.get("label"), str) or not curve["label"]:
            raise ValueError(f"Market snapshot {curve_name} requires a non-empty label.")
        if "nodes" not in curve or not curve["nodes"]:
            raise ValueError(f"Market snapshot {curve_name} must contain at least one node.")
        _validate_curve_nodes(curve["nodes"], curve_name, valuation_date)

    try:
        json.dumps(snapshot, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Market snapshot must be finite and JSON serializable.") from exc


def write_market_snapshot(snapshot: dict[str, Any], output_path: str | Path) -> Path:
    """Write market snapshot JSON to disk."""

    validate_market_snapshot(snapshot)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, allow_nan=False)
    return path


def export_market_snapshot(
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Build full OIS/SOFR curves and export them as a market snapshot JSON."""

    result = build_full_curves(project_root=project_root)
    snapshot = build_market_snapshot(result)
    validate_market_snapshot(snapshot)
    target = Path(output_path) if output_path is not None else result.config.output_dir / "curves" / "sofr_market_snapshot.json"
    return write_market_snapshot(snapshot, target)


def _validate_curve_nodes(nodes: list[dict[str, Any]], curve_name: str, valuation_date: date) -> None:
    previous_time = -float("inf")
    for index, node in enumerate(nodes):
        for field in ["date", "time", "df", "zero_rate"]:
            if field not in node:
                raise ValueError(f"Market snapshot {curve_name} node {index} missing {field!r}.")
        try:
            node_date = date.fromisoformat(str(node["date"]))
            time = float(node["time"])
            df = float(node["df"])
            zero_rate = float(node["zero_rate"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Market snapshot {curve_name} node {index} contains an invalid value.") from exc
        if not all(math.isfinite(value) for value in (time, df, zero_rate)):
            raise ValueError(f"Market snapshot {curve_name} node {index} values must be finite.")
        if node_date <= valuation_date:
            raise ValueError(f"Market snapshot {curve_name} node dates must be after valuation_date.")
        expected_time = (node_date - valuation_date).days / 365.0
        if not math.isclose(time, expected_time, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Market snapshot {curve_name} node time does not match ACT/365F.")
        if time <= 0.0 or time <= previous_time:
            raise ValueError(f"Market snapshot {curve_name} node times must be strictly increasing.")
        if df <= 0.0:
            raise ValueError(f"Market snapshot {curve_name} node discount factors must be positive.")
        expected_zero = -math.log(df) / time
        if not math.isclose(zero_rate, expected_zero, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(
                f"Market snapshot {curve_name} node zero_rate must reconcile to the canonical discount factor."
            )
        previous_time = time


def _validate_model(model: object) -> None:
    if not isinstance(model, dict):
        raise ValueError("Market snapshot model must be an object.")
    for field in ("mean_reversion", "sigma"):
        if field not in model:
            raise ValueError(f"Market snapshot model missing {field!r}.")
        value = float(model[field])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Market snapshot model {field!r} must be finite and non-negative.")

    calibration = model.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Market snapshot model requires calibration diagnostics.")
    required = {
        "method",
        "converged",
        "objective_value",
        "message",
        "iterations",
        "function_evaluations",
        "parameter_values",
        "parameter_bounds",
        "boundary_hit",
        "boundary_hits",
        "is_regularization_proxy",
    }
    missing = sorted(required - set(calibration))
    if missing:
        raise ValueError(f"Market snapshot calibration diagnostics missing: {', '.join(missing)}")
    if not isinstance(calibration["method"], str) or not calibration["method"]:
        raise ValueError("Calibration method must be a non-empty string.")
    if calibration["converged"] is not True:
        raise ValueError("Only converged calibration diagnostics may be exported.")
    objective = calibration["objective_value"]
    if objective is not None and (not math.isfinite(float(objective)) or float(objective) < 0.0):
        raise ValueError("Calibration objective_value must be finite and non-negative when present.")
    parameter_values = calibration["parameter_values"]
    parameter_bounds = calibration["parameter_bounds"]
    boundary_values = calibration["boundary_hits"]
    if not isinstance(parameter_values, dict) or not parameter_values:
        raise ValueError("Calibration parameter_values must be a non-empty object.")
    if not isinstance(parameter_bounds, dict) or not isinstance(boundary_values, dict):
        raise ValueError("Calibration bounds and boundary hits must be objects.")
    if any(not math.isfinite(float(value)) for value in parameter_values.values()):
        raise ValueError("Calibration parameter values must be finite.")
    for name, bounds in parameter_bounds.items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"Calibration bounds for {name!r} must contain two values.")
        lower, upper = float(bounds[0]), float(bounds[1])
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise ValueError(f"Calibration bounds for {name!r} are invalid.")
        if name in parameter_values and not lower <= float(parameter_values[name]) <= upper:
            raise ValueError(f"Calibration parameter {name!r} is outside its bounds.")
    if any(side not in {"lower", "upper"} for side in boundary_values.values()):
        raise ValueError("Calibration boundary hit values must be 'lower' or 'upper'.")
    if calibration["boundary_hit"] is not bool(boundary_values):
        raise ValueError("Calibration boundary_hit must agree with boundary_hits.")


def _jsonable_mapping(values: dict[str, Any]) -> dict[str, Any]:
    jsonable: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            jsonable[key] = value
        elif hasattr(value, "item"):
            jsonable[key] = value.item()
    return jsonable
