from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bootstrap import CurveBuildResult, build_full_curves


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

    return {
        "valuation_date": result.config.market.valuation_date.isoformat(),
        "source": "multi_curve_sofr",
        "discount_curve": {
            "label": result.discount_curve.label,
            "nodes": curve_to_nodes(result.discount_curve),
        },
        "projection_curve": {
            "label": result.projection_curve.label,
            "nodes": curve_to_nodes(result.projection_curve),
        },
        "model": {
            "mean_reversion": float(result.config.model.mean_reversion),
            "sigma": float(result.sigma),
        },
        "diagnostics": _jsonable_mapping(result.diagnostics),
    }


def validate_market_snapshot(snapshot: dict[str, Any]) -> None:
    """Validate exported curve snapshot before writing it to disk."""

    required_sections = ["valuation_date", "source", "discount_curve", "projection_curve", "model", "diagnostics"]
    missing = [section for section in required_sections if section not in snapshot]
    if missing:
        raise ValueError(f"Market snapshot missing required section(s): {', '.join(missing)}")
    if not snapshot["valuation_date"]:
        raise ValueError("Market snapshot valuation_date is required.")

    for curve_name in ["discount_curve", "projection_curve"]:
        curve = snapshot[curve_name]
        if "nodes" not in curve or not curve["nodes"]:
            raise ValueError(f"Market snapshot {curve_name} must contain at least one node.")
        _validate_curve_nodes(curve["nodes"], curve_name)


def write_market_snapshot(snapshot: dict[str, Any], output_path: str | Path) -> Path:
    """Write market snapshot JSON to disk."""

    validate_market_snapshot(snapshot)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)
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


def _validate_curve_nodes(nodes: list[dict[str, Any]], curve_name: str) -> None:
    previous_time = -float("inf")
    for index, node in enumerate(nodes):
        for field in ["date", "time", "df", "zero_rate"]:
            if field not in node:
                raise ValueError(f"Market snapshot {curve_name} node {index} missing {field!r}.")
        time = float(node["time"])
        df = float(node["df"])
        if time <= previous_time:
            raise ValueError(f"Market snapshot {curve_name} node times must be strictly increasing.")
        if df <= 0.0:
            raise ValueError(f"Market snapshot {curve_name} node discount factors must be positive.")
        previous_time = time


def _jsonable_mapping(values: dict[str, Any]) -> dict[str, Any]:
    jsonable: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            jsonable[key] = value
        elif hasattr(value, "item"):
            jsonable[key] = value.item()
    return jsonable
