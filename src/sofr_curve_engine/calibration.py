from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationDiagnostics:
    """Serializable optimizer diagnostics shared by all model calibrations."""

    method: str
    converged: bool
    objective_value: float | None
    message: str
    iterations: int | None
    function_evaluations: int | None
    parameter_values: dict[str, float]
    parameter_bounds: dict[str, tuple[float, float]]
    boundary_hits: dict[str, str]
    is_regularization_proxy: bool = False

    @property
    def boundary_hit(self) -> bool:
        return bool(self.boundary_hits)

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "converged": self.converged,
            "objective_value": self.objective_value,
            "message": self.message,
            "iterations": self.iterations,
            "function_evaluations": self.function_evaluations,
            "parameter_values": dict(self.parameter_values),
            "parameter_bounds": {
                name: [float(bounds[0]), float(bounds[1])]
                for name, bounds in self.parameter_bounds.items()
            },
            "boundary_hit": self.boundary_hit,
            "boundary_hits": dict(self.boundary_hits),
            "is_regularization_proxy": self.is_regularization_proxy,
        }


def boundary_hits(
    parameter_values: dict[str, float],
    parameter_bounds: dict[str, tuple[float, float]],
) -> dict[str, str]:
    """Report parameters numerically pinned to a lower or upper bound."""

    hits: dict[str, str] = {}
    for name, value in parameter_values.items():
        lower, upper = parameter_bounds[name]
        tolerance = max(1e-10, 1e-6 * max(upper - lower, 1.0))
        if math.isclose(value, lower, rel_tol=0.0, abs_tol=tolerance):
            hits[name] = "lower"
        elif math.isclose(value, upper, rel_tol=0.0, abs_tol=tolerance):
            hits[name] = "upper"
    return hits


def configured_diagnostics(
    *,
    method: str,
    parameter_values: dict[str, float],
    message: str,
) -> CalibrationDiagnostics:
    """Diagnostics for configured or explicitly overridden model parameters."""

    return CalibrationDiagnostics(
        method=method,
        converged=True,
        objective_value=None,
        message=message,
        iterations=None,
        function_evaluations=None,
        parameter_values=dict(parameter_values),
        parameter_bounds={},
        boundary_hits={},
        is_regularization_proxy=False,
    )
