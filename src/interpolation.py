from __future__ import annotations

import math

import numpy as np


class LogDFInterpolator:
    def __init__(self, times: list[float] | np.ndarray, dfs: list[float] | np.ndarray) -> None:
        raw_times = np.asarray(times, dtype=float)
        raw_dfs = np.asarray(dfs, dtype=float)
        if raw_times.ndim != 1 or raw_dfs.ndim != 1:
            raise ValueError("Times and discount factors must be one-dimensional arrays.")
        if raw_times.size != raw_dfs.size:
            raise ValueError("Times and discount factors must have the same length.")
        if raw_times[0] != 0.0:
            raise ValueError("Interpolator requires time zero as the first pillar.")
        if np.any(raw_dfs <= 0.0):
            raise ValueError("Discount factors must be strictly positive.")

        self.times = raw_times
        self.dfs = raw_dfs
        self.log_dfs = np.log(raw_dfs)
        self._tail_zero = -self.log_dfs[-1] / max(self.times[-1], 1e-12)

    def df(self, time: float) -> float:
        t = float(time)
        if t <= 0.0:
            return 1.0
        if t >= self.times[-1]:
            return math.exp(-self._tail_zero * t)
        interpolated = float(np.interp(t, self.times, self.log_dfs))
        return math.exp(interpolated)

    def zero(self, time: float) -> float:
        t = float(time)
        if t <= 0.0:
            return self.zero(self.times[1])
        return -math.log(self.df(t)) / t
