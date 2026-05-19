from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

from .config import EngineConfig
from .instruments import FuturesQuote, SwapQuote
from .utils import ensure_date


class DataInputError(ValueError):
    """Base class for malformed input data."""


class DataSchemaError(DataInputError):
    """Raised when an input table is missing expected structure."""


class DataValidationError(DataInputError):
    """Raised when an input cell cannot be converted to the domain type."""


@dataclass(frozen=True)
class MarketData:
    config: EngineConfig
    fixings: pd.DataFrame
    futures_1m: list[FuturesQuote]
    futures_3m: list[FuturesQuote]
    swaps: list[SwapQuote]
    ois_curve: pd.DataFrame


@runtime_checkable
class MarketDataSource(Protocol):
    def load(self, config: EngineConfig) -> MarketData:
        """Load and normalize market data for the engine."""
        ...


@runtime_checkable
class LiborCalibrationDataSource(Protocol):
    def load(self, config: EngineConfig) -> pd.DataFrame:
        """Load and normalize LIBOR calibration data."""
        ...


@dataclass(frozen=True)
class _CsvDataSource:
    data_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))

    def _read_csv(
        self,
        relative_path: str | Path,
        required_columns: tuple[str, ...],
        *,
        allow_empty: bool = False,
    ) -> tuple[pd.DataFrame, Path]:
        path = self.data_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - pandas owns the exact parser errors
            raise DataInputError(f"Could not read input file {path}: {exc}") from exc

        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            joined = ", ".join(missing)
            raise DataSchemaError(f"{path}: missing required columns: {joined}")
        if frame.empty and not allow_empty:
            raise DataSchemaError(f"{path}: expected at least one row")
        return frame, path

    def _normalize_date_column(self, frame: pd.DataFrame, path: Path, column: str) -> None:
        frame[column] = [
            self._date_value(value, path=path, column=column, row_number=row_number)
            for row_number, value in enumerate(frame[column], start=2)
        ]

    def _normalize_float_column(self, frame: pd.DataFrame, path: Path, column: str) -> None:
        frame[column] = [
            self._float_value(value, path=path, column=column, row_number=row_number)
            for row_number, value in enumerate(frame[column], start=2)
        ]

    def _normalize_string_column(self, frame: pd.DataFrame, path: Path, column: str) -> None:
        frame[column] = [
            self._string_value(value, path=path, column=column, row_number=row_number)
            for row_number, value in enumerate(frame[column], start=2)
        ]

    def _date_value(self, value: object, *, path: Path, column: str, row_number: int) -> date:
        if self._is_missing(value):
            raise DataValidationError(f"{self._cell(path, column, row_number)}: date is required")
        try:
            return ensure_date(value)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"{self._cell(path, column, row_number)}: invalid date value {value!r}"
            ) from exc

    def _float_value(self, value: object, *, path: Path, column: str, row_number: int) -> float:
        if self._is_missing(value):
            raise DataValidationError(f"{self._cell(path, column, row_number)}: numeric value is required")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"{self._cell(path, column, row_number)}: invalid numeric value {value!r}"
            ) from exc
        if not math.isfinite(parsed):
            raise DataValidationError(f"{self._cell(path, column, row_number)}: numeric value must be finite")
        return parsed

    def _string_value(self, value: object, *, path: Path, column: str, row_number: int) -> str:
        if self._is_missing(value):
            raise DataValidationError(f"{self._cell(path, column, row_number)}: text value is required")
        parsed = str(value).strip()
        if not parsed:
            raise DataValidationError(f"{self._cell(path, column, row_number)}: text value is required")
        return parsed

    def _ensure_unique(self, frame: pd.DataFrame, path: Path, column: str) -> None:
        duplicates = frame.loc[frame[column].duplicated(), column]
        if not duplicates.empty:
            raise DataSchemaError(f"{path}: duplicate {column} value {duplicates.iloc[0]!r}")

    def _ensure_strict_date_interval(self, start: date, end: date, path: Path, row_number: int) -> None:
        if end <= start:
            raise DataValidationError(
                f"{path} row {row_number}: end_date must be after start_date; got {start.isoformat()} to {end.isoformat()}"
            )

    @staticmethod
    def _is_missing(value: object) -> bool:
        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _cell(path: Path, column: str, row_number: int) -> str:
        return f"{path} row {row_number} column '{column}'"


@dataclass(frozen=True)
class CsvMarketDataSource(_CsvDataSource):
    """CSV-backed market data source with schema and domain validation."""

    def load(self, config: EngineConfig) -> MarketData:
        return MarketData(
            config=config,
            fixings=self._load_fixings(),
            futures_1m=self._load_futures(
                Path("market") / "sofr_1m_futures.csv",
                contract_type="SOFR_1M",
            ),
            futures_3m=self._load_futures(
                Path("market") / "sofr_3m_futures.csv",
                contract_type="SOFR_3M",
            ),
            swaps=self._load_swaps(),
            ois_curve=self._load_ois_curve(),
        )

    def _load_fixings(self) -> pd.DataFrame:
        frame, path = self._read_csv(
            Path("fixings") / "sofr_fixings.csv",
            required_columns=("date", "sofr"),
            allow_empty=True,
        )
        if frame.empty:
            return frame
        frame = frame.copy()
        self._normalize_date_column(frame, path, "date")
        self._normalize_float_column(frame, path, "sofr")
        self._ensure_unique(frame, path, "date")
        return frame.sort_values("date").reset_index(drop=True)

    def _load_futures(
        self,
        relative_path: Path,
        *,
        contract_type: Literal["SOFR_1M", "SOFR_3M"],
    ) -> list[FuturesQuote]:
        frame, path = self._read_csv(
            relative_path,
            required_columns=("contract_code", "start_date", "end_date", "price"),
        )
        frame = frame.copy()
        self._normalize_string_column(frame, path, "contract_code")
        self._ensure_unique(frame, path, "contract_code")

        quotes: list[FuturesQuote] = []
        for row_number, row in enumerate(frame.to_dict("records"), start=2):
            start = self._date_value(row["start_date"], path=path, column="start_date", row_number=row_number)
            end = self._date_value(row["end_date"], path=path, column="end_date", row_number=row_number)
            self._ensure_strict_date_interval(start, end, path, row_number)
            price = self._float_value(row["price"], path=path, column="price", row_number=row_number)
            if price <= 0.0:
                raise DataValidationError(f"{path} row {row_number} column 'price': price must be positive")
            quotes.append(
                FuturesQuote(
                    contract_type=contract_type,
                    contract_code=row["contract_code"],
                    start_date=start,
                    end_date=end,
                    price=price,
                )
            )
        return quotes

    def _load_swaps(self) -> list[SwapQuote]:
        frame, path = self._read_csv(
            Path("market") / "sofr_swaps.csv",
            required_columns=("tenor", "start_date", "end_date", "fixed_rate", "pay_freq", "day_count"),
        )
        frame = frame.copy()
        self._normalize_string_column(frame, path, "tenor")
        self._normalize_string_column(frame, path, "pay_freq")
        self._normalize_string_column(frame, path, "day_count")
        self._ensure_unique(frame, path, "tenor")

        quotes: list[SwapQuote] = []
        for row_number, row in enumerate(frame.to_dict("records"), start=2):
            start = self._date_value(row["start_date"], path=path, column="start_date", row_number=row_number)
            end = self._date_value(row["end_date"], path=path, column="end_date", row_number=row_number)
            self._ensure_strict_date_interval(start, end, path, row_number)
            quotes.append(
                SwapQuote(
                    tenor=row["tenor"],
                    start_date=start,
                    end_date=end,
                    fixed_rate=self._float_value(
                        row["fixed_rate"],
                        path=path,
                        column="fixed_rate",
                        row_number=row_number,
                    ),
                    pay_freq=row["pay_freq"],
                    day_count=row["day_count"],
                )
            )
        return quotes

    def _load_ois_curve(self) -> pd.DataFrame:
        frame, path = self._read_csv(
            Path("market") / "ois_curve.csv",
            required_columns=("end_date", "zero_rate"),
        )
        frame = frame.copy()
        self._normalize_date_column(frame, path, "end_date")
        self._normalize_float_column(frame, path, "zero_rate")
        if "tenor" in frame.columns:
            self._normalize_string_column(frame, path, "tenor")
        self._ensure_unique(frame, path, "end_date")
        return frame.sort_values("end_date").reset_index(drop=True)


@dataclass(frozen=True)
class CsvLiborCalibrationDataSource(_CsvDataSource):
    """CSV-backed LIBOR calibration data source."""

    def load(self, config: EngineConfig) -> pd.DataFrame:
        frame, path = self._read_csv(
            Path("market") / config.joint_model.libor_calibration_file,
            required_columns=("tenor", "start_date", "end_date", "libor_forward", "atm_caplet_vol"),
        )
        frame = frame.copy()
        self._normalize_string_column(frame, path, "tenor")
        self._normalize_date_column(frame, path, "start_date")
        self._normalize_date_column(frame, path, "end_date")
        for column in ("libor_forward", "shift", "atm_caplet_vol", "correlation_to_ois"):
            if column in frame.columns:
                self._normalize_float_column(frame, path, column)
        for row_number, row in enumerate(frame.to_dict("records"), start=2):
            self._ensure_strict_date_interval(row["start_date"], row["end_date"], path, row_number)
        return frame.sort_values("start_date").reset_index(drop=True)
