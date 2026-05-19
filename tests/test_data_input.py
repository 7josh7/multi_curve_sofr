from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.bootstrap import build_full_curves, load_market_data
from src.config import load_engine_config
from src.data_input import CsvLiborCalibrationDataSource, CsvMarketDataSource, DataSchemaError, MarketData
from src.libor_model import build_joint_model_calibration


class InMemoryMarketDataSource:
    def __init__(self) -> None:
        self.loaded_config = None

    def load(self, config) -> MarketData:
        self.loaded_config = config
        return MarketData(
            config=config,
            fixings=pd.DataFrame({"date": [date(2025, 4, 15)], "sofr": [0.043]}),
            futures_1m=[],
            futures_3m=[],
            swaps=[],
            ois_curve=pd.DataFrame({"end_date": [date(2025, 5, 15)], "zero_rate": [0.042]}),
        )


class CountingMarketDataSource:
    def __init__(self, wrapped: CsvMarketDataSource) -> None:
        self.wrapped = wrapped
        self.load_count = 0

    def load(self, config) -> MarketData:
        self.load_count += 1
        return self.wrapped.load(config)


class CountingLiborCalibrationDataSource:
    def __init__(self, wrapped: CsvLiborCalibrationDataSource) -> None:
        self.wrapped = wrapped
        self.load_count = 0

    def load(self, config) -> pd.DataFrame:
        self.load_count += 1
        return self.wrapped.load(config)


def test_load_market_data_accepts_duck_typed_source() -> None:
    source = InMemoryMarketDataSource()

    market = load_market_data(".", data_source=source)

    assert market is not None
    assert source.loaded_config is market.config


def test_build_full_curves_uses_injected_market_data_source() -> None:
    config = load_engine_config(".")
    source = CountingMarketDataSource(CsvMarketDataSource(config.data_dir))

    result = build_full_curves(project_root=".", data_source=source)

    assert source.load_count == 1
    assert result.config is not None


def test_joint_model_calibration_uses_injected_data_sources() -> None:
    config = load_engine_config(".")
    market_source = CountingMarketDataSource(CsvMarketDataSource(config.data_dir))
    libor_source = CountingLiborCalibrationDataSource(CsvLiborCalibrationDataSource(config.data_dir))

    result = build_joint_model_calibration(
        project_root=".",
        market_data_source=market_source,
        libor_data_source=libor_source,
    )

    assert market_source.load_count == 1
    assert libor_source.load_count == 1
    assert len(result.calibration_table) > 0


def test_csv_market_data_source_normalizes_domain_inputs() -> None:
    config = load_engine_config(".")

    market = CsvMarketDataSource(config.data_dir).load(config)

    assert isinstance(market.fixings.loc[0, "date"], date)
    assert isinstance(market.ois_curve.loc[0, "end_date"], date)
    assert isinstance(market.futures_1m[0].start_date, date)
    assert isinstance(market.swaps[0].fixed_rate, float)


def test_csv_market_data_source_rejects_missing_required_columns(tmp_path: Path) -> None:
    _write_minimal_market_data(tmp_path)
    (tmp_path / "market" / "sofr_1m_futures.csv").write_text(
        "contract_code,start_date,end_date\n"
        "SR1M1,2025-04-15,2025-05-15\n",
        encoding="utf-8",
    )

    with pytest.raises(DataSchemaError, match="price"):
        CsvMarketDataSource(tmp_path).load(load_engine_config("."))


def _write_minimal_market_data(data_dir: Path) -> None:
    (data_dir / "fixings").mkdir(parents=True)
    (data_dir / "market").mkdir(parents=True)
    (data_dir / "fixings" / "sofr_fixings.csv").write_text(
        "date,sofr\n"
        "2025-04-14,0.043\n",
        encoding="utf-8",
    )
    (data_dir / "market" / "sofr_1m_futures.csv").write_text(
        "contract_code,start_date,end_date,price\n"
        "SR1M1,2025-04-15,2025-05-15,95.7\n",
        encoding="utf-8",
    )
    (data_dir / "market" / "sofr_3m_futures.csv").write_text(
        "contract_code,start_date,end_date,price\n"
        "SR3M1,2025-05-15,2025-08-15,95.4\n",
        encoding="utf-8",
    )
    (data_dir / "market" / "sofr_swaps.csv").write_text(
        "tenor,start_date,end_date,fixed_rate,pay_freq,day_count\n"
        "2Y,2025-04-15,2027-04-15,0.049,Annual,ACT/360\n",
        encoding="utf-8",
    )
    (data_dir / "market" / "ois_curve.csv").write_text(
        "tenor,end_date,zero_rate\n"
        "1M,2025-05-15,0.042\n",
        encoding="utf-8",
    )
