import logging

import numpy as np
import pandas as pd
import pytest

from loadcast.data import dataset
from loadcast.data.dataset import _fill_isolated_gaps
from loadcast.data.entsoe import _to_hourly_utc


def test_entsoe_quarter_hours_to_hourly_utc_dropping_zeros():
    index = pd.date_range("2024-01-01 00:00", periods=8, freq="15min", tz="Europe/Madrid")
    raw = pd.DataFrame({"Actual Load": [100, 100, 0, 100, 200, 200, 200, 200]}, index=index)
    hourly = _to_hourly_utc(raw)
    # Local midnight in Madrid is 23:00 UTC; the zero is treated as missing.
    assert hourly.index[0] == pd.Timestamp("2023-12-31 23:00", tz="UTC")
    assert hourly.tolist() == [100.0, 200.0]


def test_only_isolated_gaps_are_filled():
    s = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
    out = _fill_isolated_gaps(s)
    assert out.iloc[1] == 2.0
    assert out.iloc[3:5].isna().all()


def test_dry_run_reports_the_cache_without_downloading(cfg, tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(
        dataset.entsoe, "download_year", lambda *a: pytest.fail("dry run must not download")
    )
    monkeypatch.setattr(
        dataset.weather, "download_year", lambda *a: pytest.fail("dry run must not download")
    )
    raw = tmp_path / "raw"
    year = pd.Timestamp.now().year
    isolated = type(cfg)(
        **{
            **cfg.__dict__,
            "raw_dir": raw,
            "end": f"{year}-12-31",
            "countries": {"ES": cfg.countries["ES"]},
        }
    )
    (raw / "entsoe").mkdir(parents=True)
    pd.DataFrame().to_parquet(raw / "entsoe" / f"ES_{year}.parquet")
    with caplog.at_level(logging.INFO):
        dataset.download(isolated, dry_run=True)
    assert f"ES {year}: entsoe refresh, weather fetch" in caplog.text
    assert "ES 2018: entsoe fetch, weather fetch" in caplog.text
    assert "no request was made" in caplog.text
