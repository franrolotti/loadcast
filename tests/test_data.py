import logging

import numpy as np
import pandas as pd
import pytest

from loadcast.data import dataset
from loadcast.data.dataset import _fill_isolated_gaps
from loadcast.data.entsoe import _to_hourly_utc
from tests.conftest import synthetic_country


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


def test_forecast_weather_replaces_covered_hours_only(cfg, tmp_path, monkeypatch):
    raw = synthetic_country("2024-01-15", "2024-01-25")
    weather_columns = ["temperature", "relative_humidity", "wind_speed", "shortwave_radiation"]
    forecast = raw.loc["2024-01-20":, weather_columns] + 1.0  # archive starts 2024-01-20

    def download(country, year, out_dir):
        forecast.to_parquet(tmp_path / "forecast.parquet")
        return tmp_path / "forecast.parquet"

    monkeypatch.setattr(dataset.weather, "download_forecast_year", download)
    days = raw.index.floor("D").unique()
    out, covered = dataset.with_forecast_weather(cfg, "ES", raw, days)
    # The last hour of the archive has no radiation once it is moved one hour back.
    assert list(covered) == list(days[(days >= "2024-01-20") & (days < "2024-01-25")])
    assert (
        (out.loc[:"2024-01-19", weather_columns] == raw.loc[:"2024-01-19", weather_columns])
        .all()
        .all()
    )
    assert (out.loc["2024-01-21", "temperature"] == raw.loc["2024-01-21", "temperature"] + 1).all()
    assert (out["load"] == raw["load"]).all()

    # Years without archived forecasts are not downloaded.
    monkeypatch.setattr(dataset.weather, "download_forecast_year", None)
    before = pd.DatetimeIndex(["2023-06-01"], tz="UTC")
    assert dataset.with_forecast_weather(cfg, "ES", raw, before)[1].empty
