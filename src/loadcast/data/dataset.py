"""Download raw sources and assemble one hourly UTC table per country."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from loadcast.config import Config
from loadcast.data import entsoe, weather

log = logging.getLogger(__name__)

COLUMNS = [
    "load",
    "tso_forecast",
    "temperature",
    "relative_humidity",
    "wind_speed",
    "shortwave_radiation",
]


def _years(cfg: Config) -> range:
    return range(pd.Timestamp(cfg.start).year, pd.Timestamp(cfg.end).year + 1)


def download(cfg: Config) -> None:
    for code, country in cfg.countries.items():
        for year in _years(cfg):
            entsoe.download_year(code, year, cfg.raw_dir / "entsoe")
            weather.download_year(country, year, cfg.raw_dir / "weather")


def harmonise(
    load: pd.DataFrame, wx: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Join ENTSO-E and weather on a complete hourly UTC index [start, end) and clean.

    Used for both the historical table and the live forecast, so that the models
    always see identically prepared data.
    """
    # Open-Meteo radiation at hh:00 is the mean over the *preceding* hour; move it one
    # hour back so every row describes the interval [hh:00, hh+1:00), like the load.
    wx = wx.drop(columns="shortwave_radiation").join(
        wx["shortwave_radiation"].shift(-1, freq="1h"), how="outer"
    )
    index = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    frame = load.join(wx, how="outer").reindex(index)[COLUMNS]
    frame.index.name = "time"
    # Fill isolated one-hour gaps in the load (as DemandCast does); longer gaps
    # stay missing and the affected days are skipped by the models.
    frame["load"] = _fill_isolated_gaps(frame["load"])
    return frame


def build(cfg: Config) -> None:
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    for code in cfg.countries:
        load = pd.concat(
            pd.read_parquet(cfg.raw_dir / "entsoe" / f"{code}_{y}.parquet") for y in _years(cfg)
        )
        wx = pd.concat(
            pd.read_parquet(cfg.raw_dir / "weather" / f"{code}_{y}.parquet") for y in _years(cfg)
        )
        start = pd.Timestamp(cfg.start, tz="UTC")
        end = pd.Timestamp(cfg.end, tz="UTC") + pd.Timedelta("1D")
        frame = harmonise(load, wx, start, end)
        _log_quality(code, frame)
        frame.to_parquet(processed_path(cfg, code))


def load_processed(cfg: Config, code: str) -> pd.DataFrame:
    return pd.read_parquet(processed_path(cfg, code))


def processed_path(cfg: Config, code: str) -> Path:
    return cfg.processed_dir / f"{code}.parquet"


def _fill_isolated_gaps(series: pd.Series) -> pd.Series:
    isolated = series.isna() & series.shift(1).notna() & series.shift(-1).notna()
    return series.where(~isolated, (series.shift(1) + series.shift(-1)) / 2)


def _log_quality(code: str, frame: pd.DataFrame) -> None:
    missing = frame.isna().mean().mul(100).round(2)
    log.info("%s: %d hours, %% missing:\n%s", code, len(frame), missing.to_string())
