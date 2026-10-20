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


def download(cfg: Config, dry_run: bool = False) -> None:
    """Fetch ENTSO-E and Open-Meteo per (country, year); past years are cached.

    With dry_run, report what is cached and what would be fetched without any
    request: the current year always refreshes, and a missing year would make
    `build` fail on the gap.
    """
    if dry_run:
        cached = fetched = 0
        for code in cfg.countries:
            for year in _years(cfg):
                states = [
                    _cache_state(code, year, source, cfg.raw_dir / source)
                    for source in ("entsoe", "weather")
                ]
                cached += states.count("cached")
                fetched += states.count("fetch") + states.count("refresh")
                log.info("%s %d: entsoe %s, weather %s", code, year, *states)
        log.info("%d cached, %d to fetch or refresh; no request was made", cached, fetched)
        return
    for code, country in cfg.countries.items():
        for year in _years(cfg):
            entsoe.download_year(code, year, cfg.raw_dir / "entsoe")
            weather.download_year(country, year, cfg.raw_dir / "weather")


def _cache_state(code: str, year: int, source: str, out_dir: Path) -> str:
    if not (out_dir / f"{code}_{year}.parquet").exists():
        return "fetch"
    return "refresh" if year >= pd.Timestamp.now(tz="UTC").year else "cached"


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


def with_forecast_weather(
    cfg: Config, code: str, raw: pd.DataFrame, days: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Replace the realised weather by the archived forecast in every hour that has one,
    and keep the `days` fully covered by forecasts.

    The hours before each issue time are replaced too, although their weather was
    already observed then; this can only make the models look worse, never better.
    Years before weather.FORECASTS_FROM have no forecasts and are not downloaded.
    """
    years = sorted({y for y in days.year if y >= weather.FORECASTS_FROM.year})
    if not years:
        return raw, days[:0]
    frames = [
        pd.read_parquet(
            weather.download_forecast_year(
                cfg.countries[code], year, cfg.raw_dir / "weather_forecast"
            )
        )
        for year in years
    ]
    forecast = harmonise(
        raw[["load", "tso_forecast"]],
        pd.concat(frames),
        raw.index[0],
        raw.index[-1] + pd.Timedelta("1h"),
    )
    columns = [c for c in COLUMNS if c not in ("load", "tso_forecast")]
    covered = forecast[columns].notna().all(axis=1)
    raw = raw.copy()
    raw.loc[covered, columns] = forecast.loc[covered, columns]
    full_days = covered.groupby(covered.index.floor("D")).sum() == 24
    return raw, days[days.isin(full_days.index[full_days])]
