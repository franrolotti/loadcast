"""Population-weighted hourly weather from Open-Meteo (no key needed).

- Backtest: the archive API (ERA5/ERA5-Land reanalysis). Using realised weather as a
  model input amounts to assuming a *perfect weather forecast* (SPEC.md §5, A1).
- Live: the forecast API, i.e. the weather forecast actually available at issue time.
- Backtest without A1: the Previous Runs API, archived forecasts issued 48h before each
  hour ("previous_day2"), available from 20 January 2024.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from loadcast.config import Country

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECASTS_FROM = pd.Timestamp("2024-01-20", tz="UTC")  # first archived forecast
TIMEOUT_S, ATTEMPTS = 60, 4  # the API occasionally stalls: retry with a growing wait
VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
]


def _fetch_cities(
    country: Country, url: str, suffix: str = "", **period: str | int
) -> list[pd.DataFrame]:
    params = {
        "latitude": ",".join(str(c.lat) for c in country.cities),
        "longitude": ",".join(str(c.lon) for c in country.cities),
        "hourly": ",".join(v + suffix for v in VARIABLES),
        "timezone": "UTC",
        **period,
    }
    payload = _get(url, params)
    payload = payload if isinstance(payload, list) else [payload]
    frames = []
    for location in payload:
        hourly = location["hourly"]
        index = pd.to_datetime(hourly.pop("time")).tz_localize("UTC")
        frame = pd.DataFrame(hourly, index=index, dtype=float)
        frames.append(frame.rename(columns=lambda c: c.removesuffix(suffix)))
    return frames


def _get(url: str, params: dict) -> dict | list:
    for attempt in range(1, ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT_S)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if attempt == ATTEMPTS:
                raise
            log.warning("Open-Meteo attempt %d failed (%s), retrying", attempt, exc)
            time.sleep(15 * attempt)
    raise AssertionError("unreachable")


def population_weighted(frames: list[pd.DataFrame], weights: list[float]) -> pd.DataFrame:
    """Weighted average across locations, renormalising weights when values are NaN."""
    w = np.asarray(weights, dtype=float)
    stacked = np.stack([f.to_numpy(dtype=float) for f in frames])  # (city, time, var)
    mask = ~np.isnan(stacked)
    weighted = np.nansum(stacked * w[:, None, None], axis=0)
    total = (mask * w[:, None, None]).sum(axis=0)
    with np.errstate(invalid="ignore"):
        values = weighted / total
    return pd.DataFrame(values, index=frames[0].index, columns=frames[0].columns)


def _national(country: Country, frames: list[pd.DataFrame]) -> pd.DataFrame:
    weather = population_weighted(frames, [c.pop for c in country.cities])
    weather = weather.rename(
        columns={
            "temperature_2m": "temperature",
            "relative_humidity_2m": "relative_humidity",
            "wind_speed_10m": "wind_speed",
        }
    )
    weather.index.name = "time"
    return weather


def download_year(country: Country, year: int, out_dir: Path) -> Path:
    path = out_dir / f"{country.code}_{year}.parquet"
    today = pd.Timestamp.now(tz="UTC").normalize()
    if path.exists() and year < today.year:
        return path

    # The archive lags real time by a few days; the missing tail stays NaN.
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = max(start, min(pd.Timestamp(f"{year}-12-31", tz="UTC"), today - pd.Timedelta("1D")))
    frames = _fetch_cities(
        country,
        ARCHIVE_URL,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )
    weather = _national(country, frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(path)
    log.info("Open-Meteo %s %d: %d hours", country.code, year, len(weather))
    return path


def download_forecast_year(country: Country, year: int, out_dir: Path) -> Path:
    """Archived forecasts issued 48h before each hour of `year` (NaN before 2024-01-20).

    48h is longer than our longest lead time (38h, SPEC §2), so the forecast was always
    available at the issue time and is, if anything, slightly worse than the real one.
    """
    path = out_dir / f"{country.code}_{year}.parquet"
    if path.exists() and year < pd.Timestamp.now(tz="UTC").year:
        return path
    weather = archived_forecast(country, f"{year}-01-01", f"{year}-12-31")
    out_dir.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(path)
    log.info(
        "Open-Meteo forecasts %s %d: %d hours", country.code, year, weather.notna().all(1).sum()
    )
    return path


def archived_forecast(country: Country, start_date: str, end_date: str) -> pd.DataFrame:
    """Forecasts issued 48h before each hour of the days start_date to end_date."""
    frames = _fetch_cities(
        country,
        PREVIOUS_RUNS_URL,
        suffix="_previous_day2",
        start_date=start_date,
        end_date=end_date,
    )
    return _national(country, frames)


def recent_and_forecast(country: Country, past_days: int, forecast_days: int) -> pd.DataFrame:
    """Recent weather and the current forecast, as known right now."""
    frames = _fetch_cities(country, FORECAST_URL, past_days=past_days, forecast_days=forecast_days)
    return _national(country, frames)
