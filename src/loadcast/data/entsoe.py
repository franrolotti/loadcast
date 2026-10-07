"""Actual load and TSO day-ahead load forecast from the ENTSO-E Transparency Platform.

Thin wrapper around entsoe-py. Historical data are cached as one parquet per
(country, year) so that interrupted downloads resume and re-runs are free; the
current year is always refreshed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient

log = logging.getLogger(__name__)

TIMEOUT_S = 120


def _client() -> EntsoePandasClient:
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("ENTSOE_API_KEY is not set. Copy .env.example to .env.")
    # Without a timeout a stalled connection hangs the download forever; entsoe-py
    # retries timed-out requests (retry_count) before giving up.
    return EntsoePandasClient(api_key=api_key, timeout=TIMEOUT_S)


def _to_hourly_utc(df: pd.DataFrame) -> pd.Series:
    """Single-column ENTSO-E frame (15/30/60 min, local tz) -> hourly mean in UTC.

    Timestamps mark the *start* of each interval (ENTSO-E convention), so the value
    at 08:00 is the mean load over 08:00-09:00. Zero or negative values are how
    missing data often shows up on the platform and are dropped before averaging.
    """
    series = df.iloc[:, 0].astype(float).tz_convert("UTC")
    series = series[~series.index.duplicated()]
    return series[series > 0].resample("1h").mean()


def fetch(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Hourly UTC `load` and `tso_forecast` (MW) on [start, end)."""
    client = _client()
    load = _to_hourly_utc(client.query_load(country, start=start, end=end))
    tso = _to_hourly_utc(client.query_load_forecast(country, start=start, end=end))
    frame = pd.DataFrame({"load": load, "tso_forecast": tso})
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    frame.index.name = "time"
    return frame


def download_year(country: str, year: int, out_dir: Path) -> Path:
    path = out_dir / f"{country}_{year}.parquet"
    now = pd.Timestamp.now(tz="UTC")
    if path.exists() and year < now.year:
        return path

    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    # The TSO forecast is published up to D+1, so never ask beyond that.
    end = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), now.normalize() + pd.Timedelta("2D"))
    frame = fetch(country, start, end)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    log.info("ENTSO-E %s %d: %d hours", country, year, len(frame))
    return path
