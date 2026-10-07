"""Feature engineering shared by all models.

Two groups, distinguished by *when* they are known (SPEC.md §2-§4):

- KNOWN_FUTURE: calendar and weather. Known for the target day at issue time
  (calendar trivially, weather under the perfect-forecast assumption).
- Lags of the load. Only lags >= MIN_LAG_HOURS are used, so every target hour of
  day D is computable at the issue time on D-1.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

# Degree-day thresholds (°C). Load is roughly flat between them and rises outside.
HEATING_BASE = 15.5
COOLING_BASE = 22.0

MIN_LAG_HOURS = 48
LAGS = [48, 72, 168, 336]

CALENDAR = [
    "hour",
    "dow",
    "month",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_holiday",
    "is_bridge",
]
WEATHER = [
    "temperature",
    "hdd",
    "cdd",
    "temperature_ema24",
    "relative_humidity",
    "wind_speed",
    "shortwave_radiation",
]
KNOWN_FUTURE = CALENDAR + WEATHER
LAG_FEATURES = [f"load_lag{lag}" for lag in LAGS] + ["load_lag48_mean24"]


def check_information_set(issue_hour_utc: int, horizon_hours: int) -> None:
    """The longest lead time (issue on D-1 -> last hour of D) must not exceed MIN_LAG."""
    max_lead = (24 - issue_hour_utc) + horizon_hours
    if max_lead > MIN_LAG_HOURS:
        raise ValueError(
            f"Max lead time {max_lead}h exceeds the minimum load lag "
            f"({MIN_LAG_HOURS}h): lag features would leak future information."
        )


def calendar_features(index: pd.DatetimeIndex, country: str, timezone: str) -> pd.DataFrame:
    """Calendar in *local* time: demand follows people's clocks, not UTC."""
    local = index.tz_convert(timezone)
    local_date = local.normalize().tz_localize(None)

    days = pd.date_range(
        local_date.min() - pd.Timedelta("1D"),
        local_date.max() + pd.Timedelta("1D"),
        freq="D",
    )
    national = holidays.country_holidays(country, years=range(days.year.min(), days.year.max() + 1))
    holiday = pd.Series([d in national for d in days.date], index=days)
    weekend = pd.Series(days.dayofweek >= 5, index=days)
    off = holiday | weekend
    # A "bridge" is a working day squeezed between two days off (e.g. Fri after a
    # Thursday holiday). Many people take it off, so load looks like a weekend.
    bridge = ~off & off.shift(1, fill_value=False) & off.shift(-1, fill_value=False)

    doy = 2 * np.pi * local.dayofyear.to_numpy() / 365.25
    return pd.DataFrame(
        {
            "hour": local.hour,
            "dow": local.dayofweek,
            "month": local.month,
            "doy_sin": np.sin(doy),
            "doy_cos": np.cos(doy),
            "is_weekend": weekend.reindex(local_date).to_numpy(dtype=int),
            "is_holiday": holiday.reindex(local_date).to_numpy(dtype=int),
            "is_bridge": bridge.reindex(local_date).to_numpy(dtype=int),
        },
        index=index,
    )


def weather_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["temperature"]
    return pd.DataFrame(
        {
            "temperature": t,
            "hdd": (HEATING_BASE - t).clip(lower=0),
            "cdd": (t - COOLING_BASE).clip(lower=0),
            # Buildings have thermal inertia: load reacts to a smoothed temperature.
            "temperature_ema24": t.ewm(span=24, adjust=False).mean(),
            "relative_humidity": df["relative_humidity"],
            "wind_speed": df["wind_speed"],
            # Behind-the-meter PV lowers the load measured by the TSO on sunny hours.
            "shortwave_radiation": df["shortwave_radiation"],
        },
        index=df.index,
    )


def lag_features(load: pd.Series) -> pd.DataFrame:
    out = {f"load_lag{lag}": load.shift(lag) for lag in LAGS}
    out["load_lag48_mean24"] = load.shift(MIN_LAG_HOURS).rolling(24).mean()
    return pd.DataFrame(out, index=load.index)


def build(df: pd.DataFrame, country: str, timezone: str) -> pd.DataFrame:
    """All features plus the target, on the hourly UTC index of `df`."""
    return pd.concat(
        [
            calendar_features(df.index, country, timezone),
            weather_features(df),
            lag_features(df["load"]),
            df[["load", "tso_forecast"]],
        ],
        axis=1,
    )


def target_hours(days: pd.DatetimeIndex, horizon_hours: int = 24) -> pd.DatetimeIndex:
    """All hourly timestamps covered by the given forecast days (UTC midnights)."""
    offsets = pd.to_timedelta(np.arange(horizon_hours), unit="h")
    return pd.DatetimeIndex((days.values[:, None] + offsets.values).ravel(), tz="UTC")
