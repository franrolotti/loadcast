"""Synthetic but realistic hourly data so the whole pipeline runs without API keys."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loadcast.config import Config
from loadcast.features import build

ROOT = Path(__file__).resolve().parents[1]


def synthetic_country(
    start: str = "2021-01-01", end: str = "2022-12-31", seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta("23h"), freq="1h", tz="UTC")
    hour = index.hour.to_numpy()
    doy = index.dayofyear.to_numpy()
    weekend = index.dayofweek.to_numpy() >= 5
    temperature = 15 - 10 * np.cos(2 * np.pi * doy / 365) + 5 * np.sin(2 * np.pi * (hour - 9) / 24)
    temperature += rng.normal(0, 1.5, len(index))
    daily = 1 + 0.25 * np.sin(np.pi * np.clip(hour - 6, 0, 16) / 16)
    load = 30_000 * daily * np.where(weekend, 0.85, 1.0)
    load += 400 * np.clip(15.5 - temperature, 0, None) + 600 * np.clip(temperature - 22, 0, None)
    load += rng.normal(0, 500, len(index))
    return pd.DataFrame(
        {
            "load": load,
            "tso_forecast": load + rng.normal(0, 800, len(index)),
            "temperature": temperature,
            "relative_humidity": 60 + rng.normal(0, 10, len(index)),
            "wind_speed": np.abs(rng.normal(12, 4, len(index))),
            "shortwave_radiation": np.clip(600 * np.sin(np.pi * (hour - 6) / 12), 0, None),
        },
        index=index,
    ).rename_axis("time")


@pytest.fixture(scope="session")
def cfg() -> Config:
    base = Config.load(ROOT / "config.yaml")
    small = {
        "xgboost": {
            **base.params["xgboost"],
            "n_estimators": 50,
            "early_stopping_rounds": 10,
        },
        "torch": {**base.params["torch"], "max_epochs": 2, "patience": 1},
        "lstm": {
            **base.params["lstm"],
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "transformer": {**base.params["transformer"], "d_model": 16, "n_layers": 1},
    }
    return Config(
        **{
            **base.__dict__,
            "test_years": [2022],
            "validation_days": 60,
            "params": small,
        }
    )


@pytest.fixture(scope="session")
def table() -> pd.DataFrame:
    return build(synthetic_country(), "ES", "Europe/Madrid")
