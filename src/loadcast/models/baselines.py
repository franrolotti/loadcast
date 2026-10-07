"""Reference forecasts every other model has to beat. See docs/models/baselines.md."""

from __future__ import annotations

import numpy as np
import pandas as pd

from loadcast.features import target_hours
from loadcast.models.base import Forecaster, qcol


class SeasonalNaive(Forecaster):
    """Same hour one week earlier."""

    name = "seasonal_naive"

    def predict(self, features: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        hours = target_hours(days, self.cfg.horizon_hours)
        return pd.DataFrame({qcol(0.5): features["load_lag168"].reindex(hours)})


class TSO(Forecaster):
    """Day-ahead load forecast published by the TSO on ENTSO-E."""

    name = "tso"

    def predict(self, features: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        hours = target_hours(days, self.cfg.horizon_hours)
        return pd.DataFrame({qcol(0.5): features["tso_forecast"].reindex(hours)})


class VanillaMLR(Forecaster):
    """Hong's "vanilla" multiple linear regression (GEFCom2012 benchmark).

    load = trend + month + weekday x hour + f(T) + f(T) x month + f(T) x hour,
    with f(T) a cubic polynomial in temperature. No lagged load.
    """

    name = "vanilla_mlr"

    def fit(self, features, train_days, valid_days) -> None:
        hours = target_hours(train_days.union(valid_days), self.cfg.horizon_hours)
        data = features.reindex(hours).dropna(subset=["load", "temperature"])
        self._origin = data.index[0]
        X = self._design(data)
        self._coef, *_ = np.linalg.lstsq(X, data["load"].to_numpy(), rcond=None)

    def predict(self, features: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        data = features.reindex(target_hours(days, self.cfg.horizon_hours))
        pred = self._design(data) @ self._coef
        return pd.DataFrame({qcol(0.5): pred}, index=data.index)

    def _design(self, data: pd.DataFrame) -> np.ndarray:
        years = (data.index - self._origin) / pd.Timedelta("365.25D")
        t = data["temperature"].to_numpy()
        poly = np.column_stack([t, t**2, t**3])
        month = _one_hot(data["month"].to_numpy() - 1, 12)
        hour = _one_hot(data["hour"].to_numpy(), 24)
        week_hour = _one_hot(data["dow"].to_numpy() * 24 + data["hour"].to_numpy(), 168)
        return np.column_stack(
            [
                np.asarray(years, dtype=float),
                month[:, 1:],  # week_hour already spans the intercept
                week_hour,
                poly,
                _interact(poly, month),
                _interact(poly, hour),
            ]
        )


def _one_hot(codes: np.ndarray, n: int) -> np.ndarray:
    return np.eye(n)[codes.astype(int)]


def _interact(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)
