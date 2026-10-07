"""Common interface: every model is fit on training days and predicts whole days."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from loadcast.config import Config


def qcol(q: float) -> str:
    return f"q{q:g}"


class Forecaster(ABC):
    """A day-ahead forecaster.

    `features` is the full output of `loadcast.features.build`. It contains the load
    for every hour, so it is each model's responsibility to only use information
    available at the issue time. tests/test_leakage.py enforces this.
    """

    name: str

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def fit(
        self,
        features: pd.DataFrame,
        train_days: pd.DatetimeIndex,
        valid_days: pd.DatetimeIndex,
    ) -> None:
        """Estimate parameters. Default: nothing to estimate."""

    @abstractmethod
    def predict(self, features: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        """Hourly forecasts for `days` (UTC midnights).

        Returns a frame indexed by target hour with one column per quantile (`q0.5`
        for point forecasters).
        """
