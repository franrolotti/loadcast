"""Common interface: every model is fit on training days and predicts whole days."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
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
    # Learning curve of the last fit, for models trained iteratively: {"metric",
    # "train", "valid", "best", optional "step"}. Recorded in the model card.
    curve: dict | None = None
    # Conformal shift (MW) of each outer quantile column, set by `calibrate`.
    offsets: dict[str, float] | None = None

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

    def calibrate(self, features: pd.DataFrame, days: pd.DatetimeIndex) -> None:
        """Conformalised quantiles (Romano et al., 2019), one tail at a time.

        Quantile models trained with the pinball loss are overconfident out of sample.
        Each outer quantile q_tau is shifted by the tau-quantile of the residuals
        y - q_tau on `days`, so that a share tau of those observations falls below it.
        The median is left alone: point forecasts do not change.
        """
        self.offsets = None
        pred = self.predict(features, days)
        actual = features["load"].reindex(pred.index)
        known = actual.notna()
        self.offsets = {
            qcol(q): float(np.quantile(actual[known] - pred.loc[known, qcol(q)], q))
            for q in self.cfg.quantiles
            if q != 0.5 and qcol(q) in pred
        }

    def _calibrated(self, pred: pd.DataFrame) -> pd.DataFrame:
        for column, shift in (self.offsets or {}).items():
            pred[column] += shift
        sorted_ = np.sort(pred.to_numpy(), axis=1)  # shifted quantiles may cross
        return pd.DataFrame(sorted_, index=pred.index, columns=pred.columns)
