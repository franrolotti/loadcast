"""Gradient-boosted trees with a multi-quantile objective. See docs/models/xgboost.md."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from loadcast.features import KNOWN_FUTURE, LAG_FEATURES, target_hours
from loadcast.models.base import Forecaster, qcol

FEATURES = KNOWN_FUTURE + LAG_FEATURES


class XGBoost(Forecaster):
    name = "xgboost"

    def fit(self, features, train_days, valid_days) -> None:
        train = self._rows(features, train_days).dropna(subset=["load"])
        valid = self._rows(features, valid_days).dropna(subset=["load"])
        params = dict(self.cfg.params["xgboost"])
        self.model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=np.array(self.cfg.quantiles),
            tree_method="hist",
            random_state=self.cfg.params["torch"].get("seed", 0),
            **params,
        )
        self.model.fit(
            train[FEATURES],
            train["load"],
            eval_set=[(valid[FEATURES], valid["load"])],
            verbose=False,
        )

    def predict(self, features: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        rows = self._rows(features, days)
        pred = self.model.predict(rows[FEATURES]).reshape(len(rows), -1)
        pred = np.sort(pred, axis=1)  # independent quantile fits can cross
        return pd.DataFrame(pred, index=rows.index, columns=[qcol(q) for q in self.cfg.quantiles])

    def _rows(self, features: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        return features.reindex(target_hours(days, self.cfg.horizon_hours))
