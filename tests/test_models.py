"""Every model trains and predicts, and none of them looks into the future."""

import numpy as np
import pandas as pd
import pytest

from loadcast.backtest import make_folds
from loadcast.models import MODELS
from loadcast.models.base import qcol


@pytest.fixture(scope="module")
def fold(cfg, table):
    f = make_folds(table.index, cfg.test_years, cfg.validation_days)[0]
    f.test_days = f.test_days[:14]
    return f


@pytest.fixture(scope="module")
def fitted(cfg, table, fold):
    models = {}
    for name, cls in MODELS.items():
        model = cls(cfg)
        model.fit(table, fold.train_days, fold.valid_days)
        models[name] = model
    return models


@pytest.mark.parametrize("name", list(MODELS))
def test_predicts_full_days_with_ordered_quantiles(name, fitted, table, fold, cfg):
    pred = fitted[name].predict(table, fold.test_days)
    assert len(pred) == len(fold.test_days) * cfg.horizon_hours
    assert pred[qcol(0.5)].notna().all()
    if qcol(0.1) in pred:
        assert (np.diff(pred.to_numpy(), axis=1) >= 0).all()


@pytest.mark.parametrize("name", [n for n in MODELS if n != "tso"])
def test_no_leakage(name, fitted, table, fold, cfg):
    """Corrupting every load observed after the issue time must not change forecasts."""
    day = fold.test_days[7:8]
    issue = day[0] - pd.Timedelta("1D") + pd.Timedelta(hours=cfg.issue_hour_utc)

    raw = table[
        [
            "load",
            "tso_forecast",
            "temperature",
            "relative_humidity",
            "wind_speed",
            "shortwave_radiation",
        ]
    ].copy()
    raw.loc[raw.index >= issue, "load"] *= 10
    from loadcast.features import build

    corrupted = build(raw, "ES", "Europe/Madrid")
    before = fitted[name].predict(table, day)
    after = fitted[name].predict(corrupted, day)
    pd.testing.assert_frame_equal(before, after)
