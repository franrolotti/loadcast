"""Expanding-window backtest: one fold per test year, every model, every country."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from loadcast import features as feats
from loadcast.config import Config
from loadcast.data.dataset import load_processed
from loadcast.models import MODELS

log = logging.getLogger(__name__)

PREDICTIONS_DIR = Path("results/predictions")


@dataclass
class Fold:
    test_year: int
    train_days: pd.DatetimeIndex
    valid_days: pd.DatetimeIndex
    test_days: pd.DatetimeIndex


def make_folds(index: pd.DatetimeIndex, test_years: list[int], validation_days: int) -> list[Fold]:
    days = index.floor("D").unique()
    folds = []
    for year in test_years:
        before = days[days.year < year]
        folds.append(
            Fold(
                test_year=year,
                train_days=before[:-validation_days],
                valid_days=before[-validation_days:],
                test_days=days[days.year == year],
            )
        )
    return folds


def run(cfg: Config, code: str) -> pd.DataFrame:
    feats.check_information_set(cfg.issue_hour_utc, cfg.horizon_hours)
    country = cfg.countries[code]
    table = feats.build(load_processed(cfg, code), code, country.timezone)

    predictions = []
    for fold in make_folds(table.index, cfg.test_years, cfg.validation_days):
        for name in cfg.models:
            log.info("%s | %d | %s", code, fold.test_year, name)
            model = MODELS[name](cfg)
            model.fit(table, fold.train_days, fold.valid_days)
            pred = model.predict(table, fold.test_days)
            pred["model"], pred["test_year"] = name, fold.test_year
            predictions.append(pred)

    out = pd.concat(predictions)
    out["actual"] = table["load"].reindex(out.index).to_numpy()
    out.index.name = "time"
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PREDICTIONS_DIR / f"{code}.parquet")
    return out
