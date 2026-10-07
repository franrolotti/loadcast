"""Expanding-window backtest: one fold per test year, every model, every country.

Folds are independent, so they run in parallel processes (one per country and test
year). With `run`, the scores are also stored in the cards of that training run,
which `loadcast publish --run` then uploads.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from loadcast import features as feats
from loadcast.config import MODELS_DIR, Config
from loadcast.data.dataset import load_processed

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


def run(cfg: Config, run: str | None = None) -> None:
    feats.check_information_set(cfg.issue_hour_utc, cfg.horizon_hours)
    jobs = [(code, year) for code in cfg.countries for year in cfg.test_years]
    workers = min(len(jobs), max(1, (os.cpu_count() or 2) - 1))
    log.info("%d folds on %d processes", len(jobs), workers)
    with ProcessPoolExecutor(workers) as pool:
        preds = list(pool.map(_fold, [cfg] * len(jobs), *zip(*jobs, strict=True)))

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for code in cfg.countries:
        out = pd.concat([p for (c, _), p in zip(jobs, preds, strict=True) if c == code])
        out.to_parquet(PREDICTIONS_DIR / f"{code}.parquet")
        if run:
            _attach(cfg, run, code, out)


def _fold(cfg: Config, code: str, year: int) -> pd.DataFrame:
    """Train every model on the days before `year` and forecast every day of it."""
    from loadcast.models import MODELS  # imported here: each worker sets up torch itself

    table = feats.build(load_processed(cfg, code), code, cfg.countries[code].timezone)
    fold = make_folds(table.index, [year], cfg.validation_days)[0]
    predictions = []
    for name in cfg.models:
        log.info("%s | %d | %s", code, year, name)
        model = MODELS[name](cfg)
        model.fit(table, fold.train_days, fold.valid_days)
        predictions.append(model.predict(table, fold.test_days).assign(model=name, test_year=year))
    out = pd.concat(predictions)
    out["actual"] = table["load"].reindex(out.index).to_numpy()
    out.index.name = "time"
    return out


def _attach(cfg: Config, run: str, code: str, pred: pd.DataFrame) -> None:
    from loadcast.live import _git_sha, scores

    path = MODELS_DIR / run / code / "card.json"
    if not path.exists():
        raise SystemExit(f"No card for run {run} and {code}: train it first")
    card = json.loads(path.read_text())
    sha = _git_sha()
    if sha != card["git_sha"]:
        log.warning(
            "run %s was trained at %s but the backtest ran at %s", run, card["git_sha"], sha
        )
    card["backtest"] = {
        "test_years": sorted(int(y) for y in pred["test_year"].unique()),
        "git_sha": sha,
        "scores": scores(pred, cfg.quantiles),
    }
    path.write_text(json.dumps(card, indent=2))
    log.info("backtest stored in %s; upload it with: loadcast publish --run %s", path, run)
