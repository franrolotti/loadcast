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
FORECASTS_FROM = 2024  # first year with archived weather forecasts


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


def run(cfg: Config, run: str | None = None, weather: str = "reanalysis") -> None:
    """With weather="forecast", the test days use archived weather forecasts instead of
    the realised weather (removes assumption A1, SPEC §5); only test days from
    2024-01-20 on have them, and the results go to {country}_forecast_weather.parquet."""
    feats.check_information_set(cfg.issue_hour_utc, cfg.horizon_hours)
    years = [y for y in cfg.test_years if weather == "reanalysis" or y >= FORECASTS_FROM]
    jobs = [(code, year) for code in cfg.countries for year in years]
    workers = min(len(jobs), max(1, (os.cpu_count() or 2) - 1))
    log.info("%d folds on %d processes", len(jobs), workers)
    with ProcessPoolExecutor(workers, initializer=_log_to_stderr) as pool:
        preds = list(
            pool.map(_fold, [cfg] * len(jobs), *zip(*jobs, strict=True), [weather] * len(jobs))
        )

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for code in cfg.countries:
        out = pd.concat([p for (c, _), p in zip(jobs, preds, strict=True) if c == code])
        suffix = "" if weather == "reanalysis" else "_forecast_weather"
        path = PREDICTIONS_DIR / f"{code}{suffix}.parquet"
        merge(out, pd.read_parquet(path) if path.exists() else None).to_parquet(path)
        if run and weather == "reanalysis":
            _attach(cfg, run, code, out)


def merge(new: pd.DataFrame, old: pd.DataFrame | None) -> pd.DataFrame:
    """Replace the (model, test year) pairs that were re-run and keep the others, so a
    backtest filtered by model or country does not drop earlier results."""
    if old is None:
        return new
    rerun = pd.MultiIndex.from_frame(new[["model", "test_year"]].drop_duplicates())
    kept = old[~pd.MultiIndex.from_frame(old[["model", "test_year"]]).isin(rerun)]
    for model in sorted(set(kept["model"]) - set(new["model"])):
        log.info("keeping earlier backtest of %s", model)
    return pd.concat([kept, new])


def _log_to_stderr() -> None:
    """Worker processes start without the CLI's logging set-up (spawn on macOS)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")


def _fold(cfg: Config, code: str, year: int, weather: str = "reanalysis") -> pd.DataFrame:
    """Train every model on the days before `year` and forecast every day of it.

    Training always uses the realised weather, as the live models do; with
    weather="forecast" only the inputs of the test days change.
    """
    from loadcast.models import MODELS, with_baselines  # each worker sets up torch itself

    timezone = cfg.countries[code].timezone
    raw = load_processed(cfg, code)
    table = feats.build(raw, code, timezone)
    fold = make_folds(table.index, [year], cfg.validation_days)[0]
    test_table = table
    if weather == "forecast":
        raw, fold.test_days = _with_forecast_weather(cfg, code, raw, fold.test_days)
        test_table = feats.build(raw, code, timezone)
    predictions = []
    for name in with_baselines(cfg.models):
        log.info("%s | %d | %s", code, year, name)
        model = MODELS[name](cfg)
        model.fit(table, fold.train_days, fold.valid_days)
        predictions.append(
            model.predict(test_table, fold.test_days).assign(model=name, test_year=year)
        )
    out = pd.concat(predictions)
    out["actual"] = table["load"].reindex(out.index).to_numpy()
    out.index.name = "time"
    return out


def _with_forecast_weather(
    cfg: Config, code: str, raw: pd.DataFrame, days: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Replace the realised weather by the archived forecast wherever there is one, and
    keep the test days fully covered by forecasts. Earlier hours keep the realised
    weather, which was observed by the issue time."""
    from loadcast.data import dataset, weather

    frames = [
        pd.read_parquet(
            weather.download_forecast_year(
                cfg.countries[code], year, cfg.raw_dir / "weather_forecast"
            )
        )
        for year in sorted(set(days.year))
    ]
    forecast = dataset.harmonise(
        raw[["load", "tso_forecast"]],
        pd.concat(frames),
        raw.index[0],
        raw.index[-1] + pd.Timedelta("1h"),
    )
    columns = [c for c in dataset.COLUMNS if c not in ("load", "tso_forecast")]
    covered = forecast[columns].notna().all(axis=1)
    raw = raw.copy()
    raw.loc[covered, columns] = forecast.loc[covered, columns]
    full_days = covered.groupby(covered.index.floor("D")).sum() == 24
    return raw, days[days.isin(full_days.index[full_days])]


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
