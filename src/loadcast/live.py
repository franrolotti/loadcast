"""Operational mode: train on all history, forecast tomorrow every day, keep a record.

Unlike the backtest, live forecasts use the *weather forecast* available at issue
time (Open-Meteo forecast API), so the daily comparison with the TSO is like for
like (assumption A1 of SPEC.md §5 does not apply here).

History layout (the `data` branch, checked out under `history/`):

    forecasts/{country}/{day}.csv   one file per forecast day, never rewritten
    actuals/{country}/{day}.csv     observed load + TSO forecast, last 7 days refreshed
    runs/{day}.json                 status of every daily run
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time

import pandas as pd

from loadcast import features as feats
from loadcast import metrics
from loadcast.config import HISTORY_DIR, MODELS_DIR, Config
from loadcast.data import entsoe, weather
from loadcast.data.dataset import harmonise, load_processed
from loadcast.models import MODELS
from loadcast.models.base import qcol

log = logging.getLogger(__name__)

PAST_DAYS = 21  # covers the 168h lookback and the 336h lag, plus warm-up
ACTUALS_DAYS = 7  # ENTSO-E data get revised; re-record the last week every day


def train(cfg: Config) -> None:
    """Fit every model on all available data and pickle it under models/{country}/."""
    for code, country in cfg.countries.items():
        table = feats.build(load_processed(cfg, code), code, country.timezone)
        days = table.index[table["load"].notna()].floor("D").unique()
        train_days, valid_days = days[: -cfg.validation_days], days[-cfg.validation_days :]

        out = MODELS_DIR / code
        out.mkdir(parents=True, exist_ok=True)
        card = {
            "country": code,
            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "train_period": [str(train_days[0].date()), str(train_days[-1].date())],
            "validation_period": [str(valid_days[0].date()), str(valid_days[-1].date())],
            "git_sha": os.environ.get("GITHUB_SHA", "local"),
            "validation": {},
        }
        for name in cfg.models:
            log.info("training %s | %s", code, name)
            model = MODELS[name](cfg)
            model.fit(table, train_days, valid_days)
            pred = model.predict(table, valid_days)
            actual = table["load"].reindex(pred.index)
            ok = actual.notna() & pred[qcol(0.5)].notna()
            y, p = actual[ok].to_numpy(), pred.loc[ok, qcol(0.5)].to_numpy()
            card["validation"][name] = {"MAE": metrics.mae(y, p), "MAPE": metrics.mape(y, p)}
            with open(out / f"{name}.pkl", "wb") as f:
                pickle.dump(model, f)
        (out / "card.json").write_text(json.dumps(card, indent=2))


def forecast(cfg: Config, now: pd.Timestamp | None = None) -> None:
    """Forecast tomorrow (UTC) for every country and record results in HISTORY_DIR."""
    started = time.time()
    now = now or pd.Timestamp.now(tz="UTC")
    today = now.normalize()
    issue = today + pd.Timedelta(hours=cfg.issue_hour_utc)
    target = today + pd.Timedelta("1D")
    status = {}

    for code in cfg.countries:
        try:
            frame = _recent_frame(cfg, code, target)
            _write_actuals(code, frame, today)
            pred = predict_day(cfg, code, frame, issue, target)
            path = HISTORY_DIR / "forecasts" / code / f"{target.date()}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            pred.assign(issued_at=now.isoformat(timespec="seconds")).to_csv(path)
            status[code] = "ok"
        except Exception as exc:  # one country failing must not stop the others
            log.exception("%s failed", code)
            status[code] = f"error: {type(exc).__name__}: {exc}"[:300]

    run = {
        "run_at": now.isoformat(timespec="seconds"),
        "issue_time": issue.isoformat(),
        "target_day": str(target.date()),
        "git_sha": os.environ.get("GITHUB_SHA", "local"),
        "duration_s": round(time.time() - started, 1),
        "status": status,
    }
    path = HISTORY_DIR / "runs" / f"{today.date()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2))
    if not any(s == "ok" for s in status.values()):
        raise SystemExit("All countries failed; see the log above.")


def predict_day(
    cfg: Config, code: str, frame: pd.DataFrame, issue: pd.Timestamp, target: pd.Timestamp
) -> pd.DataFrame:
    """Forecasts of every saved model for `target`, using only data before `issue`."""
    frame = frame.copy()
    frame.loc[frame.index >= issue, "load"] = float("nan")  # enforce the information set
    frame["load"] = _patch_publication_delay(frame["load"], issue)
    table = feats.build(frame, code, cfg.countries[code].timezone)

    preds = []
    for name in cfg.models:
        path = MODELS_DIR / code / f"{name}.pkl"
        if not path.exists():
            log.warning("%s: no trained %s model, skipping", code, name)
            continue
        with open(path, "rb") as f:
            model = pickle.load(f)
        preds.append(model.predict(table, pd.DatetimeIndex([target])).assign(model=name))
    return pd.concat(preds).rename_axis("time")


def _recent_frame(cfg: Config, code: str, target: pd.Timestamp) -> pd.DataFrame:
    start = target - pd.Timedelta(days=PAST_DAYS)
    end = target + pd.Timedelta("1D")
    load = entsoe.fetch(code, start, end)
    wx = weather.recent_and_forecast(cfg.countries[code], past_days=PAST_DAYS + 1, forecast_days=3)
    return harmonise(load, wx, start, end)


def _patch_publication_delay(load: pd.Series, issue: pd.Timestamp) -> pd.Series:
    """ENTSO-E publishes with a lag of an hour or more. Fill missing hours in the last
    48h before the issue time with the value one week earlier (seasonal naive), so the
    sequence models still get a complete history."""
    recent = (load.index >= issue - pd.Timedelta("48h")) & (load.index < issue)
    missing = recent & load.isna().to_numpy()
    if missing.any():
        log.info("patching %d unpublished hours with t-168h values", missing.sum())
    return load.where(~missing, load.shift(168))


def _write_actuals(code: str, frame: pd.DataFrame, today: pd.Timestamp) -> None:
    for k in range(1, ACTUALS_DAYS + 1):
        day = today - pd.Timedelta(days=k)
        rows = frame.loc[day : day + pd.Timedelta("23h"), ["load", "tso_forecast"]]
        if rows["load"].notna().any():
            path = HISTORY_DIR / "actuals" / code / f"{day.date()}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            rows.to_csv(path)
