"""Operational mode: train on all history, forecast tomorrow every day, keep a record.

Unlike the backtest, live forecasts use the *weather forecast* available at issue
time (Open-Meteo forecast API), so the daily comparison with the TSO is like for
like (assumption A1 of SPEC.md §5 does not apply here).

Training is done by hand. Every training run gets an id and is kept under
`models/{run}/{country}/`; `publish` uploads it as the release `models-{run}` and
live.yaml chooses which (run, model) pairs make the daily forecast.

History layout (the `data` branch, checked out under `history/`):

    forecasts/{country}/{day}.csv   one file per forecast day; rows are only ever added
    actuals/{country}/{day}.csv     observed load + TSO forecast, last 7 days refreshed
    runs/{day}.json                 status of every daily run
    cards/{run}/{country}.json      model card of every run that has forecast
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import tarfile
import time

import pandas as pd
import yaml

from loadcast import features as feats
from loadcast import metrics
from loadcast.config import HISTORY_DIR, LIVE_FILE, MODELS_DIR, Config
from loadcast.data import entsoe, weather
from loadcast.data.dataset import harmonise, load_processed
from loadcast.models import MODELS
from loadcast.models.base import qcol

log = logging.getLogger(__name__)

PAST_DAYS = 21  # covers the 168h lookback and the 336h lag, plus warm-up
ACTUALS_DAYS = 7  # ENTSO-E data get revised; re-record the last week every day


def train(cfg: Config, run: str | None = None) -> str:
    """Fit every model and pickle it under models/{run}/{country}/.

    The last `cfg.test_days` are held out of training and early stopping; the card
    records how the saved models (and the TSO) did on them.
    """
    run = run or pd.Timestamp.now(tz="UTC").strftime("%Y%m%d-%H%M")
    for code, country in cfg.countries.items():
        table = feats.build(load_processed(cfg, code), code, country.timezone)
        days = table.index[table["load"].notna()].floor("D").unique()
        test_days = days[len(days) - cfg.test_days :] if cfg.test_days else days[:0]
        fit_days = days[: len(days) - len(test_days)]
        train_days, valid_days = fit_days[: -cfg.validation_days], fit_days[-cfg.validation_days :]

        out = MODELS_DIR / run / code
        out.mkdir(parents=True, exist_ok=True)
        card = {
            "run": run,
            "country": code,
            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "train_period": _period(train_days),
            "validation_period": _period(valid_days),
            "test_period": _period(test_days),
            "git_sha": _git_sha(),
            "params": cfg.params,
            "validation": {},
            "curves": {},
        }
        tested = []
        for name in cfg.models:
            log.info("training %s | %s | %s", run, code, name)
            model = MODELS[name](cfg)
            model.fit(table, train_days, valid_days)
            pred = model.predict(table, valid_days)
            actual = table["load"].reindex(pred.index)
            ok = actual.notna() & pred[qcol(0.5)].notna()
            y, p = actual[ok].to_numpy(), pred.loc[ok, qcol(0.5)].to_numpy()
            card["validation"][name] = {"MAE": metrics.mae(y, p), "MAPE": metrics.mape(y, p)}
            if model.curve:
                card["curves"][name] = model.curve
            if len(test_days):
                tested.append(model.predict(table, test_days).assign(model=name))
            with open(out / f"{name}.pkl", "wb") as f:
                pickle.dump(model, f)
        if tested:
            pred = pd.concat(tested)
            pred["actual"] = table["load"].reindex(pred.index).to_numpy()
            card["test"] = scores(pred, cfg.quantiles)
        (out / "card.json").write_text(json.dumps(card, indent=2))
    log.info("trained run %s; upload it with: loadcast publish --run %s", run, run)
    return run


def scores(pred: pd.DataFrame, quantiles: list[float]) -> list[dict]:
    """metrics.summarise as JSON-ready records (one per model)."""
    table = metrics.summarise(pred, quantiles).reset_index()
    return json.loads(table.to_json(orient="records"))


def _period(days: pd.DatetimeIndex) -> list[str]:
    return [str(days[0].date()), str(days[-1].date())] if len(days) else []


def publish(run: str) -> None:
    """Upload models/{run}/ as the GitHub release `models-{run}` (needs the gh CLI).

    Publishing a run again (e.g. after `loadcast backtest --run`) only replaces its
    cards: the models of a release never change.
    """
    cards = [json.loads(p.read_text()) for p in sorted((MODELS_DIR / run).glob("*/card.json"))]
    if not cards:
        raise SystemExit(f"No trained models under {MODELS_DIR / run}")
    # The cards also go up on their own, so the dashboard can describe every run
    # without downloading its models.
    card_file = MODELS_DIR / f"{run}-cards.json"
    card_file.write_text(json.dumps(cards))
    tag, notes = f"models-{run}", _release_notes(cards)

    exists = subprocess.run(["gh", "release", "view", tag], capture_output=True).returncode == 0
    if exists:
        subprocess.run(["gh", "release", "upload", tag, str(card_file), "--clobber"], check=True)
        subprocess.run(["gh", "release", "edit", tag, "--notes", notes], check=True)
        return
    archive = MODELS_DIR / f"{run}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(MODELS_DIR / run, arcname=run)
    subprocess.run(
        ["gh", "release", "create", tag, str(archive), str(card_file), "--latest=false"]
        + ["--title", f"Models {run}", "--notes", notes],
        check=True,
    )


def _release_notes(cards: list[dict]) -> str:
    """MAPE (%) per country and model on the held-out test days (validation if none)."""
    first = cards[0]
    held_out = "test" in first
    models = [m for m in first["validation"] if m != "tso" or held_out]
    lines = [
        f"MAPE (%) on the {'held-out test' if held_out else 'validation'} days "
        f"{' → '.join(first['test_period' if held_out else 'validation_period'])}.",
        "",
        "| Country | " + " | ".join(models) + " |",
        "|---" * (len(models) + 1) + "|",
    ]
    for card in cards:
        if held_out:
            mape = {r["model"]: r["MAPE (%)"] for r in card["test"]}
        else:
            mape = {m: v["MAPE"] for m, v in card["validation"].items()}
        cells = [f"{mape[m]:.2f}" if mape.get(m) is not None else "–" for m in models]
        lines.append(f"| {card['country']} | " + " | ".join(cells) + " |")
    if "backtest" in first:
        lines += [
            "",
            f"Backtest on {first['backtest']['test_years']}: see the dashboard.",
        ]
    lines += ["", f"Trained on {' → '.join(first['train_period'])} at commit {first['git_sha']}."]
    lines += ["Add it to live.yaml to forecast with it."]
    return "\n".join(lines)


def live_models() -> list[tuple[str, str]]:
    """(run, model) pairs that make the daily forecast, as listed in live.yaml."""
    entries = yaml.safe_load(LIVE_FILE.read_text(encoding="utf-8"))["live"] or []
    return [(str(e["run"]), model) for e in entries for model in e["models"]]


def fetch() -> None:
    """Download the models of every run in live.yaml that is not under models/ yet, and
    record the latest card of every published run in the history."""
    MODELS_DIR.mkdir(exist_ok=True)
    for run in dict.fromkeys(run for run, _ in live_models()):
        if (MODELS_DIR / run).exists():
            continue
        log.info("downloading models-%s", run)
        _gh_download(run, f"{run}.tar.gz")
        with tarfile.open(MODELS_DIR / f"{run}.tar.gz") as tar:
            tar.extractall(MODELS_DIR, filter="data")

    tags = subprocess.run(
        ["gh", "release", "list", "--limit", "1000", "--json", "tagName", "--jq", ".[].tagName"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    # Always refreshed: publishing a run again (after a backtest) updates its cards.
    for run in (t.removeprefix("models-") for t in tags if t.startswith("models-")):
        _gh_download(run, f"{run}-cards.json")
        for card in json.loads((MODELS_DIR / f"{run}-cards.json").read_text()):
            _save_card(card, overwrite=True)


def _gh_download(run: str, asset: str) -> None:
    subprocess.run(
        ["gh", "release", "download", f"models-{run}", "--pattern", asset]
        + ["--dir", str(MODELS_DIR), "--clobber"],
        check=True,
    )


def forecast(cfg: Config, now: pd.Timestamp | None = None) -> None:
    """Forecast tomorrow (UTC) for every country and record results in HISTORY_DIR."""
    started = time.time()
    now = now or pd.Timestamp.now(tz="UTC")
    today = now.normalize()
    issue = today + pd.Timedelta(hours=cfg.issue_hour_utc)
    target = today + pd.Timedelta("1D")
    pairs = live_models()
    if not pairs:
        log.warning("live.yaml lists no runs: nothing to forecast")
        return
    status = {}

    for code in cfg.countries:
        try:
            path = HISTORY_DIR / "forecasts" / code / f"{target.date()}.csv"
            # A second job on the same day only adds the pairs that are new in live.yaml.
            done, todo = None, pairs
            if path.exists():
                done = pd.read_csv(path, parse_dates=["time"], dtype={"run": str})
                done = done.set_index("time")
                made = set(zip(done["run"], done["model"], strict=True))
                todo = [p for p in pairs if p not in made]
            # The weather is only needed to forecast; the actuals are refreshed anyway.
            frame = _recent_frame(cfg, code, target, with_weather=bool(todo))
            _write_actuals(code, frame, today)
            if todo:
                pred = predict_day(cfg, code, frame, issue, target, todo)
                pred = pred.assign(issued_at=now.isoformat(timespec="seconds"))
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.concat([done, pred]).to_csv(path)
                _record_cards(code, {run for run, _ in todo})
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
    cfg: Config,
    code: str,
    frame: pd.DataFrame,
    issue: pd.Timestamp,
    target: pd.Timestamp,
    pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Forecasts of the (run, model) `pairs` for `target`, using only data before `issue`."""
    frame = frame.copy()
    frame.loc[frame.index >= issue, "load"] = float("nan")  # enforce the information set
    frame["load"] = _patch_publication_delay(frame["load"], issue)
    table = feats.build(frame, code, cfg.countries[code].timezone)

    preds = []
    for run, name in pairs:
        path = MODELS_DIR / run / code / f"{name}.pkl"
        if not path.exists():
            log.warning("%s: run %s has no %s model, skipping", code, run, name)
            continue
        with open(path, "rb") as f:
            model = pickle.load(f)
        preds.append(model.predict(table, pd.DatetimeIndex([target])).assign(model=name, run=run))
    if not preds:
        raise ValueError(f"no trained model for {code} in live.yaml")
    return pd.concat(preds).rename_axis("time")


def _recent_frame(
    cfg: Config, code: str, target: pd.Timestamp, with_weather: bool = True
) -> pd.DataFrame:
    start = target - pd.Timedelta(days=PAST_DAYS)
    end = target + pd.Timedelta("1D")
    load = entsoe.fetch(code, start, end)
    if not with_weather:
        return load
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
    """Load and TSO forecast from tomorrow (TSO only: its day-ahead forecast is out) and
    today (the hours published so far) back to ACTUALS_DAYS ago. Every file is rewritten
    on later days until it is complete and no longer revised."""
    for k in range(-1, ACTUALS_DAYS + 1):
        day = today - pd.Timedelta(days=k)
        rows = frame.loc[day : day + pd.Timedelta("23h"), ["load", "tso_forecast"]]
        if rows.notna().any().any():
            path = HISTORY_DIR / "actuals" / code / f"{day.date()}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            rows.to_csv(path)


def _record_cards(code: str, runs: set[str]) -> None:
    """Keep the card of every run that forecasts, so the dashboard can describe it later."""
    for run in runs:
        src = MODELS_DIR / run / code / "card.json"
        if src.exists():
            _save_card(json.loads(src.read_text()))


def _save_card(card: dict, overwrite: bool = False) -> None:
    path = HISTORY_DIR / "cards" / card["run"] / f"{card['country']}.json"
    if overwrite or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(card, indent=2))


def _git_sha() -> str:
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
