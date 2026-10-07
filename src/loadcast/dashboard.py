"""Build the static dashboard: site/index.html + site/data.json from the live history."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from loadcast.config import HISTORY_DIR, MODELS_DIR, Config

TEMPLATE = Path(__file__).parent / "dashboard.html"
RESULTS = Path("results")
RECENT_DAYS = 14
LEADERBOARD_DAYS = 30


def _read_csvs(pattern: str) -> pd.DataFrame:
    files = sorted(HISTORY_DIR.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f, parse_dates=["time"]).assign(country=f.parent.name) for f in files]
    return pd.concat(frames, ignore_index=True)


def _only(df: pd.DataFrame, code: str) -> pd.DataFrame:
    return df[df["country"] == code] if not df.empty else df


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    for col in df.select_dtypes(include=["datetimetz", "datetime"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%MZ")
    return json.loads(df.replace({np.nan: None}).to_json(orient="records", double_precision=3))


def _scored(forecasts: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Hourly forecasts joined with outcomes. The TSO forecast is taken from the
    actuals files, which keep being refreshed after publication."""
    tso = actuals[["country", "time", "tso_forecast"]].rename(columns={"tso_forecast": "q0.5"})
    tso["model"] = "tso"
    fc = pd.concat([forecasts[forecasts["model"] != "tso"], tso], ignore_index=True)
    out = fc.merge(actuals[["country", "time", "load"]], on=["country", "time"])
    out = out.dropna(subset=["load", "q0.5"])
    out["ape"] = (out["load"] - out["q0.5"]).abs() / out["load"] * 100
    out["abs_err"] = (out["load"] - out["q0.5"]).abs()
    if "q0.1" in out:
        out["covered"] = np.where(
            out["q0.1"].notna(), (out["load"] >= out["q0.1"]) & (out["load"] <= out["q0.9"]), np.nan
        )
    out["day"] = out["time"].dt.floor("D")
    return out


def _country_block(code: str, fc: pd.DataFrame, act: pd.DataFrame, scored: pd.DataFrame) -> dict:
    block: dict = {"code": code}
    if not fc.empty:
        last_day = fc["time"].dt.floor("D").max()
        latest = fc[fc["time"].dt.floor("D") == last_day]
        tso = act[act["time"].dt.floor("D") == last_day]  # usually empty: D not observed yet
        block["latest"] = {
            "day": str(last_day.date()),
            "rows": _records(latest.drop(columns="country")),
        }
        if not tso.empty:
            block["latest"]["tso"] = _records(tso[["time", "tso_forecast"]])

    if not scored.empty:
        since = scored["day"].max() - pd.Timedelta(days=RECENT_DAYS - 1)
        recent = scored[scored["day"] >= since]
        block["recent"] = _records(recent[["time", "model", "q0.5", "load"]])

        daily = scored.groupby(["day", "model"])["ape"].mean().rename("mape").reset_index()
        block["daily_mape"] = _records(daily)

        since = scored["day"].max() - pd.Timedelta(days=LEADERBOARD_DAYS - 1)
        board = (
            scored[scored["day"] >= since]
            .groupby("model")
            .agg(
                days=("day", "nunique"),
                mae=("abs_err", "mean"),
                mape=("ape", "mean"),
                coverage=("covered", lambda c: 100 * c.mean() if c.notna().any() else np.nan),
            )
            .sort_values("mape")
            .reset_index()
        )
        block["leaderboard"] = _records(board)
    return block


def build(cfg: Config, out: Path = Path("site")) -> Path:
    forecasts = _read_csvs("forecasts/*/*.csv")
    actuals = _read_csvs("actuals/*/*.csv")
    scored = _scored(forecasts, actuals) if not forecasts.empty and not actuals.empty else None

    data: dict = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "models": cfg.models,
        "countries": [],
        "runs": [
            json.loads(p.read_text())
            for p in sorted((HISTORY_DIR / "runs").glob("*.json"), reverse=True)[:60]
        ],
        "model_cards": [json.loads(p.read_text()) for p in sorted(MODELS_DIR.glob("*/card.json"))],
        "backtest": {},
    }
    for code in cfg.countries:
        data["countries"].append(
            _country_block(
                code,
                _only(forecasts, code),
                _only(actuals, code),
                _only(scored, code) if scored is not None else pd.DataFrame(),
            )
        )
        metrics_file = RESULTS / f"metrics_{code}.csv"
        if metrics_file.exists():
            data["backtest"][code] = _records(pd.read_csv(metrics_file))

    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data))
    shutil.copy(TEMPLATE, out / "index.html")
    return out
