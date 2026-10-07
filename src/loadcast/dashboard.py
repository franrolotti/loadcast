"""Build the static dashboard: site/index.html + site/data.json from the live history.

data.json holds the raw hourly series (hours since the Unix epoch, MW) and the page
computes every score in the browser, so the visitor can pick runs, windows and days.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from loadcast.config import HISTORY_DIR, Config
from loadcast.live import live_models

TEMPLATE = Path(__file__).parent / "dashboard.html"
RESULTS = Path("results")
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _read_csvs(pattern: str) -> pd.DataFrame:
    files = sorted(HISTORY_DIR.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = [
        pd.read_csv(f, parse_dates=["time"], dtype={"run": str}).assign(country=f.parent.name)
        for f in files
    ]
    return pd.concat(frames, ignore_index=True)


def _rows(df: pd.DataFrame, cols: list[str]) -> list[list]:
    """[[hour, *cols], ...] with MW rounded to integers and missing values as null."""
    hours = ((df["time"] - EPOCH) // pd.Timedelta("1h")).astype(int)
    values = df.reindex(columns=cols).astype(float).round(0)
    return [
        [h, *(None if np.isnan(v) else v for v in vals)]
        for h, vals in zip(hours, values.to_numpy(), strict=True)
    ]


def _country_block(code: str, forecasts: pd.DataFrame, actuals: pd.DataFrame) -> dict:
    block: dict = {"code": code, "actual": [], "series": []}
    if not actuals.empty:
        act = actuals[actuals["country"] == code].sort_values("time")
        block["actual"] = _rows(act, ["load", "tso_forecast"])
    if not forecasts.empty:
        fc = forecasts[(forecasts["country"] == code) & (forecasts["model"] != "tso")]
        for (run, model), rows in fc.sort_values("time").groupby(["run", "model"]):
            block["series"].append(
                {"model": model, "run": run, "rows": _rows(rows, ["q0.1", "q0.5", "q0.9"])}
            )
    return block


def build(cfg: Config, out: Path = Path("site")) -> Path:
    forecasts = _read_csvs("forecasts/*/*.csv")
    actuals = _read_csvs("actuals/*/*.csv")
    data: dict = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "models": cfg.models,
        "live": [{"run": run, "model": model} for run, model in live_models()],
        "countries": [_country_block(code, forecasts, actuals) for code in cfg.countries],
        "jobs": [
            json.loads(p.read_text())
            for p in sorted((HISTORY_DIR / "runs").glob("*.json"), reverse=True)[:60]
        ],
        "cards": [json.loads(p.read_text()) for p in sorted(HISTORY_DIR.glob("cards/*/*.json"))],
        "backtest": {},
    }
    for code in cfg.countries:
        metrics_file = RESULTS / f"metrics_{code}.csv"
        if metrics_file.exists():
            table = pd.read_csv(metrics_file).replace({np.nan: None})
            data["backtest"][code] = table.to_dict(orient="records")

    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    shutil.copy(TEMPLATE, out / "index.html")
    return out
