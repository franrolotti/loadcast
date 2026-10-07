"""Typed access to config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

# Operational artefacts (see loadcast.live): trained models and the live record,
# which the GitHub workflows keep on a release and on the `data` branch.
MODELS_DIR = Path("models")
HISTORY_DIR = Path("history")


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    pop: float


@dataclass(frozen=True)
class Country:
    code: str
    timezone: str
    cities: list[City]


@dataclass(frozen=True)
class Config:
    start: str
    end: str
    raw_dir: Path
    processed_dir: Path
    issue_hour_utc: int
    horizon_hours: int
    lookback_hours: int
    quantiles: list[float]
    test_years: list[int]
    validation_days: int
    models: list[str]
    countries: dict[str, Country]
    params: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> Config:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        countries = {
            code: Country(code, c["timezone"], [City(**city) for city in c["cities"]])
            for code, c in raw["countries"].items()
        }
        return cls(
            start=raw["data"]["start"],
            end=raw["data"]["end"],
            raw_dir=Path(raw["data"]["raw_dir"]),
            processed_dir=Path(raw["data"]["processed_dir"]),
            issue_hour_utc=raw["forecast"]["issue_hour_utc"],
            horizon_hours=raw["forecast"]["horizon_hours"],
            lookback_hours=raw["forecast"]["lookback_hours"],
            quantiles=raw["forecast"]["quantiles"],
            test_years=raw["backtest"]["test_years"],
            validation_days=raw["backtest"]["validation_days"],
            models=raw["models"],
            countries=countries,
            params={k: raw.get(k, {}) for k in ("xgboost", "torch", "lstm", "transformer")},
        )

    def with_end(self, end: str | None) -> Config:
        """`end="yesterday"` lets the operational jobs use everything up to now."""
        if end == "yesterday":
            end = str((datetime.now(UTC) - timedelta(days=1)).date())
        return replace(self, end=end) if end else self
