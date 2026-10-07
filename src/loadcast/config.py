"""Typed access to config.yaml."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

# Operational artefacts (see loadcast.live): trained models (one release per training
# run), the live record (the `data` branch) and the runs that forecast every day.
MODELS_DIR = Path("models")
HISTORY_DIR = Path("history")
LIVE_FILE = Path("live.yaml")


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
    test_days: int
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
            test_days=raw["training"]["test_days"],
            models=raw["models"],
            countries=countries,
            params={k: raw.get(k, {}) for k in ("xgboost", "torch", "lstm", "transformer")},
        )

    def with_end(self, end: str | None) -> Config:
        """`end="yesterday"` lets the operational jobs use everything up to now."""
        if end == "yesterday":
            end = str((datetime.now(UTC) - timedelta(days=1)).date())
        return replace(self, end=end) if end else self

    def with_params(self, overrides: list[str] | dict[str, Any] | None) -> Config:
        """Override hyper-parameters: `--param xgboost.max_depth=6` or, from `tune`,
        {"xgboost.max_depth": 6}. Unknown sections and names are rejected so that
        `config.yaml` stays the reference; a run's card records what was used."""
        params = {section: dict(values) for section, values in self.params.items()}
        items = overrides.items() if isinstance(overrides, dict) else _parse_params(overrides)
        for key, value in items:
            section, _, name = key.strip().partition(".")
            if not name or section not in params or name not in params[section]:
                known = ", ".join(sorted(params))
                raise SystemExit(f"unknown hyper-parameter {key!r} (sections: {known})")
            params[section][name] = value
        return replace(self, params=params)


def _parse_params(specs: list[str] | None) -> Iterator[tuple[str, Any]]:
    """`["xgboost.max_depth=6"]` -> `[("xgboost.max_depth", 6)]`; yaml scalars."""
    for spec in specs or []:
        key, sep, raw = spec.partition("=")
        if not sep or "" in (key.strip(), raw.strip()):
            raise SystemExit(f"--param expects <section>.<name>=<value>, got {spec!r}")
        yield key.strip(), yaml.safe_load(raw)
