"""A small, explicit hyper-parameter search on validation days.

`loadcast tune --country ES --model xgboost --grid xgboost.max_depth=6,8 ...` fits
every candidate of the grid on the same train/validation split as a training run
and prints the validation scores, best candidate first. The last `test_days` are
held out and never touched (SPEC §7), so the winner can be trained honestly with
`make train PARAMS=...` afterwards. Nothing is written under models/ or results/.
"""

from __future__ import annotations

import itertools
import logging

import pandas as pd
import yaml

from loadcast import features as feats
from loadcast import metrics
from loadcast.config import Config
from loadcast.data.dataset import load_processed
from loadcast.models import MODELS
from loadcast.models.base import qcol

log = logging.getLogger(__name__)


def parse_grid(specs: list[str] | None) -> dict[str, list]:
    """`["xgboost.max_depth=6,8"]` -> `{"xgboost.max_depth": [6, 8]}`; yaml scalars."""
    grid = {}
    for spec in specs or []:
        key, sep, raw = spec.partition("=")
        if not sep or "." not in key or raw.strip() == "":
            raise SystemExit(f"--grid expects <section>.<name>=v1,v2,..., got {spec!r}")
        grid[key.strip()] = [yaml.safe_load(v) for v in raw.split(",") if v.strip() != ""]
    return grid


def candidates(grid: dict[str, list]) -> list[dict]:
    """The cartesian product of the grid; an empty grid yields the defaults."""
    keys = list(grid)
    if not keys:
        return [{}]
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def run(cfg: Config, grid_specs: list[str] | None) -> pd.DataFrame:
    """Score every candidate on the validation days; returns the table it prints."""
    grid = parse_grid(grid_specs)
    if not grid:
        raise SystemExit("tune needs at least one --grid section.name=v1,v2")
    cands = candidates(grid)
    log.info("%d candidates from %s", len(cands), grid)
    if len(cfg.models) > 1:
        log.info("tuning models %s; add --model to tune a single one", cfg.models)
    rows = []
    for code, country in cfg.countries.items():
        table = feats.build(load_processed(cfg, code), code, country.timezone)
        days = table.index[table["load"].notna()].floor("D").unique()
        # Same split as a training run: the last test_days stay out of selection.
        fit_days = days[: len(days) - cfg.test_days]
        train_days, valid_days = fit_days[: -cfg.validation_days], fit_days[-cfg.validation_days :]
        for name in cfg.models:
            for i, candidate in enumerate(cands, 1):
                log.info("tune %s | %s | %d/%d %s", code, name, i, len(cands), candidate)
                model = MODELS[name](cfg.with_params(candidate))
                model.fit(table, train_days, valid_days)
                pred = model.predict(table, valid_days)
                actual = table["load"].reindex(pred.index)
                ok = actual.notna() & pred[qcol(0.5)].notna()
                y, p = actual[ok].to_numpy(), pred.loc[ok, qcol(0.5)].to_numpy()
                rows.append(
                    {
                        "country": code,
                        "model": name,
                        **candidate,
                        "MAE": metrics.mae(y, p),
                        "MAPE": metrics.mape(y, p),
                    }
                )
    results = pd.DataFrame(rows)
    for (code, name), grp in results.groupby(["country", "model"], sort=True):
        grp = grp.sort_values("MAPE")
        print(f"\n{code} | {name}")
        print(grp.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
        best = grp.iloc[0]
        print("best: " + " ".join(f"{c}={best[c]}" for c in grid) + f"  (MAPE {best['MAPE']:.2f}%)")
    return results
