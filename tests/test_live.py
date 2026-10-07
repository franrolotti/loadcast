"""The operational loop end to end: train -> daily forecasts -> dashboard data."""

import json

import pandas as pd
import pytest
import yaml

from loadcast import dashboard, live
from tests.conftest import synthetic_country


@pytest.fixture(scope="module")
def operated(cfg, tmp_path_factory):
    root = tmp_path_factory.mktemp("ops")
    data = synthetic_country("2021-01-01", "2022-08-31")
    cfg = type(cfg)(**{**cfg.__dict__, "countries": {"ES": cfg.countries["ES"]}})
    patch = pytest.MonkeyPatch()
    patch.setattr(live, "MODELS_DIR", root / "models")
    patch.setattr(live, "HISTORY_DIR", root / "history")
    patch.setattr(live, "LIVE_FILE", root / "live.yaml")
    patch.setattr(dashboard, "HISTORY_DIR", root / "history")
    patch.setattr(live, "load_processed", lambda cfg, code: data[:"2022-06-30"])
    patch.setattr(
        live,
        "_recent_frame",
        lambda cfg, code, target: data[target - pd.Timedelta("21D") : target + pd.Timedelta("23h")],
    )

    # Run "old" forecasts alone for five days, then a second run joins it as a challenger.
    old = live.train(cfg, run="old")
    new = live.train(cfg, run="new")
    models = [m for m in cfg.models if m != "tso"]
    live_file = {"live": [{"run": old, "models": models}]}
    for day in pd.date_range("2022-07-01", "2022-07-10", tz="UTC"):
        if day == pd.Timestamp("2022-07-06", tz="UTC"):
            live_file["live"].append({"run": new, "models": ["xgboost"]})
        (root / "live.yaml").write_text(yaml.safe_dump(live_file))
        live.forecast(cfg, now=day + pd.Timedelta("9h15min"))
    site = dashboard.build(cfg, out=root / "site")
    yield root, models, json.loads((site / "data.json").read_text())
    patch.undo()


def test_one_forecast_file_per_day_with_every_live_model(operated):
    root, models, _ = operated
    files = sorted((root / "history" / "forecasts" / "ES").glob("*.csv"))
    assert [f.stem for f in files][:2] == ["2022-07-02", "2022-07-03"] and len(files) == 10
    first, last = pd.read_csv(files[0]), pd.read_csv(files[-1])
    assert set(first["model"]) == set(models) and set(first["run"]) == {"old"}
    assert len(last) == 24 * (len(models) + 1)
    assert last.loc[last["model"] == "transformer", "q0.5"].notna().all()


def test_rerun_only_adds_new_pairs(operated, cfg):
    root, models, _ = operated
    path = root / "history" / "forecasts" / "ES" / "2022-07-11.csv"
    before = pd.read_csv(path)
    live.forecast(
        type(cfg)(**{**cfg.__dict__, "countries": {"ES": cfg.countries["ES"]}}),
        now=pd.Timestamp("2022-07-10 10:00", tz="UTC"),
    )
    pd.testing.assert_frame_equal(pd.read_csv(path), before)


def test_dashboard_has_every_series_and_card(operated):
    _, models, data = operated
    es = data["countries"][0]
    keys = {(s["run"], s["model"]) for s in es["series"]}
    assert keys == {("old", m) for m in models} | {("new", "xgboost")}
    new = next(s for s in es["series"] if s["run"] == "new")
    assert len(new["rows"]) == 24 * 5 and len(new["rows"][0]) == 4
    assert es["actual"] and all(len(r) == 3 for r in es["actual"])
    assert {(c["run"], c["country"]) for c in data["cards"]} == {("old", "ES"), ("new", "ES")}
    assert data["cards"][0]["validation"]["xgboost"]["MAPE"] > 0
    curves = data["cards"][0]["curves"]
    assert set(curves) == {"xgboost", "lstm", "transformer"}
    for curve in curves.values():
        assert len(curve["train"]) == len(curve["valid"]) > 0 and curve["best"] >= 0
    assert len(data["jobs"]) == 10 and data["jobs"][0]["status"] == {"ES": "ok"}
