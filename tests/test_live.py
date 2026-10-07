"""The operational loop end to end: train -> daily forecasts -> dashboard data."""

import json

import pandas as pd
import pytest

from loadcast import dashboard, live
from tests.conftest import synthetic_country


@pytest.fixture(scope="module")
def operated(cfg, tmp_path_factory):
    root = tmp_path_factory.mktemp("ops")
    data = synthetic_country("2021-01-01", "2022-08-31")
    cfg = type(cfg)(**{**cfg.__dict__, "countries": {"ES": cfg.countries["ES"]}})
    patch = pytest.MonkeyPatch()
    for module in (live, dashboard):
        patch.setattr(module, "MODELS_DIR", root / "models")
        patch.setattr(module, "HISTORY_DIR", root / "history")
    patch.setattr(live, "load_processed", lambda cfg, code: data[:"2022-06-30"])
    patch.setattr(
        live,
        "_recent_frame",
        lambda cfg, code, target: data[target - pd.Timedelta("21D") : target + pd.Timedelta("23h")],
    )

    live.train(cfg)
    for day in pd.date_range("2022-07-01", "2022-07-10", tz="UTC"):
        live.forecast(cfg, now=day + pd.Timedelta("9h15min"))
    site = dashboard.build(cfg, out=root / "site")
    yield root, json.loads((site / "data.json").read_text())
    patch.undo()


def test_one_forecast_file_per_day_with_all_models(operated, cfg):
    root, _ = operated
    files = sorted((root / "history" / "forecasts" / "ES").glob("*.csv"))
    assert [f.stem for f in files][:2] == ["2022-07-02", "2022-07-03"] and len(files) == 10
    pred = pd.read_csv(files[0])
    assert set(pred["model"]) == set(cfg.models)
    assert len(pred) == 24 * len(cfg.models)
    assert pred.loc[pred["model"] == "transformer", "q0.5"].notna().all()


def test_dashboard_scores_past_forecasts(operated, cfg):
    _, data = operated
    es = data["countries"][0]
    assert es["latest"]["day"] == "2022-07-11"
    board = {row["model"]: row for row in es["leaderboard"]}
    assert set(board) == set(cfg.models)
    assert board["tso"]["coverage"] is None and board["xgboost"]["coverage"] is not None
    assert len(data["runs"]) == 10 and data["runs"][0]["status"] == {"ES": "ok"}
    assert data["model_cards"][0]["validation"]["xgboost"]["MAPE"] > 0
