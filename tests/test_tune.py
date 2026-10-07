"""Hyper-parameter overrides and the small tune grid."""

import pytest

from loadcast import tune
from loadcast.config import Config
from tests.conftest import ROOT, synthetic_country


def test_with_params_overrides_config_values():
    cfg = Config.load(ROOT / "config.yaml")
    over = cfg.with_params(
        ["xgboost.max_depth=6", "torch.learning_rate=0.01", "transformer.patch_len=1"]
    )
    assert over.params["xgboost"]["max_depth"] == 6
    assert over.params["torch"]["learning_rate"] == 0.01
    assert over.params["transformer"]["patch_len"] == 1
    assert cfg.params["xgboost"]["max_depth"] == 8  # the base config is untouched


def test_with_params_rejects_unknown_keys():
    cfg = Config.load(ROOT / "config.yaml")
    with pytest.raises(SystemExit):
        cfg.with_params(["nosection.key=1"])
    with pytest.raises(SystemExit):
        cfg.with_params(["xgboost.not_a_key=1"])


def test_parse_grid_and_candidates():
    grid = tune.parse_grid(["xgboost.max_depth=6,8", "torch.learning_rate=0.03,0.05"])
    assert grid == {"xgboost.max_depth": [6, 8], "torch.learning_rate": [0.03, 0.05]}
    cands = tune.candidates(grid)
    assert len(cands) == 4
    assert {"xgboost.max_depth": 8, "torch.learning_rate": 0.05} in cands
    assert tune.candidates({}) == [{}]


def test_tune_scores_candidates_on_validation_days(cfg, monkeypatch):
    data = synthetic_country("2021-01-01", "2022-08-31")
    monkeypatch.setattr(tune, "load_processed", lambda cfg, code: data)
    es = type(cfg)(
        **{**cfg.__dict__, "models": ["xgboost"], "countries": {"ES": cfg.countries["ES"]}}
    )
    results = tune.run(es, ["xgboost.max_depth=6,8"])
    assert len(results) == 2
    assert set(results["country"]) == {"ES"}
    assert (results["MAPE"] > 0).all()
